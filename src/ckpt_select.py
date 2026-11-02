"""Choose the checkpoint the project ships by what it draws, not by its eps-MSE.

`train.py` ranks epochs by validation loss, which is the MSE on the noise the network has
to predict at a uniformly sampled diffusion step. That objective spends most of its budget
on the high-noise steps, where the right answer is close to the prior, while a pen path is
judged by where the last few steps put 512 points. The two rankings are not the same, and
on the 1600-style CPU run they disagree loudly: the best eps-MSE epoch is 10, which samples
a 4.9-stroke scribble at average point distance 0.83, while epoch 120 samples 20.2 strokes
at 0.61. Selecting on one number and reporting the other would be the quietest way to put
a figure in this project that does not belong to the artifact it describes.

So this script samples real paths from every checkpoint in `ckpt/`, scores them on the
validation styles with one shared sampler seed, and writes the winner to
`ckpt/policy_ship.pt` with the whole comparison attached. `train.py` calls it before the
test report, so the headline test number is measured on the weights that ship.

Validation decides. Test styles are only ever measured afterwards, because choosing on the
test set and reporting on the test set would empty the reported number of meaning.

    python src/ckpt_select.py [--val-n 24] [--test-n 32]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import data
import model as M
import train as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "ckpt")
OUT = os.environ.get("CAT_OUT") or os.path.join(ROOT, "out")
SHIP = M.SHIP          # one name for the shipped weights, defined in model.py


def candidates():
    """Every trained checkpoint in ckpt/, except the one this script writes."""
    if not os.path.isdir(CKPT):
        return []
    return sorted(f for f in os.listdir(CKPT)
                  if f.startswith("policy_") and f.endswith(".pt") and f != SHIP)


def sample_score(pol, goals, paths, device, n, seed):
    """Sampled-path metrics for one checkpoint on one set, as a plain dict."""
    rep, _, sec = T.evaluate(pol, goals, paths, device, n=n, seed=seed)
    return {k: float(v[0]) for k, v in rep.items()} | {"sec": round(sec, 1)}


def rank(device, val_n=24, seed=1234, verbose=True):
    """(ranking best-first, rows) over every checkpoint in ckpt/, by validation ade.

    Every candidate sees the same styles and the same sampler noise, so this ranks weights
    and not two different drawings.
    """
    names = candidates()
    if not names:
        raise SystemExit("no policy_*.pt in ckpt/ - train before selecting")
    vg, vp = T.load_or_build("val.npz", max(32, val_n), 100000)
    rows = {}
    for nm in names:
        pol, ck = M.load_policy(os.path.join(CKPT, nm), device)
        v = sample_score(pol, vg, vp, device, val_n, seed)
        rows[nm] = {"epoch": ck.get("epoch"), "val_loss": float(ck.get("val_loss", np.nan)),
                    "steps": pol.steps, "val": v}
        if verbose:
            print("        %-16s epoch %-5s eps-mse %.4f | val ade %.4f f1 %.3f "
                  "strokes %.1f/%.1f  (%.0fs)"
                  % (nm, ck.get("epoch"), rows[nm]["val_loss"], v["ade"], v["f1"],
                     v["strokes"], v["strokes_ref"], v["sec"]))
    order = sorted(rows, key=lambda n: (rows[n]["val"]["ade"], -rows[n]["val"]["f1"]))
    return order, rows


def ship(device, val_n=24, seed=1234, verbose=True):
    """Rank the checkpoints, write the winner to policy_ship.pt, and return its name."""
    order, rows = rank(device, val_n=val_n, seed=seed, verbose=verbose)
    pick = order[0]
    by_loss = min(rows, key=lambda n: rows[n]["val_loss"])
    if verbose:
        print("RANK    " + "  <  ".join("%s ade %.4f" % (n, rows[n]["val"]["ade"])
                                        for n in order))
        print("SHIP    %s   (lowest validation eps-MSE was %s%s)"
              % (pick, by_loss, "" if by_loss == pick else " - a different checkpoint"))
        if len(order) > 1:
            print("        validation ade spread over the candidates: %.4f units"
                  % (rows[order[-1]]["val"]["ade"] - rows[order[0]]["val"]["ade"]))

    src = torch.load(os.path.join(CKPT, pick), map_location=device, weights_only=False)
    torch.save({"state": src["state"], "args": src["args"], "epoch": src["epoch"],
                "val_loss": src["val_loss"], "source_ckpt": pick,
                "selected_by": "validation average point distance of sampled paths",
                "select_seed": int(seed), "n_val": int(val_n),
                "ranking": order, "comparison": rows}, os.path.join(CKPT, SHIP))

    # the shipped file has to be the weights that were measured: sample the first
    # validation goal from both with the same seed and demand the same path out of each.
    pol_src, _ = M.load_policy(os.path.join(CKPT, pick), device)
    pol_ship, ck_ship = M.load_policy(os.path.join(CKPT, SHIP), device)
    vg, _ = T.load_or_build("val.npz", max(32, val_n), 100000)
    g = torch.as_tensor(np.asarray(vg[:1], np.float32), device=device)
    assert torch.equal(pol_src.sample(g, k=data.K, seed=seed),
                       pol_ship.sample(g, k=data.K, seed=seed)), \
        "%s does not reproduce the sample it was chosen for" % SHIP
    assert ck_ship["source_ckpt"] == pick and ck_ship["epoch"] == src["epoch"]
    return pick, by_loss, order, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-n", type=int, default=24)
    ap.add_argument("--test-n", type=int, default=32, help="0 skips the test measurement")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    dev = torch.device(a.device)
    print("SELECT  %d candidates  val %d styles  sampler seed %d  %s"
          % (len(candidates()), a.val_n, a.seed, dev))
    t0 = time.time()
    pick, by_loss, order, rows = ship(dev, val_n=a.val_n, seed=a.seed)

    if a.test_n:
        tg, tp = T.load_or_build("test.npz", max(32, a.test_n), 200000)
        for nm in order:
            pol, _ = M.load_policy(os.path.join(CKPT, nm), dev)
            rows[nm]["test"] = sample_score(pol, tg, tp, dev, a.test_n, a.seed)
            print("TEST    %-16s ade %.4f f1 %.3f strokes %.1f/%.1f"
                  % (nm, rows[nm]["test"]["ade"], rows[nm]["test"]["f1"],
                     rows[nm]["test"]["strokes"], rows[nm]["test"]["strokes_ref"]))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "checkpoint_select.json"), "w") as f:
        json.dump({"device": str(dev), "torch": torch.__version__, "pick": pick,
                   "picked_by": "validation sampled average point distance",
                   "best_val_loss_ckpt": by_loss, "ship": SHIP, "seed": a.seed,
                   "val_n": a.val_n, "test_n": a.test_n, "ranking": order,
                   "candidates": rows, "sec": round(time.time() - t0, 1)}, f, indent=1)
    print("DONE    %s verified against %s  ->  out/checkpoint_select.json  (%.0fs)"
          % (SHIP, pick, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

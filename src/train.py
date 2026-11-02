"""Train the goal-conditioned path policy, then report what it can actually draw.

The split is by style seed, not by shuffled index: train draws seeds from 0 upward,
validation from 100000, test from 200000, so a test cat is a drawing that was never in
the training set in any form. Every number this prints is measured here, in this run,
and the log it writes is the only thing the report is allowed to quote.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import data
import metrics
import model as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "ckpt")
DS = os.path.join(ROOT, "data")
OUT = os.environ.get("CAT_OUT") or os.path.join(ROOT, "out")

# The seed ranges that define what is compared with what. They are named here and written
# into out/train_log.json with the run, because a report that says "validation is split by
# style seed from 100000" is quoting this file, not memory.
VAL_SEED0, TEST_SEED0, VAL_MSE_CAP = 100000, 200000, 256


def load_or_build(name, n, seed0, cache=True):
    """(goals, paths) for n styles starting at seed0, cached as one npz.

    The cache is keyed by the seed range, so a smaller request is answered from the
    front of a bigger file rather than rebuilding - but never from a file built at a
    different seed, which would silently train on what it calls validation data.
    """
    path = os.path.join(DS, name)
    if cache and os.path.exists(path):
        z = np.load(path)
        if int(z["seed0"]) != int(seed0):
            raise SystemExit("%s was built at seed %d, asked for %d" %
                             (name, int(z["seed0"]), int(seed0)))
        if z["goals"].shape[0] >= n:
            return z["goals"][:n], z["paths"][:n]
    os.makedirs(DS, exist_ok=True)
    g, p = data.build(n, seed0)
    if cache:
        np.savez_compressed(path, goals=g, paths=p, seed0=np.int64(seed0))
    return g, p


def to_signed(paths):
    """Pen 0/1 -> +-1, so the switch shares the position channels' scale."""
    s = np.array(paths, np.float32)
    s[..., 2] = data.pen_to_sign(s[..., 2])
    return s


def evaluate(pol, goals, paths, device, n=16, seed=0, batch=8):
    """Sample one path per goal and score it four independent ways."""
    rows, t0 = [], time.time()
    pol.model.eval()
    hi = min(n, len(goals))
    for i0 in range(0, hi, batch):
        sl = slice(i0, min(i0 + batch, hi))
        gi = torch.as_tensor(goals[sl], device=device)
        x = pol.sample(gi, k=data.K, seed=seed + i0).cpu().numpy()
        for j, pi in enumerate(x):
            rows.append(metrics.score(metrics.as_path(pi), goals[sl][j], paths[sl][j]))
    return metrics.report(rows), rows, time.time() - t0


def train(args):
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tg, tp = load_or_build("train.npz", args.samples, 0)
    vg, vp = load_or_build("val.npz", max(32, args.samples // 8), VAL_SEED0)
    print("DATA  train %d  val %d  path %dx3  goal %dx%d  device %s"
          % (len(tg), len(vg), data.K, data.GOAL, data.GOAL, dev))

    ts = torch.as_tensor(to_signed(tp), device=dev)
    tsg = torch.as_tensor(tg, device=dev)
    vs = torch.as_tensor(vp, device=dev)
    vsg = torch.as_tensor(vg, device=dev)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    net = M.UNet1D().to(dev)
    pol = M.DiffusionPath(net, steps=args.steps, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    per = max(args.batch, 1)
    nsteps = max(int(np.ceil(len(ts) / per)), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=nsteps * args.epochs, pct_start=0.1)
    log, best, best_ep = [], float("inf"), 0
    os.makedirs(CKPT, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        net.train()
        tot = 0.0
        for _ in range(nsteps):
            idx = torch.randint(0, len(ts), (per,), generator=gen).to(dev)
            loss = pol.loss(tsg[idx], ts[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        tr = tot / nsteps
        with torch.no_grad():
            vl = float(pol.loss(vsg[:VAL_MSE_CAP], vs[:VAL_MSE_CAP]))
        row = {"epoch": ep, "train_loss": tr, "val_loss": vl, "sec": round(time.time() - t0, 1)}
        if vl < best:
            best = vl
            best_ep = ep
            torch.save({"state": net.state_dict(), "args": vars(args), "epoch": ep,
                        "val_loss": vl}, os.path.join(CKPT, "policy_best.pt"))
        if ep % args.eval_every == 0 or ep == args.epochs:
            rep, _, dt = evaluate(pol, vg, vp, dev, n=args.eval_n, seed=ep)
            row["val_ade"] = rep["ade"][0]
            row["val_f1"] = rep["f1"][0]
            row["val_strokes"] = rep["strokes"][0]
            row["sample_sec"] = round(dt, 1)
            print("        val ade %.4f f1 %.3f strokes %.1f (%.0fs sampling)"
                  % (row["val_ade"], row["val_f1"], row["val_strokes"], dt))
        log.append(row)
        print("EPOCH %3d/%d  train %.4f  val %.4f  %.0fs" % (ep, args.epochs, tr, vl, row["sec"]))

    torch.save({"state": net.state_dict(), "args": vars(args), "epoch": args.epochs,
                "val_loss": log[-1]["val_loss"]}, os.path.join(CKPT, "policy_last.pt"))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "train_log.json"), "w") as f:
        json.dump({"args": vars(args), "torch": torch.__version__, "device": str(dev),
                   "n_params": sum(p.numel() for p in net.parameters()),
                   "split": {"train": len(tg), "val": len(vg),
                             "val_seed0": VAL_SEED0, "test_seed0": TEST_SEED0,
                             "val_mse_n": min(VAL_MSE_CAP, len(vg))},
                   "best_epoch": best_ep, "best_val_loss": best,
                   "epochs": log}, f, indent=1)
    # which of these weights is the project's policy is not decided here: the lowest
    # validation eps-MSE and the best drawing are different epochs, so the choice is made
    # by ckpt_select.py on sampled paths, and the test report reads what it picks.
    print("CHECKPOINT  policy_best.pt epoch %d (val mse %.4f), policy_last.pt epoch %d"
          % (best_ep, best, args.epochs))
    return pol, (tg, tp), (vg, vp), dev


def report(args, dev, pol, goals, paths, train_paths):
    """Test-set numbers, with the two predictors that ignore the goal beside them."""
    rep, rows, dt = evaluate(pol, goals, paths, dev, n=args.test_n, seed=999)
    base = metrics.baselines(train_paths)
    m = metrics.report([metrics.score(base["mean"], goals[i], paths[i])
                        for i in range(min(16, len(paths)))])
    s = metrics.report([metrics.score(base["static"], goals[i], paths[i])
                        for i in range(min(16, len(paths)))])
    print("TEST  %d goals  policy ade %.4f f1 %.3f strokes %.1f/%.1f  (%.0fs)"
          % (len(rows), rep["ade"][0], rep["f1"][0], rep["strokes"][0],
             rep["strokes_ref"][0], dt))
    print("      mean-path baseline   ade %.4f f1 %.3f" % (m["ade"][0], m["f1"][0]))
    print("      static-path baseline ade %.4f f1 %.3f" % (s["ade"][0], s["f1"][0]))
    out = {"test": {k: list(v) for k, v in rep.items()},
           "baseline_mean": {k: list(v) for k, v in m.items()},
           "baseline_static": {k: list(v) for k, v in s.items()},
           "n": len(rows), "device": str(dev), "sample_sec": round(dt, 1),
           # a generative policy is a distribution, so this number belongs to one draw
           # from it. the seed and the weights are written next to it or the number is
           # not reproducible, only repeatable.
           "seed": 999, "steps": pol.steps,
           "ckpt": getattr(pol, "ckpt_file", None)}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "policy_eval.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=1600)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-n", type=int, default=16)
    ap.add_argument("--test-n", type=int, default=64)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run: prove the plumbing")
    a = ap.parse_args()
    if a.smoke:
        a.samples, a.epochs, a.batch, a.steps = 48, 2, 16, 12
        a.eval_every, a.eval_n, a.test_n, a.device = 1, 4, 4, "cpu"
    t0 = time.time()
    pol, tr, va, dev = train(a)
    te_g, te_p = load_or_build("test.npz", max(32, a.samples // 8), 200000)
    if not a.smoke:
        # imported here, not at the top: ckpt_select imports this module for its data and
        # evaluation helpers, and two modules that import each other at module level is a
        # cycle that only shows up when somebody runs the second one first.
        import ckpt_select
        pick, by_loss, order, rows = ckpt_select.ship(dev, val_n=max(16, a.eval_n),
                                                      seed=1234)
        pol, ck = M.load_policy(os.path.join(CKPT, ckpt_select.SHIP), dev)
        print("REPORT    test numbers below are measured on %s (epoch %s)"
              % (ckpt_select.SHIP, ck.get("epoch")))
    else:
        print("REPORT    smoke run: no selection, the last epoch is what gets measured")
    report(a, dev, pol, te_g, te_p, tr[1])
    print("TRAIN DONE  %.0fs  torch %s  device %s" % (time.time() - t0, torch.__version__, dev))

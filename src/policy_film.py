"""Film the drawing a trained network makes from a picture of a cat.

Everything upstream hands the machine a reference path. This does not: the only input
is the 48x48 goal, the network proposes the 512-point pen program, and the same
planner, the same simulated plotter and the same calibrated renderer that judge the
hero artwork judge this one. The registration and centroid checks are not decoration -
if the film shows a cat the path did not draw, this script fails instead of rendering.

`--goal hero` conditions on the picture of the hero drawing, which is in distribution
but was never a training target. `--goal style N` conditions on any style in the
family, including the held-out test seeds.
"""
import os
import sys

# mujoco binds its GL backend at import time, so this has to run before that import:
# windows draws through wgl, linux (the GPU box) has only egl.
os.environ.setdefault("MUJOCO_GL", "wgl" if sys.platform == "win32" else "egl")

import argparse
import json
import time

import numpy as np
import torch

import art
import data
import metrics
import model as M
import planner
import render
import robot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "ckpt")
OUT = os.environ.get("CAT_OUT") or os.path.join(ROOT, "out")
# numbers go to out/, pictures go to design_out/ - the same rule render.py writes by, so
# a reader who opens one folder finds the same kind of thing in it as in the other.
DESIGN = os.environ.get("CAT_DESIGN") or os.path.join(ROOT, "design_out")


def hero_pair():
    """(goal, reference path, ordered strokes) for the exact hero artwork."""
    strokes = art.fit_canvas(art.cat_strokes())
    ordered, _ = planner.plan_strokes(strokes, spacing=0.01, first="head")
    goal = art.rasterize(ordered, size=data.GOAL, line_width=data.INK_LINE)
    return goal, data.tour_path(ordered), ordered


def load(device, name=None):
    """The checkpoint the films are made from.

    `policy_ship.pt` is the file ckpt_select.py chose by drawn quality on validation; a
    directory that has not been through selection falls back to the best validation
    eps-loss checkpoint, which is what training itself wrote.
    """
    if name is None:
        name = M.SHIP if os.path.exists(os.path.join(CKPT, M.SHIP)) else "policy_best.pt"
    return M.load_policy(os.path.join(CKPT, name), device)


def propose(pol, goal, seed, device):
    """One sampled path, in canvas units with the pen back in 0/1."""
    g = torch.as_tensor(np.asarray(goal, np.float32)[None], device=device)
    x = pol.sample(g, k=data.K, seed=seed).cpu().numpy()[0]
    return metrics.as_path(x)


def compare_sheet(goal, ref, pred, path):
    """Goal, the reference ink, and the network's ink, side by side."""
    from PIL import Image
    r = data.raster_of_path(ref)
    q = data.raster_of_path(pred)
    tiles = [np.asarray(goal), r, q]
    grid = np.concatenate(tiles, axis=1)
    img = Image.fromarray(((1.0 - grid) * 255).astype(np.uint8)).resize(
        (data.GOAL * 3 * 10, data.GOAL * 10), Image.NEAREST)
    os.makedirs(DESIGN, exist_ok=True)
    p = os.path.join(DESIGN, path)
    img.save(p)
    return p, float(np.mean(r)), float(np.mean(q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", default="hero", help="hero or style")
    ap.add_argument("--seed", type=int, default=0, help="style seed, or sampler seed")
    ap.add_argument("--sample-seed", type=int, default=None)
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint file; default policy_ship.pt, else policy_best.pt")
    ap.add_argument("--path", choices=("policy", "reference"), default="policy",
                    help="film the network's path, or the reference path it was asked for")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--film", action="store_true")
    ap.add_argument("--draft", action="store_true")
    a = ap.parse_args()

    dev = torch.device(a.device)
    pol, ck = load(dev, a.ckpt)
    print("POLICY  %s  epoch %s  val loss %.4f  steps %d  torch %s %s"
          % (ck["ckpt_file"], ck.get("epoch"), float(ck.get("val_loss", float("nan"))),
             pol.steps, torch.__version__, dev))

    if a.goal == "hero":
        goal, ref, _ = hero_pair()
        sample_seed = a.seed if a.sample_seed is None else a.sample_seed
    else:
        goal, ref, _ = data.style_pair(int(a.seed))
        sample_seed = int(a.seed) if a.sample_seed is None else a.sample_seed

    t0 = time.time()
    if a.path == "reference":
        # the machinery test: the physics, renderer and registration checks run on a path
        # no network produced, so a broken film is found on a toy model in minutes and not
        # after a GPU run. it is written to a separate log, because ade 0 here is a fact
        # about the reference, not a result.
        print("PATH    reference (the network is loaded but not asked to draw)")
        pred = np.asarray(ref, np.float32).copy()
    else:
        pred = propose(pol, goal, sample_seed, dev)
    reach = float(np.abs(pred[:, :2]).max())
    # the sheet is [-1,1] and the gantry cannot leave it, so an escaping point is not a
    # drawing choice but a projection. the number before the clip is what gets reported.
    pred[:, :2] = np.clip(pred[:, :2], -1.0, 1.0)
    print("SAMPLE  %d DDPM steps in %.1f s  path %dx3  pen down %.0f%%  reached %.3f u"
          % (pol.steps, time.time() - t0, pred.shape[0], 100.0 * (pred[:, 2] > 0.5).mean(),
             reach))

    s = metrics.score(pred, goal, ref)
    print("SCORE   ade %.4f  f1 %.3f  precision %.3f  recall %.3f  chamfer %.4f"
          % (s["ade"], s["f1"], s["precision"], s["recall"], s["chamfer"]))
    print("        strokes %d/%d  off-target %.4f  stray %.3f  pen agreement %.3f  ink %.2f u"
          % (s["strokes"], s["strokes_ref"], s["off_target"], s["stray_fraction"],
             s["pen"], s["ink_length"]))

    tag = "%s%s" % (a.goal, "_reference" if a.path == "reference" else "")
    sheet, ref_ink, pred_ink = compare_sheet(goal, ref, pred, "policy_sheet_%s.png" % tag)
    print("SHEET   %s  ink pixels goal %.4f reference %.4f policy %.4f"
          % (sheet, float(np.mean(goal)), ref_ink, pred_ink))

    strokes = data.strokes_from_path(pred)
    if len(strokes) < 2:
        raise SystemExit("the policy produced %d strokes - nothing to draw" % len(strokes))
    named = [("%02d" % i, xy) for i, xy in enumerate(strokes)]
    pr = planner.plan(named)
    rob = robot.Plotter()
    out = rob.run(pr)
    st = planner.stats(pr)
    print("PLAN    %d strokes %d frames  ink %.2f travel %.2f (%.0f%% lift)"
          % (st["strokes"], st["frames"], st["ink_length"], st["travel_length"],
             100.0 * st["travel_fraction"]))
    err = np.asarray(out["err"])[np.asarray(out["draw"], bool)]
    print("ROBOT   ink tracking rms %.5f max %.4f units  contact %.3f N"
          % (float(np.sqrt((err ** 2).mean())), float(err.max()),
             float(np.asarray(out["force"])[np.asarray(out["draw"], bool)].mean())))

    res = (1280, 720) if a.draft else (1920, 1080)
    scene = render.Scene(width=res[0], height=res[1])
    assert scene.calib_err_px < 6.0, "ink would land on the wrong part of the page"
    worst, rows, (skip_n, skip_len) = render.stroke_registration(scene, out, pr)
    assert rows, "no stroke the policy drew is long enough to lay a measurable dot"
    assert worst < 8.0, "the film shows strokes %.1f px from where the pen drew them" % worst
    render.paint_upto(scene, out, out["T"], render.LINE_W)
    scene.park()
    scene.apply_cam(scene._top_cam(scene.top_orient))
    scene.upload()
    dpx, npix = scene.ink_centroid_check(out)
    assert dpx < 12.0, "the finished page is %.1f px off where the pen says it is" % dpx
    print("RENDER  worst stroke %.2f px  centroid %.1f px over %.0f ink px"
          % (worst, dpx, npix))
    print("        %d strokes registered, %d too short to lay a measurable dot (%.4f u)"
          % (len(rows), skip_n, skip_len))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "policy_draw%s.json"
                           % ("_reference" if a.path == "reference" else "")), "w") as f:
        json.dump({"ckpt": ck["ckpt_file"], "ckpt_epoch": ck.get("epoch"), "goal": a.goal,
                   "path": a.path,
                   "goal_seed": int(a.seed), "sample_seed": int(sample_seed),
                   "device": str(dev), "ddpm_steps": pol.steps,
                   "path_reach_units": reach,
                   "metrics": {k: (float(v) if np.ndim(v) == 0 else v)
                               for k, v in s.items()},
                   "plan": st,
                   "robot_ink_err_rms": float(np.sqrt((err ** 2).mean())),
                   "registration_worst_px": float(worst),
                   "centroid_px": float(dpx)}, f, indent=1)

    if a.film:
        os.makedirs(DESIGN, exist_ok=True)
        mp4 = os.path.join(DESIGN, "policy_film_%s%s.mp4" % (tag, "_draft" if a.draft else ""))
        film = render.Film(scene, out, pr, title=[("THE  NETWORK'S  CAT", 46),
                                                  ("one goal picture in, one pen program out", 24)],
                           credits=[("%d DDPM steps  ade %.3f  ink F1 %.3f  %d strokes"
                                     % (pol.steps, s["ade"], s["f1"], s["strokes"]), 20),
                                    ("worst stroke %.2f px from where the pen drew it" % worst, 20)])
        n = film.build(mp4, on_frame=lambda shot, k: None if k % 100 else print("  %s %d" % (shot, k)))
        print("FILM %s  %d frames  %.0f s  rendered in %.1f s"
              % (mp4, n, n / film.fps, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

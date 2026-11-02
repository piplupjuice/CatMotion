"""The style family as data: a goal raster and the one pen path that lays it.

The policy's job is stated here in one sentence, so that the metric can be honest:
given the 48x48 picture of a cat, output the ordered pen path - 512 arc-length
samples of (x, y, pen) - that draws it. The path is the geometry `planner.plan_strokes`
produces and nothing else: no pen ramps, no settle holds, because `planner.plan`
adds those to whatever path it is handed, predicted or not.
"""
import os
import sys

import numpy as np

import art
import planner
from geometry import polyline_length, resample

K = 512               # points in a target path
GOAL = 48             # goal raster side
INK_LINE = 1.6        # raster line width, in pixels, for the goal image
PEN_SIGN = 2.0        # the pen switch is carried at +-1, so one scale fits all channels
MAX_RASTER_PTS = 4096  # ceiling on the points a scored path may be resampled to


def pen_to_sign(p):
    return p * PEN_SIGN - 1.0


def sign_to_pen(s):
    return np.clip((np.asarray(s, np.float64) + 1.0) / PEN_SIGN, 0.0, 1.0)


def tour_segments(ordered):
    """[(xy, pen)] with a straight pen-up run inserted between consecutive strokes."""
    segs = []
    for i, (_, xy) in enumerate(ordered):
        if i:
            a, b = ordered[i - 1][1][-1], xy[0]
            segs.append((np.array([a, b], np.float64), 0.0))
        segs.append((np.asarray(xy, np.float64), 1.0))
    return segs


def tour_path(ordered, k=K):
    """One fixed-length (k,3) path: x, y, pen, sampled evenly along the tour.

    Even arc-length sampling is what makes the target representable at all: the
    policy is scored against a path of exactly k points, so the target must not
    depend on how many waypoints the artwork happened to use.
    """
    segs = tour_segments(ordered)
    X = np.concatenate([s[0] for s in segs], axis=0)
    P = np.concatenate([np.full(len(s[0]), s[1]) for s in segs])
    seg = np.linalg.norm(np.diff(X, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    u = np.linspace(0.0, total, k)
    xy = np.column_stack([np.interp(u, cum, X[:, 0]), np.interp(u, cum, X[:, 1])])
    idx = np.clip(np.searchsorted(cum, u, side="right") - 1, 0, len(P) - 1)
    return np.column_stack([xy, P[idx]])


def style_pair(seed):
    """One (goal, path, meta) sample from the style family, reproducible by seed."""
    rng = np.random.default_rng(seed)
    strokes = art.fit_canvas(art.cat_strokes(art.sample_style(rng)))
    ordered, _ = planner.plan_strokes(strokes, spacing=0.01, first="head")
    goal = art.rasterize(ordered, size=GOAL, line_width=INK_LINE)
    return goal, tour_path(ordered), {
        "seed": int(seed),
        "strokes": [n for n, _ in ordered],
        "lengths": [float(polyline_length(xy)) for _, xy in ordered],
        "ink_length": float(sum(polyline_length(xy) for _, xy in ordered)),
    }


def _pair(arg):
    """Worker entry: (seed0, index) -> (goal, path). The seed travels in the argument,
    never in a module global, because a spawned child re-imports this file with its own
    globals and would otherwise silently build validation data out of training seeds."""
    seed0, i = arg
    g, p, _ = style_pair(seed0 + i)
    return g, p


def build(n, seed0=0, k=K, goal=GOAL, workers=None):
    """(n,goal,goal) float32 rasters and (n,k,3) float32 paths, in seed order.

    Building 8000 styles is the slowest CPU stage in the pipeline and the GPU has
    nothing to do while it runs, so the seeds are farmed across processes. The result
    is indexed by seed, not by arrival, which makes the array identical to the serial
    one - the self-test at the bottom of this file proves that.
    """
    if workers is None:
        workers = min(os.cpu_count() or 1, max(1, n))
    args = [(seed0, i) for i in range(n)]
    pairs = None
    if workers > 1 and n > 1:
        try:
            import multiprocessing as mp
            ctx = mp.get_context("spawn") if sys.platform == "win32" else mp.get_context()
            with ctx.Pool(workers) as pool:
                pairs = pool.map(_pair, args, chunksize=max(1, n // (workers * 8)))
        except Exception as e:
            print("data: parallel build unavailable (%s), falling back to one process" % e)
            pairs = None
    if pairs is None:
        pairs = [_pair(a) for a in args]
    gs = [g for g, _ in pairs]
    ps = [p for _, p in pairs]
    return np.stack(gs).astype(np.float32), np.stack(ps).astype(np.float32)


def strokes_from_path(path, min_pen=0.5):
    """Split a (k,3) path back into strokes wherever the pen lifts.

    This is the only place a predicted path becomes drawable: `planner.plan` needs
    strokes, and a prediction is a flat array, so the split has to be made by the
    pen channel the model was trained to output rather than by any extra oracle.
    """
    out, cur = [], []
    for xy, pen in zip(np.asarray(path, np.float64)[:, :2], path[:, 2]):
        if pen >= min_pen:
            cur.append(xy)
        elif len(cur) > 1:
            out.append(np.array(cur))
            cur = []
    if len(cur) > 1:
        out.append(np.array(cur))
    return out


def raster_of_path(path, size=GOAL, line_width=INK_LINE, spacing=0.004):
    """The picture a path actually lays, for comparing prediction with goal.

    `spacing` is the floor, not the answer: an untrained prediction wanders across the
    canvas hundreds of times, and resampling that at a fixed step is a hundred thousand
    points to splat. The step widens until the ink fits the budget, which costs a
    converged path nothing because it never reaches it.
    """
    strokes = strokes_from_path(path)
    total = sum(polyline_length(xy) for xy in strokes)
    step = max(spacing, total / MAX_RASTER_PTS) if total > 0 else spacing
    st = [(str(i), resample(xy, step)) for i, xy in enumerate(strokes)]
    return art.rasterize(st, size=size, line_width=line_width) if st else np.zeros(
        (size, size), np.float32)


def overlap(a, b, thresh=0.5):
    """Precision / recall / F1 of one ink map against another, in pixels."""
    A, B = np.asarray(a) > thresh, np.asarray(b) > thresh
    hit = float((A & B).sum())
    pa, pb = float(A.sum()), float(B.sum())
    prec = hit / pa if pa else 0.0
    rec = hit / pb if pb else 0.0
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return prec, rec, f1


if __name__ == "__main__":
    g, p, meta = style_pair(0)
    assert g.shape == (GOAL, GOAL) and p.shape == (K, 3), "%s %s" % (g.shape, p.shape)
    assert np.isfinite(g).all() and np.isfinite(p).all(), "non-finite sample"
    assert np.abs(p[:, :2]).max() <= 1.0, "path escapes the canvas"
    assert set(np.unique(p[:, 2])) <= {0.0, 1.0}, "pen is not a switch"
    assert p[:, 2].sum() > 0.5 * K, "most of the tour is lift, not ink"

    # the target must be a faithful stand-in for the drawing it is scored against:
    # resample the arc-length path back to an image and compare with the goal
    back = raster_of_path(p)
    prec, rec, f1 = overlap(g, back)
    assert f1 > 0.80, "the 512-point path no longer draws the cat: F1 %.3f" % f1

    # a K-point path samples the tour every `step` units, so a stroke shorter than
    # a few steps cannot survive it. the artwork has to live inside that budget or
    # the dataset holds targets no policy could ever be credited for drawing.
    step = float(polyline_length(p[:, :2]) / (K - 1))
    carry = [L for L in meta["lengths"] if L > 3.0 * step]
    n = len(strokes_from_path(p))
    assert len(carry) == len(meta["lengths"]), (
        "%d of %d strokes are under 3 path steps (%.4f u): the drawing is finer "
        "than the representation" % (len(meta["lengths"]) - len(carry),
                                     len(meta["lengths"]), step))
    assert n == len(carry), "path has %d strokes, planner ordered %d" % (n, len(carry))

    # every style in the family must survive the same round trip against its own
    # goal, or the dataset quietly contains samples no policy could be credited for
    rng = np.random.default_rng(11)
    seeds = [1000 + int(rng.integers(1 << 30)) for _ in range(12)]
    fam, thin = [], []
    for s in seeds:
        gi, pi, mi = style_pair(s)
        st = float(polyline_length(pi[:, :2]) / (K - 1))
        fam.append(overlap(gi, raster_of_path(pi))[2])
        thin.append(sum(1 for L in mi["lengths"] if L <= 3.0 * st))
    fam = np.array(fam)
    print("DATA PASS  goal %d^2  path %d x 3  ink %.2f u  strokes %d  step %.4f u" %
          (GOAL, K, meta["ink_length"], n, step))
    print("         goal raster line width %.1f px" % INK_LINE)
    print("         path->raster round trip F1 %.3f (family worst %.3f, median %.3f)"
          % (f1, fam.min(), float(np.median(fam))))
    print("         family styles %d, strokes below path resolution %d"
          % (len(seeds), int(np.sum(thin))))
    assert fam.min() > 0.70, "a family style cannot be drawn from its own path"
    assert not any(thin), "a family style is finer than the path can carry"

    # seeds are farmed across processes but results are collected by index, so the
    # array cannot depend on arrival order. this proves the parallel builder is the
    # serial one rather than trusting it, because a silently different validation set
    # would still train and still report a number.
    m = 6
    g1, p1 = build(m, seed0=777000, workers=1)
    g4, p4 = build(m, seed0=777000, workers=4)
    assert g1.shape == g4.shape and p1.shape == p4.shape, "shape differs by worker count"
    assert np.array_equal(g1, g4), "parallel build changed the rasters"
    assert np.array_equal(p1, p4), "parallel build changed the paths"
    print("         parallel build byte-identical to serial over %d seeds" % m)

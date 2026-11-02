"""What a predicted pen path is worth, measured four independent ways.

A single number lies about a drawing. Distance to the reference path says nothing about
whether the ink lands where the goal picture has ink; image overlap says nothing about
whether the pen ever lifted; and neither says anything about whether the machine can
follow the path at its own speed limits. So the policy is scored on all four, and the
report quotes them together or not at all.

Nothing here is learned: every function is pure geometry on arrays, so the same code
scores the reference path, a baseline, and the network.
"""
import numpy as np

import data
from geometry import polyline_length, resample

# the order-blind distance runs on a deterministic thinning of each path, and the size of
# that thinning is part of what the number means - so the self-test prints it.
CHAMFER_N = 256


def as_path(pred):
    """(K,3) model output -> (K,3) canvas path, pen in [0,1] and hard-switched."""
    p = np.asarray(pred, np.float64).copy()
    p[:, 2] = data.sign_to_pen(p[:, 2]) > 0.5
    return p


def _pairwise(a, b):
    d = a[:, None, :] - b[None, :, :]
    return np.sqrt((d * d).sum(-1))


def path_errors(pred, truth):
    """ADE and FDE in canvas units, point by point in path order.

    Order matters here on purpose: two paths can cover the same ink in a different
    order and score perfectly on Chamfer while being a different drawing program.
    """
    d = np.linalg.norm(np.asarray(pred)[:, :2] - np.asarray(truth)[:, :2], axis=1)
    return float(d.mean()), float(d[-1])


def _thin(P, n):
    """Every floor(len/n)-th point: a deterministic subsample, so a path scored
    against itself comes back exactly zero instead of a seed-dependent residue."""
    return P[::max(len(P) // n, 1)]


def chamfer(a, b, n=CHAMFER_N):
    """Symmetric nearest-neighbour distance, ignoring where the points sit in order.

    Subsampled because this runs once per test sample and the sample count is the
    honest part; 512 x 512 is fine for one pair and wasteful for a hundred.
    """
    A = _thin(np.asarray(a, np.float64)[:, :2], n)
    B = _thin(np.asarray(b, np.float64)[:, :2], n)
    D = _pairwise(A, B)
    return float(0.5 * (D.min(axis=1).mean() + D.min(axis=0).mean()))


def ink_metrics(pred, goal, size=data.GOAL):
    """Precision / recall / F1 of the ink the path lays against the goal picture."""
    return data.overlap(data.raster_of_path(pred, size=size), goal)


def pen_agreement(pred, truth, thresh=0.5):
    """Fraction of path samples where the predicted switch matches the reference."""
    return float(((np.asarray(pred)[:, 2] > thresh) ==
                  (np.asarray(truth)[:, 2] > thresh)).mean())


def structure(pred, truth, tol=0.02, max_pts=2048):
    """Stroke count, and how much of the predicted ink is not on the reference.

    `off_target` is the mean distance from each predicted pen-down point to the
    reference path: a policy can fill the page with ink and still score well on
    recall, and this is the number that catches it.

    The resample step is widened to keep the pairwise matrix inside `max_pts` per
    side. A reference cat is already under it, so the cap only ever binds on a
    prediction that wanders - and an unconstrained wander is a 50k x 50k matrix.
    """
    ps, ts = data.strokes_from_path(pred), data.strokes_from_path(truth)
    if not ps or not ts:
        return len(ps), len(ts), float("nan"), float("nan")

    def dense(strokes):
        total = sum(polyline_length(xy) for xy in strokes)
        step = max(tol, total / max_pts) if total > 0 else tol
        return np.concatenate([resample(xy, step) for xy in strokes])

    P, T = dense(ps), dense(ts)
    d = _pairwise(P[:, :2], T[:, :2]).min(axis=1)
    return len(ps), len(ts), float(d.mean()), float((d > 0.05).mean())


def ink_length(path):
    """Length of ink actually laid: summed inside each stroke, never across a lift."""
    return float(sum(polyline_length(xy) for xy in data.strokes_from_path(path)))


def score(pred, goal, truth, size=data.GOAL):
    """All of it at once, for one sample. `pred` is a canvas path, pen in 0/1."""
    p = np.asarray(pred, np.float64)
    ade, fde = path_errors(p, truth)
    prec, rec, f1 = ink_metrics(p, goal, size)
    ns, nt, off, stray = structure(p, truth)
    return {
        "ade": ade, "fde": fde, "chamfer": chamfer(p, truth),
        "precision": prec, "recall": rec, "f1": f1,
        "pen": pen_agreement(p, truth), "strokes": ns, "strokes_ref": nt,
        "off_target": off, "stray_fraction": stray, "ink_length": ink_length(p),
        # the sheet is [-1,1]: how far the path reached outside it is the difference
        # between an untidy drawing and one the plotter could not physically make
        "xy_max": float(np.abs(p[:, :2]).max()),
    }


def baselines(paths):
    """Two predictors a policy has to beat, from the training paths only.

    `mean` is the average reference path - what a regression head converges to when
    the target has many right answers. `static` replays one training path for every
    goal, i.e. a policy that learned to draw *a* cat and nothing else.
    """
    A = np.asarray(paths)
    mean = A.mean(axis=0)
    mean[:, 2] = mean[:, 2] > 0.5
    return {"mean": mean, "static": A[0].copy()}


def report(rows, names=("ade", "fde", "chamfer", "precision", "recall", "f1",
                        "pen", "off_target", "stray_fraction", "strokes", "strokes_ref",
                        "xy_max")):
    """Mean and worst over a list of score dicts, for the log."""
    out = {}
    for k in names:
        v = np.array([r[k] for r in rows], np.float64)
        out[k] = (float(np.nanmean(v)), float(np.nanmax(v)))
    return out


if __name__ == "__main__":
    g, p, meta = data.style_pair(0)
    signed = p.copy()
    signed[:, 2] = data.pen_to_sign(signed[:, 2])

    # a metric that cannot tell the reference from itself is not a metric
    s = score(p, g, p)
    assert s["ade"] < 1e-12 and s["fde"] < 1e-12, "self distance is not zero: %s" % s["ade"]
    assert s["chamfer"] < 1e-12, "self chamfer is not zero"
    assert s["f1"] > 0.98 and s["pen"] > 0.999, "self overlap %.3f pen %.3f" % (
        s["f1"], s["pen"])
    assert s["strokes"] == s["strokes_ref"], "%d vs %d strokes for the same path" % (
        s["strokes"], s["strokes_ref"])

    # and it has to move when the path moves
    rng = np.random.default_rng(3)
    jit = p + rng.normal(0.0, 0.02, p.shape)
    jit[:, 2] = p[:, 2]
    j = score(jit, g, p)
    assert j["ade"] > 0.01 and j["f1"] < s["f1"], "the metrics ignore a 2 cm jitter"

    # a blank page must score as no ink, not as a perfect drawing
    blank = np.zeros_like(p)
    b = score(blank, g, p)
    assert b["recall"] < 0.02 and b["strokes"] == 0, "an empty path scored ink: %s" % b["recall"]

    # the model's own encoding has to round-trip through as_path
    back = as_path(signed)
    assert np.allclose(back[:, :2], p[:, :2]) and np.array_equal(back[:, 2], p[:, 2]), \
        "as_path does not invert the pen encoding"

    # a predictor that ignores the goal has to score worse than the reference, or
    # these numbers are measuring the dataset rather than the policy
    mean_b = baselines(np.stack([p, jit]))["mean"]
    m = score(mean_b, g, p)
    assert m["f1"] < s["f1"] and m["ade"] > s["ade"], "the mean path is not worse"

    r = report([s, j, m])
    print("METRICS PASS  self ade %.6f f1 %.3f strokes %d | jitter ade %.4f f1 %.3f"
          % (s["ade"], s["f1"], s["strokes"], j["ade"], j["f1"]))
    print("        empty path recall %.3f strokes %d  mean-path f1 %.3f ade %.4f"
          % (b["recall"], b["strokes"], m["f1"], m["ade"]))
    print("        off-target self %.4f  stray %.3f  ink %.2f u"
          % (s["off_target"], s["stray_fraction"], s["ink_length"]))
    print("        chamfer subsample %d points per side" % CHAMFER_N)
    print("        report means %s" % ", ".join("%s=%.3f" % (k, v[0]) for k, v in r.items()))

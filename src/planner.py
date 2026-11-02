"""From a line drawing to a dense pen program the machine can actually follow.

Three defects in the previous attempt are what made its video read as noise, and
each has a concrete answer here:

* spacing - the path is resampled by arc length, so constant-speed playback lays
  down even ink instead of visible facets;
* ordering - strokes are visited by greedy nearest neighbour with free direction
  reversal, so the carriage does not drag itself across the drawing;
* pen timing - the pen ramps and then settles at each end of every stroke, so the
  servo is already at rest on the paper before ink flows, and fully lifted before
  the carriage moves away.

Frames are consecutive commands at a fixed dt. `pen` is 1.0 with the tip on the
paper and 0.0 lifted. `draw` marks the frames that lay ink; settle holds are
deliberately excluded so a parked pen does not blot.
"""
import numpy as np

from geometry import polyline_length, resample

SPACING = 0.006       # canvas units between commanded points
R_HOME, R_TRAVEL, R_RAMP_DOWN, R_SETTLE, R_DRAW, R_RAMP_UP = range(6)


def _travel_cost(seq, home, return_home):
    """Lift distance for an ordered, direction-decided stroke sequence.

    `home=None` means the origin is the first touchdown, so the tour is a cycle and
    its cost is the sum of the joins plus the closing leg back to the start.
    """
    if not seq:
        return 0.0
    origin = seq[0][1][0] if home is None else np.asarray(home, np.float64)
    c = 0.0 if home is None else float(np.linalg.norm(seq[0][1][0] - origin))
    for k in range(1, len(seq)):
        c += float(np.linalg.norm(seq[k][1][0] - seq[k - 1][1][-1]))
    if return_home:
        c += float(np.linalg.norm(origin - seq[-1][1][-1]))
    return c


def _refine(seq, home, return_home, max_passes=60):
    """2-opt with direction flips, the standard plotter ordering improvement.

    Reversing a slice of the sequence only stays legal if every stroke inside it
    also turns around: the slice's internal joins are then unchanged and only the
    two outer joins move, which is exactly what the candidate cost measures.
    """
    seq = list(seq)
    best = _travel_cost(seq, home, return_home)
    for _ in range(max_passes):
        improved = False
        for i in range(len(seq) - 1):
            for j in range(i + 1, len(seq)):
                mid = [(n, np.ascontiguousarray(xy[::-1])) for n, xy in seq[i:j + 1]]
                cand = seq[:i] + mid[::-1] + seq[j + 1:]
                c = _travel_cost(cand, home, return_home)
                if c < best - 1e-9:
                    seq, best, improved = cand, c, True
        if not improved:
            break
    return seq


def _rotate_to_front(seq, name):
    """Cycle-shift so `name` is drawn first, and park the origin on its touchdown.

    A closed tour (origin == first touchdown, return home at the end) is made of
    the same legs however it is rotated, so this changes the reveal order and
    nothing else - the lift distance is provably identical.
    """
    k = next((i for i, (n, _) in enumerate(seq) if n == name), None)
    if k is None or k == 0:
        return seq
    return seq[k:] + seq[:k]


def order_strokes(strokes, start=None, return_home=True, refine=True, first=None):
    """Greedy nearest-neighbour over stroke endpoints, then 2-opt, flipping freely.

    With `start=None` the origin becomes the first touchdown, so the program spends
    no lift on an approach leg and returns the carriage to where it started.
    """
    items = [(n, np.asarray(xy, np.float64)) for n, xy in strokes if len(xy) >= 2]
    if not items:
        return [], np.zeros(2)
    home = np.asarray(start, np.float64) if start is not None else None
    pool = list(items)
    cur = home if home is not None else np.zeros(2)
    out = []
    while pool:
        best_d, best_i, best_flip = np.inf, 0, False
        for i, (_, xy) in enumerate(pool):
            d0 = float(np.linalg.norm(xy[0] - cur))
            d1 = float(np.linalg.norm(xy[-1] - cur))
            if min(d0, d1) < best_d:
                best_d, best_i, best_flip = min(d0, d1), i, d1 < d0
        name, xy = pool.pop(best_i)
        if best_flip:
            xy = np.ascontiguousarray(xy[::-1])
        cur = xy[-1]
        out.append((name, xy))
    if refine:
        out = _refine(out, home, return_home if start is not None else True)
    if home is None:
        if first is not None and return_home:
            out = _rotate_to_front(out, first)
        home = out[0][1][0]
    return out, home


def _prepare(ordered, spacing, close_tol=None):
    """Close near-closed loops and arc-length resample each stroke."""
    tol = 2.0 * spacing if close_tol is None else float(close_tol)
    out = []
    for name, xy in ordered:
        if len(xy) > 2 and np.linalg.norm(xy[-1] - xy[0]) <= tol:
            # an apparently open list whose ends nearly meet is a closed loop with
            # one sample of slack; without this the pen jumps the gap instead of
            # finishing the shape. a two-point stroke is not a loop with slack - it
            # is a short line, and closing it would send the pen back over itself
            # and resample it to a single point, i.e. draw nothing at all.
            xy = np.concatenate([xy, xy[:1]], axis=0)
        out.append((name, resample(xy, spacing)))
    return out


def plan_strokes(strokes, spacing=SPACING, start=None, return_home=True,
                 close_tol=None, refine=True, first=None):
    """Ordered, direction-decided, resampled strokes plus the origin.

    This is the geometry a policy learns from and writes out: position and stroke
    structure only, with no pen-timing padding. `plan()` builds the timed program on
    top of it, so a reference path and a predicted one are always the same shape.
    """
    ordered, home = order_strokes(strokes, start, return_home, refine, first)
    if not ordered:
        raise ValueError("no strokes to plan")
    return _prepare(ordered, spacing, close_tol), home


def _ramp(n, a, b):
    """Cosine ease from pen level `a` to `b` over n frames, ending exactly at b."""
    if n <= 0:
        return np.zeros(0, np.float64)
    t = np.arange(1, n + 1, dtype=np.float64) / float(n)
    return a + (b - a) * (0.5 - 0.5 * np.cos(np.pi * t))


def plan(strokes, spacing=SPACING, ramp=7, settle=5, start=None, return_home=True,
         close_tol=None, refine=True, first=None):
    """Build the full action program for one drawing.

    Returns xy (T,2), pen (T,), role (T,), draw (T,), stroke_id (T,) and the
    ordered stroke names. Strokes are arc-length resampled to `spacing` here, so
    the program - not the artwork's own sampling - sets the ink density.
    """
    # the program flies home at the end, so the tour is a cycle and `first` can
    # rotate it to any opening stroke for exactly zero extra lift
    ordered, home = plan_strokes(strokes, spacing, start, return_home, close_tol,
                                 refine, first)
    X, P, R, D, SID = [], [], [], [], []

    def push(xy, pen, role, draw=False, sid=-1):
        X.append(np.asarray(xy, np.float64))
        P.append(float(pen))
        R.append(int(role))
        D.append(bool(draw))
        SID.append(int(sid))

    def push_many(pts, pen, role, draw=False, sid=-1):
        for k in range(len(pts)):
            push(pts[k], pen[k] if np.ndim(pen) else pen, role, draw, sid)

    cur = home
    if not np.allclose(np.asarray(ordered[0][1][0]), home):
        push_many(_travel(cur, np.asarray(ordered[0][1][0]), spacing), 0.0, R_TRAVEL)
        cur = ordered[0][1][0]

    for sid, (name, xy) in enumerate(ordered):
        p0, p1 = xy[0], xy[-1]
        if np.linalg.norm(p0 - cur) > 1e-9:
            push_many(_travel(cur, p0, spacing), 0.0, R_TRAVEL)
        push_many([p0] * ramp, _ramp(ramp, 0.0, 1.0), R_RAMP_DOWN, sid=sid)
        push_many([p0] * settle, 1.0, R_SETTLE, sid=sid)
        push_many(xy, 1.0, R_DRAW, draw=True, sid=sid)
        push_many([p1] * ramp, _ramp(ramp, 1.0, 0.0), R_RAMP_UP, sid=sid)
        cur = p1

    if return_home:
        push_many(_travel(cur, home, spacing), 0.0, R_TRAVEL)

    xy_all = np.asarray(X, np.float64)
    draw = np.asarray(D, bool)
    sid_all = np.asarray(SID, np.int64)
    pr = {
        "xy": xy_all,
        "pen": np.asarray(P, np.float64),
        "role": np.asarray(R, np.int64),
        "draw": draw,
        "stroke_id": sid_all,
        "names": [n for n, _ in ordered],
        "strokes": ordered,
        "home": home,
        "spacing": float(spacing),
        "T": int(len(xy_all)),
    }
    # ink is the length actually laid on the paper: summed inside each stroke, so a
    # lift between strokes can never be counted as drawing
    pr["ink_length"] = float(sum(polyline_length(xy_all[a:b])
                                 for a, b in stroke_slices(pr)))
    pr["travel_length"] = float(polyline_length(xy_all) - pr["ink_length"])
    return pr


def _travel(a, b, spacing):
    d = float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
    if d < 1e-12:
        return np.asarray([a], np.float64)
    n = max(int(round(d / spacing)), 1)
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    return a + t * (b - a)


def stroke_slices(pr):
    """(start, stop) into pr['xy'] of the drawn points for each stroke, in order."""
    sid = pr["stroke_id"]
    draw = pr["draw"]
    out = []
    for k in range(len(pr["names"])):
        idx = np.flatnonzero(draw & (sid == k))
        out.append((int(idx[0]), int(idx[-1]) + 1) if len(idx) else (-1, -1))
    return out


def segments(pr):
    """Consecutive ink segments as ((T,2) start, (T,2) end, stroke index).

    Splitting wherever the pen lifts or the stroke changes is what keeps the
    renderer from drawing a chord across the page between two strokes.
    """
    xy, draw, sid = pr["xy"], pr["draw"], pr["stroke_id"]
    ok = draw[1:] & draw[:-1] & (sid[1:] == sid[:-1])
    i = np.flatnonzero(ok)
    return xy[i], xy[i + 1], sid[i]


def stats(pr):
    total = pr["ink_length"] + pr["travel_length"]
    return {
        "strokes": len(pr["names"]),
        "frames": pr["T"],
        "ink_length": pr["ink_length"],
        "travel_length": pr["travel_length"],
        "travel_fraction": pr["travel_length"] / max(total, 1e-9),
    }


if __name__ == "__main__":
    import art

    st = art.fit_canvas(art.cat_strokes())
    pr = plan(st)
    xy, pen, role = pr["xy"], pr["pen"], pr["role"]

    assert np.isfinite(xy).all() and np.isfinite(pen).all(), "non-finite program"
    assert pen.min() >= -1e-12 and pen.max() <= 1.0 + 1e-12, "pen outside [0,1]"
    assert len(xy) == len(pen) == len(role) == len(pr["draw"]) == len(pr["stroke_id"]) == pr["T"]
    assert set(pr["stroke_id"][pr["draw"]]) == set(range(len(pr["names"]))), \
        "a stroke was planned but never drawn"
    assert np.all(pen[role == R_DRAW] > 0.999), "drawing while the pen is not down"
    assert np.all(pen[role == R_TRAVEL] < 1e-9), "dragging the pen between strokes"
    assert pen[0] < 0.05 and pen[-1] < 1e-9, "the program does not begin and end lifted"
    assert role[0] == R_RAMP_DOWN and role[-1] in (R_TRAVEL, R_RAMP_UP)

    # every stroke must be drawn, in the planned order, without a gap
    sl = stroke_slices(pr)
    assert all(a >= 0 for a, b in sl), "a stroke produced no ink"
    for k, (name, pts) in enumerate(pr["strokes"]):
        a, b = sl[k]
        got = xy[a:b]
        assert np.allclose(got, pts, atol=1e-9), "stroke %s was not reproduced exactly" % name
        assert np.array_equal(pr["stroke_id"][a:b], np.full(len(got), k)), "stroke label drift"

    # resampling is fine enough that playback cannot facet the curves. Measured
    # inside each stroke: between strokes the pen legitimately jumps.
    gaps = [np.linalg.norm(np.diff(xy[a:b], axis=0), axis=1) for a, b in sl if b > a]
    gaps = np.concatenate(gaps)
    assert gaps.max() < 2.2 * pr["spacing"], "ink spacing is uneven: max %.4f" % gaps.max()

    # the ink metric must equal the geometry it drew, not the carriage path
    expect = float(sum(polyline_length(p) for _, p in pr["strokes"]))
    assert abs(pr["ink_length"] - expect) < 1e-9, "ink_length disagrees with geometry"
    assert abs(pr["ink_length"] + pr["travel_length"] - polyline_length(xy)) < 1e-6

    # ordering: 2-opt must not lose to greedy, and the reveal rotation must be free
    greedy = plan(st, refine=False)
    assert pr["travel_length"] <= greedy["travel_length"] + 1e-9, "2-opt made travel worse"
    hero = art.fit_canvas(art.cat_strokes())
    pr_head = plan(hero, first="head")
    assert pr_head["names"][0] == "head", "opening stroke not honoured"
    assert abs(pr_head["travel_length"] - pr["travel_length"]) < 1e-6, \
        "rotating the tour to open on the head cost lift: %.4f vs %.4f" % (
            pr_head["travel_length"], pr["travel_length"])
    flat, _ = plan_strokes(hero, first="head")
    assert [n for n, _ in flat] == pr_head["names"], "plan_strokes disagrees with plan()"

    s = stats(pr_head)
    print("PLANNER PASS %d strokes %d frames  ink %.2f travel %.2f (%.0f%% lift)"
          % (s["strokes"], s["frames"], s["ink_length"], s["travel_length"],
             100.0 * s["travel_fraction"]))
    print("        order:", " -> ".join(pr_head["names"]))
    print("        spacing %.4f  gap mean %.4f std %.4f max %.4f"
          % (pr_head["spacing"], gaps.mean(), gaps.std(), gaps.max()))
    print("        strokes/plan %d/%d  lift cost rotation-free %.6f (greedy %.4f)"
          % (len(flat), len(hero), pr_head["travel_length"], greedy["travel_length"]))

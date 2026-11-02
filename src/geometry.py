"""Curve primitives: centripetal Catmull-Rom smoothing and arc-length resampling.

Strokes are authored as a handful of waypoints. Centripetal parameterization
(alpha=0.5) is what keeps the spline from looping or cusping when waypoint
spacing is uneven, which is why it is used instead of uniform Catmull-Rom.
"""
import numpy as np


def catmull_rom(points, samples_per_unit=400.0, closed=False, alpha=0.5):
    """Smooth polyline through `points` (M,2 waypoints).

    The returned curve passes through every waypoint, including the first and
    last. Padding is built so each real segment [p_j, p_j+1] has two neighbours
    on either side, and the segment list is derived from real waypoint indices
    rather than from a guessed range.
    """
    p = np.asarray(points, np.float64)
    n = len(p)
    if n < 2:
        raise ValueError("need at least 2 waypoints")
    if closed:
        if n < 3:
            raise ValueError("closed spline needs at least 3 waypoints")
        # real p_j sits at padded index j+2
        pp = np.concatenate([p[-2:], p, p[:2]], axis=0)
        segs = [j + 1 for j in range(n)]
    else:
        # real p_j sits at padded index j+1
        pp = np.concatenate([p[:1], p, p[-1:]], axis=0)
        segs = list(range(n - 1))

    d = np.linalg.norm(np.diff(pp, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(np.maximum(d, 1e-9) ** alpha)])

    out = []
    for i in segs:
        chord = _chord(pp[i + 1], pp[i + 2])
        n_s = max(int(round(samples_per_unit * chord)), 4)
        ts = np.linspace(t[i + 1], t[i + 2], n_s, endpoint=True)
        out.append(_eval_segments(pp, t, i, ts))
    xy = np.concatenate(out, axis=0)
    if closed:
        xy = xy[:-1]          # last sample repeats the first waypoint
    return _dedupe(np.nan_to_num(xy, nan=0.0))


def _chord(a, b):
    return float(np.linalg.norm(b - a))


def _eval_segments(p, t, i, ts):
    """Vectorised centripetal Catmull-Rom over one segment [t_{i+1}, t_{i+2}]."""
    t0, t1, t2, t3 = t[i], t[i + 1], t[i + 2], t[i + 3]
    p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
    tt = ts[:, None]

    def lerp(a, b, ta, tb):
        w = np.clip((tt - ta) / max(tb - ta, 1e-12), 0.0, 1.0)
        return (1.0 - w) * a + w * b

    a1 = lerp(p0, p1, t0, t1)
    a2 = lerp(p1, p2, t1, t2)
    a3 = lerp(p2, p3, t2, t3)
    b1 = lerp(a1, a2, t0, t2)
    b2 = lerp(a2, a3, t1, t3)
    return lerp(b1, b2, t1, t2)


def smooth(pts, sigma_arc=0.06, spacing=0.008, keep_ends=True, closed=False):
    """Gaussian low-pass along arc length, which rounds the corner kinks that a
    spline through sparse waypoints leaves behind.

    Positions are resampled uniformly first so the kernel width means the same
    physical distance everywhere; otherwise a Gaussian in index space would
    smear slow-moving parts of the curve and barely touch fast-moving ones.
    """
    q = resample(np.asarray(pts, np.float64), spacing)
    if len(q) < 5:
        return q
    k = max(int(round(sigma_arc / spacing)), 1)
    w = np.exp(-0.5 * (np.arange(-k, k + 1) / float(k)) ** 2)
    w /= w.sum()
    mode = "wrap" if closed else "edge"
    xs = np.convolve(np.pad(q[:, 0], k, mode=mode), w, "valid")
    ys = np.convolve(np.pad(q[:, 1], k, mode=mode), w, "valid")
    out = np.column_stack([xs, ys])
    if keep_ends and not closed:
        out[0] = q[0]
        out[-1] = q[-1]
    return out


def polyline_length(pts):
    pts = np.asarray(pts, np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def resample(pts, spacing):
    """Arc-length resample to (nearly) constant `spacing`; endpoints preserved."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 2 or spacing <= 0:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-12:
        return pts[:1].copy()
    n = max(int(round(total / spacing)), 2)
    q = np.linspace(0.0, total, n)
    xy = np.column_stack([np.interp(q, cum, pts[:, 0]), np.interp(q, cum, pts[:, 1])])
    xy[0] = pts[0]
    xy[-1] = pts[-1]
    return xy


def ellipse_arc(cx, cy, rx, ry, a0=0.0, a1=2 * np.pi, n=128, tilt=0.0):
    th = np.linspace(a0, a1, n, endpoint=(a1 - a0) < 2 * np.pi - 1e-9)
    x, y = rx * np.cos(th), ry * np.sin(th)
    if tilt:
        c, s = np.cos(tilt), np.sin(tilt)
        x, y = c * x - s * y, s * x + c * y
    return np.column_stack([x + cx, y + cy])


def mirror_x(pts):
    q = np.asarray(pts, np.float64)
    return np.column_stack([-q[:, 0], q[:, 1]])


def _dedupe(pts):
    if len(pts) < 2:
        return pts
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-6:
            keep.append(i)
    return pts[keep]


if __name__ == "__main__":
    def passes_through(pts, curve, tol=2e-3):
        """Every waypoint must appear on the curve - catches dropped segments."""
        miss = []
        for k, w in enumerate(np.asarray(pts, float)):
            if len(curve) == 0 or np.linalg.norm(curve - w, axis=1).min() > tol:
                miss.append(k)
        return miss

    tri = [(0.0, 0.0), (1.0, 0.2), (0.5, 1.0)]
    assert not passes_through(tri, catmull_rom(tri)), "open spline misses waypoints"
    assert not passes_through(tri, catmull_rom(tri, closed=True)), "closed spline misses"
    sq = catmull_rom([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)], closed=True)
    assert np.isfinite(sq).all() and len(sq) > 40, "catmull-rom produced junk"
    r = resample(sq, 0.02)
    gaps = np.linalg.norm(np.diff(r, axis=0), axis=1)
    assert abs(gaps.mean() - 0.02) < 0.002 and gaps.std() < 0.002, "resample not uniform"
    e = ellipse_arc(0, 0, 1.0, 0.5, n=200)
    assert abs(e[:, 0].max() - 1.0) < 0.01 and abs(e[:, 1].max() - 0.5) < 0.01
    print("GEOMETRY PASS  waypoints_hit=True spline=%d gap_mean=%.4f gap_std=%.5f"
          % (len(sq), gaps.mean(), gaps.std()))

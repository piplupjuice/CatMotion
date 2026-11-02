"""The cat, as a parametric line drawing.

Everything is built from a small parameter vector so that (a) the hero artwork can
be hand-tuned until it genuinely looks good, and (b) a family of styles can be
sampled to train the policy. Coordinates are y-up in roughly [-1,1]; the canvas
fit step normalises afterwards.
"""
import numpy as np

from geometry import catmull_rom, ellipse_arc, mirror_x, smooth

# Hero style: hand-tuned so the drawing reads as an elegant sitting cat.
# Angles are degrees on the head ellipse; lengths are canvas units.
HERO = dict(
    head_cy=0.44, head_rx=0.30, head_ry=0.27, head_tilt=0.0,
    ear_deg_out=146, ear_deg_in=104, ear_h=0.30, ear_tip_k=0.80, ear_mid_k=0.42,
    inner_ear=True,
    eye_cx=0.125, eye_cy=0.49, eye_w=0.118, eye_h=0.066, eye_tilt=0.18,
    pupil_r=0.023, pupil_k=0.45,
    nose_cy=0.345, nose_w=0.050, nose_h=0.042,
    mouth_cy=0.285, mouth_w=0.098, mouth_depth=0.032,
    whisk_root=0.118, whisk_len=0.30, whisk_curl=0.045,
    whisk_deg_1=172, whisk_deg_2=184, whisk_deg_3=193,
    body_top_y=0.16, shoulder=0.265, body_deg_out=228, body_deg_in=312,
    hip_x=0.40, body_h=0.60, bottom_y=-0.70, foot_x=0.34,
    leg_cx=0.118, paw_h=0.092, paw_base_y=-0.700,
    tail_dir=1.0, tail_len=1.00, tail_turn_deg=215.0, tail_th0_deg=-3.4,
)


def cat_strokes(p=None):
    """Return list of (name, xy) strokes, xy shaped (M,2) float64.

    Facial features attach to the head ellipse by angle rather than by absolute
    coordinate, so a style with a bigger or smaller head keeps its ears, eyes and
    whiskers correctly anchored.
    """
    q = dict(HERO)
    if p:
        q.update(p)
    hx, hy, rx, ry = 0.0, q["head_cy"], q["head_rx"], q["head_ry"]

    def on_head(deg):
        a = np.deg2rad(deg)
        return np.array([hx + rx * np.cos(a), hy + ry * np.sin(a)])

    def out_of(deg, k):
        """Point along the head normal at `deg`, pushed `k` units outward."""
        a = np.deg2rad(deg)
        n = np.array([np.cos(a) / max(rx, 1e-6), np.sin(a) / max(ry, 1e-6)])
        n = n / max(np.linalg.norm(n), 1e-9)
        return on_head(deg) + n * k

    S = []
    sy = q["body_top_y"]
    S.append(("head", _ellipse(hx, hy, rx, ry, tilt=q["head_tilt"])))

    # ---- ears: true triangle with a crisp apex. Waypoints are taken along the
    # two ear edges, so Catmull-Rom cannot bow the sides out into a flag shape.
    d1, d2 = q["ear_deg_out"], q["ear_deg_in"]
    am, h = (d1 + d2) * 0.5, q["ear_h"]
    b1, b2 = on_head(d1), on_head(d2)
    tip = out_of(am, h)
    k1, k2 = q["ear_tip_k"], q["ear_mid_k"]
    ear = np.array([
        b1,
        b1 + (tip - b1) * k2,
        b1 + (tip - b1) * k1,
        tip,
        b2 + (tip - b2) * k1,
        b2 + (tip - b2) * k2,
        b2,
    ])
    ear[:, 0] = -np.abs(ear[:, 0])
    ear = catmull_rom(ear)
    S.append(("ear_L", ear))
    S.append(("ear_R", mirror_x(ear)))
    if q["inner_ear"]:
        # a copy of the ear shrunk about its own centroid, so it can never poke
        # through the ear outline the way an independently-placed V can
        cen = (b1 + b2 + tip) / 3.0
        ie = cen + (np.array([b1, tip, b2]) - cen) * 0.52
        ie[:, 0] = -np.abs(ie[:, 0])
        ie = catmull_rom(ie)
        S.append(("earin_L", ie))
        S.append(("earin_R", mirror_x(ie)))

    # ---- eyes: almond lens + pupil, mirrored pair ----
    lens = _lens((q["eye_cx"], q["eye_cy"]), q["eye_w"], q["eye_h"], q["eye_tilt"])
    lens[:, 0] = -np.abs(lens[:, 0])
    S.append(("eye_L", lens))
    S.append(("eye_R", mirror_x(lens.copy())))
    pr, pk = q["pupil_r"], q["pupil_k"]
    S.append(("pup_L", _ellipse(-q["eye_cx"], q["eye_cy"], pr * pk, pr * (2.0 - pk))))
    S.append(("pup_R", _ellipse(q["eye_cx"], q["eye_cy"], pr * pk, pr * (2.0 - pk))))

    # ---- nose: rounded downward triangle; mouth: omega under it, rising in the
    # middle to meet the nose exactly. a separate philtrum line would be a 0.04-unit
    # stroke, below the resolution of the path the policy is trained to output, so
    # the connection is carried by the mouth itself instead of by a detail nothing
    # downstream can reproduce.
    nw, nh, ny = q["nose_w"], q["nose_h"], q["nose_cy"]
    nose = np.array([[-nw, ny + nh * 0.55], [nw, ny + nh * 0.55], [0.0, ny - nh],
                     [-nw, ny + nh * 0.55]])
    S.append(("nose", catmull_rom(nose, closed=True)))
    my, mw, md = q["mouth_cy"], q["mouth_w"], q["mouth_depth"]
    mouth = np.array([[-mw, my + md * 0.60], [-mw * 0.48, my - md * 0.50],
                      [0.0, ny - nh], [mw * 0.48, my - md * 0.50],
                      [mw, my + md * 0.60]])
    S.append(("mouth", catmull_rom(mouth)))

    # ---- whiskers: rooted on the whisker bed beside the muzzle, staggered down the
    # cheek and shortening as they go. three lines out of one point read as a brush;
    # three lines out of three points, each a different length, read as a fan.
    for k, deg in enumerate((q["whisk_deg_1"], q["whisk_deg_2"], q["whisk_deg_3"])):
        a = np.deg2rad(deg)
        dirv = np.array([np.cos(a), np.sin(a)])
        root = np.array([-q["whisk_root"], q["nose_cy"] - 0.004 - 0.024 * k])
        w_l = q["whisk_len"] * (1.0 - 0.11 * k)
        droop = q["whisk_curl"] * (0.25 + 0.55 * k)
        w = np.array([root,
                      root + dirv * w_l * 0.52 + np.array([0.0, -droop * 0.30]),
                      root + dirv * w_l + np.array([0.0, -droop])])
        w = catmull_rom(w)
        S.append(("whisk_L%d" % k, w))
        S.append(("whisk_R%d" % k, mirror_x(w)))

    # ---- body: open bell, anchored exactly on the head outline so the shoulders
    # emerge from the head instead of crossing through it
    b0, b1y = on_head(q["body_deg_out"]), on_head(q["body_deg_in"])
    body = np.array([
        b0,
        [-q["shoulder"], sy],
        [-q["hip_x"] * 0.78, sy - q["body_h"] * 0.42],
        [-q["hip_x"], sy - q["body_h"] * 0.72],
        [-q["hip_x"] * 1.02, q["bottom_y"] + q["foot_x"] * 0.28],
        [-q["foot_x"], q["bottom_y"]],
        [0.0, q["bottom_y"] - 0.010],
        [q["foot_x"], q["bottom_y"]],
        [q["hip_x"] * 1.02, q["bottom_y"] + q["foot_x"] * 0.28],
        [q["hip_x"], sy - q["body_h"] * 0.72],
        [q["hip_x"] * 0.78, sy - q["body_h"] * 0.42],
        [q["shoulder"], sy],
        b1y,
    ])
    S.append(("body", catmull_rom(body)))

    # ---- front legs (slightly bowed) and a 3-point dome paw. the legs are rooted
    # exactly on the head's lower outline so they emerge from the chin instead of
    # floating free inside the chest. a dome with only base/shoulder/apex waypoints
    # makes Catmull-Rom overshoot below the floor, hence the four-point arc.
    lx, gy = q["leg_cx"], q["paw_base_y"]
    ca = np.arccos(np.clip(-lx * 0.80 / max(rx, 1e-6), -1.0, 1.0))
    chin = np.array([hx + rx * np.cos(ca), hy - ry * np.sin(ca)])
    leg = np.array([chin,
                    [-lx * 0.92, (chin[1] + gy + q["paw_h"]) * 0.5 + 0.06],
                    [-lx * 1.06, (chin[1] + gy + q["paw_h"]) * 0.5 - 0.06],
                    [-lx * 1.00, gy + q["paw_h"]]])
    S.append(("leg_L", catmull_rom(leg)))
    S.append(("leg_R", mirror_x(catmull_rom(leg))))
    paw = ellipse_arc(-lx * 1.00, gy, lx * 0.62, q["paw_h"], a0=0.0, a1=np.pi, n=48)
    S.append(("paw_L", paw))
    S.append(("paw_R", mirror_x(paw)))

    # ---- tail: one continuous curve whose turn accelerates toward the tip, the way
    # a real tail thickens its curl. constant curvature gives a circular handle;
    # waypoint splines double back or kink. rooted exactly on the haunch waypoint of
    # the body outline, so it cannot start inside the silhouette.
    d, gy = q["tail_dir"], q["bottom_y"]
    root = np.array([d * q["hip_x"] * 1.02, gy + q["foot_x"] * 0.28])
    n, L = 64, max(q["tail_len"], 1e-6)
    t = np.linspace(0.0, 1.0, n)
    th = np.deg2rad(q["tail_th0_deg"]) + np.deg2rad(q["tail_turn_deg"]) * t ** 1.55
    ds = L / (n - 1)
    x = np.concatenate([[0.0], np.cumsum(ds * np.cos(th[:-1]))])
    y = np.concatenate([[0.0], np.cumsum(ds * np.sin(th[:-1]))])
    tail = np.column_stack([d * x, y]) + root
    S.append(("tail", tail))

    S = [(n, xy) for n, xy in S if len(np.asarray(xy)) > 1]
    return [(n, _ease(n, np.asarray(xy, np.float64))) for n, xy in S]


CLOSED_STROKES = {"head", "nose", "pup_L", "pup_R"}


def _ease(name, xy):
    """Round the corner kinks a spline through sparse waypoints leaves behind.

    The kernel grows with stroke length but has a floor, because the kinkiest
    strokes here (whiskers, inner ears) are also short ones: a purely
    length-proportional sigma would leave them polygonal.
    """
    L = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    if L < 0.25:
        return xy
    sigma = float(np.clip(0.035 * L + 0.022, 0.022, 0.085))
    return smooth(xy, sigma_arc=sigma, closed=name in CLOSED_STROKES)


def _ellipse(cx, cy, rx, ry, n=220, tilt=0.0):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = rx * np.cos(th), ry * np.sin(th)
    if tilt:
        c, s = np.cos(tilt), np.sin(tilt)
        x, y = c * x - s * y, s * x + c * y
    return np.column_stack([x + cx, y + cy])


def _lens(center, w, h, tilt):
    """Almond eye: upper and lower arc meeting at the two corners."""
    a = np.linspace(0.0, np.pi, 60)
    top = np.column_stack([w * 0.5 * np.cos(a), h * np.sin(a)])
    bot = np.column_stack([w * 0.5 * np.cos(a), -h * 0.42 * np.sin(a)])
    xy = np.concatenate([top, bot[::-1][1:]], axis=0)
    if tilt:
        c, s = np.cos(tilt), np.sin(tilt)
        xy = np.column_stack([c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]])
    return xy + np.asarray(center)


def fit_canvas(strokes, margin=0.06):
    """Scale + centre so the whole drawing fits inside [-1+margin, 1-margin]^2."""
    allp = np.concatenate([xy for _, xy in strokes], axis=0)
    lo, hi = allp.min(0), allp.max(0)
    extent = (hi - lo).max()
    lim = 1.0 - margin
    k = (2.0 * lim) / max(extent, 1e-6)
    mid = (lo + hi) * 0.5
    out = []
    for n, xy in strokes:
        out.append((n, (xy - mid) * k))
    return out


def total_length(strokes):
    return float(sum(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
                     for _, xy in strokes))


def rasterize(strokes, size=48, line_width=1.5):
    """Render strokes to a (size,size) float32 ink grid in [0,1].

    Row 0 is the top of the canvas (y is flipped), which is the convention the
    policy's goal conditioning uses.
    """
    img = np.zeros((size, size), np.float64)
    sigma = max(line_width * 0.5, 0.55)
    for _, xy in strokes:
        px = np.column_stack([(xy[:, 0] + 1.0) * 0.5 * (size - 1),
                              (1.0 - xy[:, 1]) * 0.5 * (size - 1)])
        if len(px) < 2:
            continue
        seg = np.linalg.norm(np.diff(px, axis=0), axis=1)
        steps = np.maximum(np.ceil(seg / 0.45), 1).astype(np.int64)
        dense = []
        for i in range(len(px) - 1):
            t = np.linspace(0.0, 1.0, int(steps[i]) + 1, endpoint=False)[:, None]
            dense.append(px[i] + t * (px[i + 1] - px[i]))
        dense.append(px[-1:])
        pts = np.concatenate(dense, axis=0)
        _splat(img, pts[:, 0], pts[:, 1], size, sigma)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _splat(img, x, y, size, sigma):
    """Vectorised soft-disk stamp; max-composited so overlapping strokes stay 1."""
    cx = np.rint(x).astype(np.int64)
    cy = np.rint(y).astype(np.int64)
    rad = int(np.ceil(sigma * 2.2))
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            d2 = (cx + dx - x) ** 2 + (cy + dy - y) ** 2
            val = np.exp(-0.5 * d2 / (sigma * sigma))
            ok = (cx + dx >= 0) & (cx + dx < size) & (cy + dy >= 0) & (cy + dy < size) & (val > 0.02)
            if ok.any():
                np.maximum.at(img, (cy[ok] + dy, cx[ok] + dx), val[ok])


def sample_style(rng):
    """Draw one style from the family, jittered around the hero so every sample
    stays a recognisable, well-proportioned cat."""
    u = rng.uniform
    bottom = u(-0.75, -0.66)
    return dict(
        head_cy=u(0.40, 0.47), head_rx=u(0.275, 0.325), head_ry=u(0.245, 0.295),
        head_tilt=u(-0.04, 0.04),
        ear_deg_out=u(138, 154), ear_deg_in=u(96, 112), ear_h=u(0.24, 0.36),
        ear_tip_k=u(0.74, 0.88), ear_mid_k=u(0.34, 0.50),
        inner_ear=bool(rng.uniform() < 0.8),
        eye_cx=u(0.105, 0.145), eye_cy=u(0.465, 0.515), eye_w=u(0.100, 0.135),
        eye_h=u(0.054, 0.078), eye_tilt=u(0.10, 0.26), pupil_r=u(0.018, 0.028),
        pupil_k=u(0.34, 0.58),
        nose_cy=u(0.325, 0.365), nose_w=u(0.042, 0.060), nose_h=u(0.034, 0.052),
        mouth_cy=u(0.265, 0.305), mouth_w=u(0.084, 0.114), mouth_depth=u(0.024, 0.042),
        whisk_root=u(0.105, 0.135), whisk_len=u(0.25, 0.34), whisk_curl=u(0.025, 0.070),
        whisk_deg_1=u(166, 178), whisk_deg_2=u(180, 190), whisk_deg_3=u(193, 204),
        body_top_y=u(0.125, 0.190), shoulder=u(0.235, 0.295),
        body_deg_out=u(222, 236), body_deg_in=u(304, 318), hip_x=u(0.365, 0.435),
        body_h=u(0.55, 0.66), bottom_y=bottom, foot_x=u(0.305, 0.375),
        leg_cx=u(0.108, 0.142), paw_h=u(0.075, 0.110),
        paw_base_y=bottom + 0.006,
        tail_dir=float(rng.choice([-1.0, 1.0])), tail_len=u(0.82, 1.10),
        tail_turn_deg=u(150.0, 240.0), tail_th0_deg=u(-9.0, 3.0),
    )


if __name__ == "__main__":
    st = fit_canvas(cat_strokes())
    names = [n for n, _ in st]
    allp = np.concatenate([xy for _, xy in st])
    assert all(np.isfinite(xy).all() for _, xy in st), "non-finite stroke"
    assert np.abs(allp).max() <= 0.95, "artwork escapes the canvas"
    assert len(names) >= 18, "missing strokes: %s" % names
    assert "ear_L" in names and "tail" in names and "head" in names
    print("ART PASS %d strokes, len %.2f units, xy in [%.2f, %.2f]"
          % (len(names), total_length(st), allp.min(), allp.max()))
    print("        ", ", ".join(names))

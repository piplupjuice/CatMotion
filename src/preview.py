"""Design preview: render stroke sets to a PNG so the artwork can be judged.

This is the design-review tool. It does not involve the robot or the policy, so a
problem here is purely an art problem.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from art import cat_strokes, fit_canvas, rasterize, sample_style, total_length


def panel(strokes, ax, title, lw=1.7, color="#7ff2ff"):
    for _, xy in strokes:
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, solid_capstyle="round",
                solid_joinstyle="round")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, color="#9fb4c8", fontsize=8, pad=2)


def preview(out_path, n_styles=4, seed=7):
    hero = fit_canvas(cat_strokes())
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, n_styles + 1, figsize=(3.0 * (n_styles + 1), 3.2))
    fig.patch.set_facecolor("#0a0c14")
    panel(hero, axes[0], "HERO  len=%.1f" % total_length(hero), lw=2.0, color="#8ff6ff")
    for i, ax in enumerate(axes[1:]):
        st = fit_canvas(cat_strokes(sample_style(rng)))
        panel(st, ax, "style %d" % i, lw=1.5, color="#c8d8e8")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    # large single-hero render for judging line quality and detail
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    fig.patch.set_facecolor("#0a0c14")
    ax.set_facecolor("#0a0c14")
    panel(hero, ax, "", lw=2.6, color="#8ff6ff")
    fig.tight_layout()
    big = out_path.replace(".png", "_hero.png")
    fig.savefig(big, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    r = rasterize(hero, size=48)
    print("PREVIEW %s  hero_strokes=%d len=%.2f raster_ink=%.4f"
          % (out_path, len(hero), total_length(hero), float(r.mean())))
    return hero


def plan_view(pr, out_path):
    """Draw the pen program: where ink lands, where the carriage lifts and flies."""
    import planner
    fig, (ax, ty) = plt.subplots(2, 1, figsize=(7, 8.6),
                                gridspec_kw=dict(height_ratios=[3, 1]))
    fig.patch.set_facecolor("#0a0c14")
    for a in (ax, ty):
        a.set_facecolor("#0a0c14")
    role = pr["role"]
    xy = pr["xy"]
    lift = xy[role == planner.R_TRAVEL]
    ax.plot(lift[:, 0], lift[:, 1], color="#6b4a2a", lw=0.7, ls=":", zorder=1)
    for k, (a, b) in enumerate(planner.stroke_slices(pr)):
        seg = xy[a:b]
        ax.plot(seg[:, 0], seg[:, 1], color="#8ff6ff", lw=1.6, zorder=2,
                solid_capstyle="round")
        ax.plot(seg[0, 0], seg[0, 1], "o", ms=4.5, mfc="#ff8f5a", mec="none", zorder=3)
        ax.annotate(str(k), seg[0] + 0.03, color="#ff8f5a", fontsize=6)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("ink cyan / lift dotted / touchdown orange   T=%d" % pr["T"],
                 color="#9fb4c8", fontsize=8, pad=2)
    t = np.arange(len(xy))
    ty.plot(t, pr["pen"], color="#8ff6ff", lw=1.0)
    ty.plot(t, (role == planner.R_DRAW).astype(float), color="#2f7d5e", lw=0.6)
    ty.set_ylim(-0.05, 1.15)
    ty.set_xlabel("frame", color="#9fb4c8", fontsize=7)
    ty.tick_params(colors="#9fb4c8", labelsize=6)
    for s in ty.spines.values():
        s.set_color("#33405a")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    s = planner.stats(pr)
    print("PLAN_VIEW %s  strokes=%d frames=%d ink=%.2f lift=%.2f (%.0f%%)"
          % (out_path, s["strokes"], s["frames"], s["ink_length"], s["travel_length"],
             100.0 * s["travel_fraction"]))
    return s


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "preview.png"
    hero = preview(out)
    names = [n for n, _ in hero]
    assert len(names) >= 18, names
    print("strokes:", ", ".join(names))
    if "--plan" in sys.argv:
        import planner
        plan_view(planner.plan(hero, first="head"), out.replace(".png", "_plan.png"))

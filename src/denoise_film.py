"""The reverse process as a picture: noise, one step at a time, becoming a cat.

Everything else in this project films the machine. This films the model: the same
checkpoint, the same goal raster, and the 512-point path at every one of the denoising
steps, so a viewer watches a pen program condense out of white noise while the one
picture the network was actually given sits lit in the corner.

It is also the cheapest film here - no MuJoCo, no GL context, no contact forces - which
is why it can run on a GPU box in seconds instead of an hour, and why every frame of it
can be checked against the tensor it was drawn from.

Nothing is decorative: the geometry of a frame is the sampler's own state, and the
numbers beside it are measured from that same state at that same step.
"""
import os
import sys

# no GL context is opened here, but render.py is imported for its typography and it
# binds the MuJoCo backend at import time, so the platform default has to be right first.
os.environ.setdefault("MUJOCO_GL", "wgl" if sys.platform == "win32" else "egl")

import json
import time

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter

import data
import metrics
import model as M
import policy_film as pf
import render

OUT = pf.OUT
DESIGN = pf.DESIGN

BG = (15, 17, 22)
PAPER = (247, 244, 238)
INK = (26, 24, 30)
GHOST = (122, 120, 130)
GOAL_ON = (236, 240, 246)
GOAL_OFF = (26, 29, 37)
WIN = 1.12          # the canvas window drawn, in units either side of the centre
# how hard the frame budget leans toward the end of the schedule; see `pacing`.
PACING_BIAS = 1.7

# the numbers in the captions come from the modules, never typed in here: a label that
# disagrees with data.py is a lie in the finished film.
TITLES = [("HOW  A  CAT  APPEARS", 46),
          ("one %d x %d picture in, one %d-point pen program out"
           % (data.GOAL, data.GOAL, data.K), 24)]
GOAL_LABEL = [("THE  INPUT", 24), ("%d x %d goal raster" % (data.GOAL, data.GOAL), 20)]


def paper_label(steps):
    return [("THE  OUTPUT", 24),
            ("%d-point pen path, %d DDPM steps" % (data.K, steps), 20)]


def _font(px, scale):
    """One size lookup, shared by the drawer and the measurers so they cannot disagree."""
    return render.font(max(11, int(round(px * scale))))


def block_h(lines, scale=1.0):
    """Rendered height of a text block - the same arithmetic text_block does."""
    d = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    return sum(d.textbbox((0, 0), t, font=_font(px, scale))[3]
               + 0.55 * px * scale for t, px in lines)


def block_w(lines, scale=1.0):
    """Rendered width of the widest line in a text block."""
    d = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    return max(d.textbbox((0, 0), t, font=_font(px, scale))[2] for t, px in lines)


def layout(size, steps):
    """Where every element of a frame lives, computed from the film size alone.

    Two columns: the paper owns the left, the goal and its numbers own the right, the
    title owns the top. Nothing is anchored to an edge of the picture, because an
    edge-anchored readout walks onto the paper as soon as the resolution changes.
    """
    w, h = size
    gs = int(0.26 * w)
    gx0 = w - gs - int(0.05 * w)
    left = (int(0.05 * w), gx0 - int(0.045 * w))
    side = min(int(0.74 * h), int(0.95 * (left[1] - left[0])))
    sc = h / 1080.0
    # the labels are hung by their own measured height, so they sit just above the cards
    # they name at any resolution instead of drifting onto them as the size changes.
    t_h = block_h(TITLES, sc)
    g_h = block_h(GOAL_LABEL, sc)
    x0 = left[0] + ((left[1] - left[0]) - side) // 2
    y0 = int(0.145 * h)
    return dict(box=(x0, y0, x0 + side, y0 + side), goal_box=(gx0, y0, gx0 + gs, y0 + gs),
                title_xy=(x0, max(6.0, y0 - 10.0 - t_h)),
                goal_label_xy=(gx0, y0 - 10.0 - g_h),
                paper_label_xy=(x0, y0 + side + int(0.022 * h)),
                hud_xy=(gx0, y0 + gs + int(0.045 * h)),
                paper_label=paper_label(steps), width=0.0055 * side, scale=sc)


def check_layout(L, size, hud, corner=None):
    """Every text block measured against the boxes it must not touch.

    A film that takes 290 frames to render does not get to be found out afterwards
    because a title landed on the drawing.
    """
    h = size[1]
    sc = L["scale"]
    box = L["box"]
    assert L["goal_box"][0] > box[2], "the goal card sits on top of the paper"
    assert L["title_xy"][1] + block_h(TITLES, sc) <= box[1] + 1, "the title is on the paper"
    assert L["title_xy"][0] + block_w(TITLES, sc) <= L["goal_box"][0], \
        "the title reached the goal card"
    assert L["goal_label_xy"][1] + block_h(GOAL_LABEL, sc) <= L["goal_box"][1], \
        "the goal label is on the goal card"
    y = L["hud_xy"][1] + block_h(hud, sc)
    if corner:
        y += int(0.03 * h) + block_h(corner, sc)
    assert y < h, "the readout runs off the bottom of the frame (%.0f > %d)" % (y, h)
    assert L["paper_label_xy"][1] + block_h(L["paper_label"], sc) < h, \
        "the paper label runs off the frame"
    assert L["paper_label_xy"][1] > box[3], "the paper label is on the paper"
    return True


def mapper(box):
    """canvas (x, y) -> screen (u, v): y is up on the page and down on the screen."""
    x0, y0, x1, y1 = box
    pad = 0.05 * (x1 - x0)
    inner = (x1 - x0) - 2 * pad

    def m(x, y):
        u = (np.asarray(x, np.float64) + WIN) / (2 * WIN)
        v = (np.asarray(y, np.float64) + WIN) / (2 * WIN)
        return np.column_stack([x0 + pad + u * inner, y1 - pad - v * inner])
    return m


def text_block(img, xy, lines, scale=1.0, color=(216, 224, 235), shadow=(5, 7, 12)):
    """Left-aligned text at an explicit origin."""
    d = ImageDraw.Draw(img)
    x, y = float(xy[0]), float(xy[1])
    for txt, px in lines:
        f = _font(px, scale)
        d.text((x + 2.0, y + 2.0), txt, font=f, fill=shadow)
        d.text((x, y), txt, font=f, fill=color)
        y += d.textbbox((0, 0), txt, font=f)[3] + 0.55 * px * scale


def paper_card(img, box, fill=PAPER):
    """A card with a soft drop shadow: paper floating in a dark room."""
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = [int(v) for v in box]
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.rectangle([x0 + 9, y0 + 13, x1 + 9, y1 + 13], fill=(0, 0, 0, 200))
    sh = sh.filter(ImageFilter.GaussianBlur(14))
    img.paste(sh, (0, 0), sh)
    d.rectangle(box, fill=fill)
    d.rectangle([x0, y0, x1, y0 + 1], fill=(255, 255, 255))


def goal_card(img, box, goal):
    """The single input the network is given, upscaled hard enough to show its pixels."""
    d = ImageDraw.Draw(img)
    paper_card(img, box, GOAL_OFF)
    inset = max(4, int(0.03 * (box[2] - box[0])))
    side = int(box[2] - box[0]) - 2 * inset
    # ink lit on a dark card: this is the picture the network sees, not a second drawing,
    # and the contrast of dark ink on white paper is already taken by the output card.
    g = np.clip(np.asarray(goal, np.float64), 0.0, 1.0)
    cell = side / float(g.shape[0])
    tile = Image.fromarray((g * 255).astype(np.uint8)).resize((side, side), Image.NEAREST)
    if cell > 6.0:
        # a hard ~10 px block edge is the worst case for x264's chroma subsampling, and
        # a whole cell of blur would turn the input into a second soft drawing. a fifth
        # of a cell keeps the grid legible as 48 x 48 while rounding the worst aliasing.
        tile = tile.filter(ImageFilter.GaussianBlur(0.15 * cell))
    tile = np.asarray(tile, np.float32) / 255.0
    on = np.array(GOAL_ON, np.float32)
    off = np.array(GOAL_OFF, np.float32)
    rgb = off[None, None, :] * (1.0 - tile[:, :, None]) + on[None, None, :] * tile[:, :, None]
    u8 = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    # every pixel of this card is a mix of two colours, so it must lie on the line
    # between them. one that does not means the 0-255 ramp was scaled twice and the
    # uint8 cast wrapped - which paints the network's input as coloured confetti.
    delta = on - off
    along = ((u8.astype(np.float32) - off) @ delta) / float(delta @ delta)
    resid = np.abs(u8.astype(np.float32) - (off + along[..., None] * delta)).max()
    assert resid <= 2.0, "goal card pixel %.1f levels off the ink line" % resid
    img.paste(Image.fromarray(u8), (int(box[0]) + inset, int(box[1]) + inset))
    d.rectangle(box, outline=(78, 88, 104), width=2)


def clip(lay, box):
    """Cut a transparent layer to the paper.

    The canvas is open: a noise frame holds pen points at x = 3, and an unclipped line
    through one of them scribbles across the room and over the goal card. The plotter
    could not draw outside the page, and neither can the film.
    """
    r, g, b, a = lay.split()
    m = Image.new("L", lay.size, 0)
    ImageDraw.Draw(m).rectangle([int(v) for v in box], fill=255)
    return Image.merge("RGBA", (r, g, b, ImageChops.multiply(a, m)))


def path_layers(size, box, xy, pen, width):
    """(ghost of the whole tour, glow, ink) as transparent layers over `size`."""
    ghost = Image.new("RGBA", size, (0, 0, 0, 0))
    glow = Image.new("RGBA", size, (0, 0, 0, 0))
    sharp = Image.new("RGBA", size, (0, 0, 0, 0))
    pts = mapper(box)(xy[:, 0], xy[:, 1])
    seg = [tuple(p) for p in pts]
    ImageDraw.Draw(ghost).line(seg, fill=GHOST + (70,),
                               width=max(1, int(0.55 * width)), joint="curve")
    down = np.asarray(pen) > 0.5
    hit = np.flatnonzero(down[:-1] & down[1:])
    if len(hit):
        gd = ImageDraw.Draw(glow)
        for i in hit:
            gd.line([seg[i], seg[i + 1]], fill=(64, 86, 140, 150),
                    width=max(3, int(width * 3.0)), joint="curve")
        glow = glow.filter(ImageFilter.GaussianBlur(max(2.0, 1.4 * width)))
        sd = ImageDraw.Draw(sharp)
        for i in hit:
            sd.line([seg[i], seg[i + 1]], fill=INK + (255,),
                    width=max(2, int(width)), joint="curve")
    return tuple(clip(lay, box) for lay in (ghost, glow, sharp))


def frame(size, L, goal, state, hud, corner=None, title=TITLES):
    """One film frame from one sampler state: cards, ink, and the numbers measured on it."""
    xy = state[:, :2]
    pen = data.sign_to_pen(state[:, 2])
    img = Image.new("RGB", size, BG)
    paper_card(img, L["box"])
    goal_card(img, L["goal_box"], goal)
    rgba = img.convert("RGBA")
    for lay in path_layers(size, L["box"], xy, pen, L["width"]):
        rgba = Image.alpha_composite(rgba, lay)
    sc = L["scale"]
    if title:
        text_block(rgba, L["title_xy"], title, sc)
    text_block(rgba, L["goal_label_xy"], GOAL_LABEL, sc, color=(150, 168, 196))
    text_block(rgba, L["paper_label_xy"], L["paper_label"], sc, color=(150, 168, 196))
    text_block(rgba, L["hud_xy"], hud, sc)
    if corner:
        text_block(rgba, (L["hud_xy"][0],
                          L["hud_xy"][1] + block_h(hud, sc) + int(0.03 * size[1])),
                   corner, sc, color=(196, 208, 226))
    # the same grade the MuJoCo films are finished with, so the sampler and the machine
    # that draws its output share one room: same lifted shadows, same grain, same falloff.
    return (np.clip(render.grade(np.asarray(rgba.convert("RGB")), seed=0), 0, 1)
            * 255.0).astype(np.uint8)


def alpha_bar_at(pol, i, n_states):
    """The noise level of traced state `i`: state 0 is pure noise, the last is the path."""
    ab = pol.ab.cpu().numpy()
    return float(ab[max(len(ab) - 1 - i, 0)])


def goal_agreement(pol, goal, seed, device):
    """Mean |difference| between the path this goal produces and a blank one.

    The number the film is really selling: if swapping the picture changes nothing, the
    network is an unconditional prior wearing a goal as a hat, and every frame below is
    decoration. Measured here, on the checkpoint the film is made from, at run time.
    """
    real = pol.sample(torch.as_tensor(np.asarray(goal, np.float32)[None], device=device),
                      k=data.K, seed=seed)
    blank = pol.sample(torch.zeros(1, data.GOAL, data.GOAL, device=device), k=data.K,
                       seed=seed)
    return float((real - blank).abs().mean())


def state_metrics(goal, state):
    """(ink F1 against the goal, pen-down fraction, ink length in canvas units)."""
    path = metrics.as_path(state)
    r = data.raster_of_path(path)
    f1 = data.overlap(np.asarray(goal) > 0.5, r > 0.5)[2]
    down = path[:, 2] > 0.5
    hit = np.flatnonzero(down[:-1] & down[1:])
    xy = path[:, :2]
    ink = float(np.linalg.norm(np.diff(xy[hit], axis=0), axis=1).sum()) if len(hit) else 0.0
    return float(f1), float(down.mean()), ink


def gt_trace(pol, x0, seed):
    """The forward process run on a known path: q(x_t | x_0) at every level the sampler uses.

    Not a policy sample - nothing is predicted here. It exists so the film machinery
    (layout, clipping, the writer, the numbers) can be proven in a fresh environment on a
    five-second toy model, instead of only after a three-hour GPU run has finished and
    discovered a layout bug. The picture it makes is honest: the condensing is real, the
    model is not in it.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    # the generator has to be handed to the draw. `randn_like` without it reads the
    # global RNG, so every run of this film got a different first frame and the numbers
    # written beside it described a noise field that no longer exists.
    eps = torch.randn_like(torch.as_tensor(x0), generator=g)
    levels = pol.ab.flip(0).cpu().numpy().astype(np.float32)
    x0 = np.asarray(x0, np.float32)
    states = [(np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * eps.numpy().astype(np.float32))
              for ab in levels]
    states.append(x0)
    tr = np.stack(states).astype(np.float32)
    assert tr.shape == (pol.steps + 1, x0.shape[0], 3), "gt trace is %s" % (tr.shape,)
    return tr


def trace_of(pol, goal, seed):
    """(the (steps+1, K, 3) denoising trace, the final path) - and proof they agree."""
    g = torch.as_tensor(np.asarray(goal, np.float32)[None], device=pol.device)
    x, tr = pol.sample(g, k=data.K, seed=seed, trace=True)
    tr = tr[:, 0].cpu().numpy().astype(np.float32)
    fin = x[0].cpu().numpy().astype(np.float32)
    assert tr.shape == (pol.steps + 1, data.K, 3), "trace is %s, not noise plus every step" % (
        tuple(tr.shape),)
    assert np.isfinite(tr).all() and np.isfinite(fin).all(), "the film has a hole in it"
    assert np.allclose(tr[-1], fin, atol=1e-6), "the traced sampler lost the final path"
    return tr, fin


def curve(pol, tr, goal):
    """Per-step numbers, in the order the sampler visited them: noise first."""
    rows = []
    for i, st in enumerate(tr):
        f1, pdn, ink = state_metrics(goal, st)
        rows.append(dict(i=int(i), t_pol=int(len(tr) - 1 - i),
                         alpha_bar=alpha_bar_at(pol, i, len(tr)),
                         f1=f1, pen_down=pdn, ink_units=ink))
    return rows


def strip_png(tr, goal, path, picks=5, size=560):
    """A print-ready row: the noise, three stops on the way down, and the answer."""
    idx = sorted(set([0] + [int(round(p * (len(tr) - 1))) for p in
                            np.linspace(0.0, 1.0, picks)[1:-1]] + [len(tr) - 1]))
    tiles = []
    box = (int(.05 * size), int(.05 * size), int(.95 * size), int(.95 * size))
    for i in idx:
        img = Image.new("RGB", (size, size), BG)
        paper_card(img, box)
        for lay in path_layers((size, size), box, tr[i][:, :2],
                               data.sign_to_pen(tr[i][:, 2]), 0.006 * (box[2] - box[0])):
            img = Image.alpha_composite(img.convert("RGBA"), lay).convert("RGB")
        tiles.append(np.asarray(img))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(np.concatenate(tiles, axis=1)).save(path)
    return path, [int(i) for i in idx]


def hud_row(rows, i, n_states):
    """The three measured numbers that sit beside state i."""
    r = rows[i]
    return [("step %3d / %d    alpha_bar %.3f" % (i, n_states - 1, r["alpha_bar"]), 26),
            ("ink F1 vs goal %.3f    pen down %.0f%%" % (r["f1"], 100 * r["pen_down"]), 26),
            ("ink on this frame %.2f canvas units" % r["ink_units"], 26)]


def stills(tr, rows, goal, steps, size, out_dir, credits=None):
    """Three frames as PNGs.

    A composition bug found here costs a second; the same bug found in a film costs the
    twenty minutes it took to render, and these three frames are also the still images
    the report ships with.
    """
    L = layout(size, steps)
    hud0, hudn = hud_row(rows, 0, len(tr)), hud_row(rows, len(tr) - 1, len(tr))
    check_layout(L, size, hud0, credits)
    os.makedirs(out_dir, exist_ok=True)
    picks = [("noise", tr[0], hud0, None),
             ("mid", tr[len(tr) // 2], hud_row(rows, len(tr) // 2, len(tr)), None),
             ("final", tr[-1], hudn, credits)]
    names = []
    for tag, st, hud, corn in picks:
        p = os.path.join(out_dir, "denoise_still_%s.png" % tag)
        Image.fromarray(frame(size, L, goal, st, hud, corn)).save(p)
        names.append(p)
    return names


def pacing(n_states, sub, bias=PACING_BIAS):
    """How many frames each transition between traced states gets.

    A cosine schedule spends most of its steps looking like noise: the drawing assembles
    itself in the last third, where alpha-bar climbs steeply. Uniform pacing therefore
    gives the only interesting part of the reverse process a second of screen time and
    the boring part three. The weights hand the frames to the end of the schedule, while
    the step counter on screen keeps reporting the step the sampler is really on.
    """
    i = np.arange(n_states - 1, dtype=np.float64)
    w = (i / max(n_states - 2.0, 1.0)) ** bias + 0.10
    counts = np.maximum(1, np.round(w / w.sum() * (n_states - 1) * sub).astype(int))
    assert counts.sum() >= n_states - 1, "every transition still appears at least once"
    return counts


def build_film(tr, rows, goal, mp4, steps, size=(1920, 1080), sub=3, fps=30.0,
               hold_first=40, hold_last=100, credits=None):
    """Every state, eased into its neighbour, with the numbers that describe it on screen."""
    import imageio
    assert tr.shape[1] == data.K, "the trace holds %d points, the caption claims %d" % (
        tr.shape[1], data.K)
    L = layout(size, steps)
    assert len(tr) - 1 == steps and L["paper_label"][1][0].endswith("%d DDPM steps" % steps), \
        "the caption counts %d steps, the trace has %d" % (steps, len(tr) - 1)
    box, goal_box = L["box"], L["goal_box"]
    check_layout(L, size, [("step 000 / 000    alpha_bar 0.000", 26),
                           ("ink F1 vs goal 0.000    pen down 00%", 26),
                           ("ink on this frame 0.00 canvas units", 26)], credits)

    def hud_for(i):
        return hud_row(rows, i, len(tr))

    tmp = mp4 + ".part.mp4"          # imageio picks its backend from the extension
    wr = imageio.get_writer(tmp, fps=fps, codec="libx264", macro_block_size=1,
                            pixelformat="yuv420p", quality=8)
    n = 0
    try:
        for _ in range(hold_first):
            wr.append_data(frame(size, L, goal, tr[0], hud_for(0)))
            n += 1
        for i, cnt in enumerate(pacing(len(tr), sub)):
            for j in range(int(cnt)):
                p = (j + 1) / float(cnt)
                st = tr[i] * (1.0 - p) + tr[i + 1] * p
                wr.append_data(frame(size, L, goal, st, hud_for(i + 1), title=None))
                n += 1
        corner = credits or [("the path the plotter is handed", 24)]
        for _ in range(hold_last):
            wr.append_data(frame(size, L, goal, tr[-1], hud_for(len(tr) - 1), corner))
            n += 1
    except BaseException:
        wr.close()
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    wr.close()
    os.replace(tmp, mp4)
    return n


if __name__ == "__main__":
    want_film = "--film" in sys.argv
    draft = "--draft" in sys.argv
    # --gt films the forward process on the reference path: it needs no checkpoint and no
    # trained network, which is what makes the film machinery testable before a GPU run.
    gt = "--gt" in sys.argv
    size = (1280, 720) if draft else (1920, 1080)
    seed = int(os.environ.get("CAT_SAMPLE_SEED", 7))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    goal, ref, _ = pf.hero_pair()

    if gt:
        import model as M
        pol = M.DiffusionPath(M.UNet1D().to(dev), device=dev)
        x0 = np.asarray(ref, np.float32).copy()
        x0[:, 2] = data.pen_to_sign(x0[:, 2])
        ck, shift = {}, float("nan")
        t0 = time.time()
        tr = gt_trace(pol, x0, seed=0)
        fin = tr[-1]
        sample_sec = time.time() - t0
    else:
        pol, ck = pf.load(dev)
        t0 = time.time()
        tr, fin = trace_of(pol, goal, seed=seed)
        sample_sec = time.time() - t0

    rows = curve(pol, tr, goal)
    first, last = rows[0], rows[-1]
    assert last["ink_units"] < 0.6 * first["ink_units"], (
        "the program never condenses: %.2f u of ink on the noise frame, %.2f u at the end"
        % (first["ink_units"], last["ink_units"]))
    assert abs(first["alpha_bar"]) < 0.05 and last["alpha_bar"] > 0.95, (
        "the trace does not span noise to path: %.4f -> %.4f"
        % (first["alpha_bar"], last["alpha_bar"]))
    # an ink-F1 that starts high on the noise frame is not a bug: a scribble over the
    # whole page covers the goal's pixels, so it is precision that has to collapse. the
    # two controls that a film like this is allowed to guarantee are that the drawing
    # matches the picture it was given better than it matches somebody else's, and - for
    # a policy sample - that swapping the picture moves the path at all.
    other_goal = data.style_pair(4242)[0]
    cross = state_metrics(other_goal, tr[-1])[0]
    assert last["f1"] > cross + 0.02, (
        "the finished path scores %.3f against the goal it was handed and %.3f against a "
        "different cat: this drawing is not conditioned on the picture in the corner"
        % (last["f1"], cross))
    if not gt:
        shift = goal_agreement(pol, goal, seed, pol.device)
        assert shift > 0.02, (
            "this goal changes the path by only %.4f: the policy is drawing an unconditional "
            "prior and the picture in the corner is scenery" % shift)
    s = metrics.score(metrics.as_path(fin), goal, ref)
    print("DENOISE  %s  %d states  %.1f s on %s  alpha_bar %.4f -> %.4f"
          % ("reference trace (no policy)" if gt else "policy sample",
             len(tr), sample_sec, dev, first["alpha_bar"], last["alpha_bar"]))
    print("         ink F1 %.3f (noise) -> %.3f (final)   pen down %.0f%% -> %.0f%%"
          % (first["f1"], last["f1"], 100 * first["pen_down"], 100 * last["pen_down"]))
    print("         ink %.1f u (noise) -> %.2f u (final)   goal vs blank goal path shift %s"
          % (first["ink_units"], last["ink_units"],
             "n/a (reference trace)" if gt else "%.3f" % shift))
    print("         final path ade %.4f f1 %.3f chamfer %.4f strokes %d/%d"
          % (s["ade"], s["f1"], s["chamfer"], s["strokes"], s["strokes_ref"]))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "denoise_trace%s.json" % ("_gt" if gt else "")), "w") as f:
        json.dump(dict(ckpt=(None if gt else ck["ckpt_file"]),
                       trace=("forward-noised reference path, no policy involved" if gt
                              else "ancestral DDPM sample conditioned on the goal"),
                       ckpt_epoch=ck.get("epoch"),
                       ckpt_val_loss=float(ck.get("val_loss", float("nan"))),
                       device=str(dev), torch=torch.__version__, steps=pol.steps,
                       sample_sec=round(sample_sec, 2), resolution=list(size),
                       goal_vs_blank_shift=shift,
                       ink_units_noise=first["ink_units"], ink_units_final=last["ink_units"],
                       f1_noise=first["f1"], f1_final=last["f1"],
                       final_ade=float(s["ade"]), final_f1=float(s["f1"]),
                       final_chamfer=float(s["chamfer"]),
                       final_strokes=int(s["strokes"]),
                       final_strokes_ref=int(s["strokes_ref"]),
                       pacing_bias=PACING_BIAS,
                       per_step=rows), f, indent=1)
    strip, shown = strip_png(tr, goal, os.path.join(
        DESIGN, "denoise_strip%s.png" % ("_gt" if gt else "")))
    print("         strip %s at traced states %s" % (strip, shown))

    if "--stills" in sys.argv:
        credits = [("the path the plotter is handed", 24),
                   ("ink F1 %.3f   %d DDPM steps" % (last["f1"], pol.steps), 20)]
        for p in stills(tr, rows, goal, pol.steps, size, DESIGN, credits):
            print("STILL %s" % p)

    if want_film:
        mp4 = os.path.join(DESIGN, "cat_denoise%s.mp4" % ("_gt" if gt else
                                                          "_draft" if draft else ""))
        n = build_film(tr, rows, goal, mp4, pol.steps, size=size,
                       credits=[("the path the plotter is handed", 24),
                                ("ink F1 %.3f   %d DDPM steps" % (last["f1"], pol.steps), 20)])
        print("FILM %s  %d frames  %.1f s at %.0f fps  built in %.1f s"
              % (mp4, n, n / 30.0, 30.0, time.time() - t0))

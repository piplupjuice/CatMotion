"""The two-minute cut, for a viewer who is not going to watch three films.

`master_cut.py` assembles the archival argument in the order the pipeline was built. This
assembles the same footage in the order a first-time viewer needs it: the finished page
first, then the picture the network was given, then the noise resolving into a path, then
the machine running that path, then what was measured. It ends on the numbers and on what
they do not cover.

Nothing here is re-rendered. Every moving frame comes from a film that already exists,
every still is a file the pipeline already wrote, and every figure on a card is read out of
`out/` at build time. A source that is missing is skipped and printed. A card whose evidence
file is missing is skipped too, because a metric nobody measured cannot be put on screen.

The frame counts it reports are counted off the frames it actually wrote, and the delivered
file is measured again afterwards by `film_audit.py`.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "wgl" if sys.platform == "win32" else "egl")

import json
import time

import numpy as np
from PIL import Image

import master_cut as MC

ROOT, OUT, DESIGN = MC.ROOT, MC.OUT, MC.DESIGN
NAME = "cat_demo.mp4"
# A card has to be on screen long enough to be read, not long enough to be noticed:
# three or four lines of a metric is a five-second object, a two-line title is a three-second one.
TITLE, CARD, INFO, STILL = 90, 90, 150, 150


def load(evidence):
    """One JSON from out/, or None when it was never written."""
    p = os.path.join(OUT, evidence)
    if not os.path.isfile(p):
        print("SKIP card evidence %s: %s does not exist" % (evidence, p))
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def f(value, dec=3):
    return ("%." + str(dec) + "f") % float(value)


def fit_still(a):
    """A still, at its own aspect ratio, centred on the card background.

    `master_cut.fit` stretches, which is right for the three films - they are already this
    size - and wrong for stills, which are whatever shape their author drew them in. The
    goal-vs-prediction sheet is three by one; stretched to sixteen by nine it becomes a frame
    of elongated cats, and a viewer who has been told the policy draws a recognisable cat
    should not be shown a stretched one.
    """
    s = min(MC.SIZE[0] / a.shape[1], MC.SIZE[1] / a.shape[0])
    w, h = max(1, int(round(a.shape[1] * s))), max(1, int(round(a.shape[0] * s)))
    im = np.asarray(Image.fromarray(a).resize((w, h), Image.LANCZOS), np.float32) / 255.0
    canvas = np.zeros((MC.SIZE[1], MC.SIZE[0], 3), np.float32) + np.array(
        [0.055, 0.062, 0.082], np.float32)
    y0, x0 = (MC.SIZE[1] - h) // 2, (MC.SIZE[0] - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = im
    return (np.clip(canvas, 0, 1) * 255).astype(np.uint8)


def still(name, n=STILL):
    """A still the pipeline already wrote, held for n frames."""
    p = os.path.join(DESIGN, name)
    if not os.path.isfile(p):
        print("SKIP still %s: %s does not exist" % (name, p))
        return
    a = np.asarray(Image.open(p).convert("RGB"))
    frame = fit_still(a)
    for _ in range(n):
        yield frame


def film(name, lo=None, hi=None):
    """Frames [lo, hi) of an existing part, decoded once, forward."""
    import imageio

    p = os.path.join(DESIGN, name)
    if not os.path.isfile(p):
        print("SKIP film %s: %s does not exist" % (name, p))
        return
    r = imageio.get_reader(p)
    try:
        for i, fr in enumerate(r):
            if lo is not None and i < lo:
                continue
            if hi is not None and i >= hi:
                break
            yield MC.fit(np.asarray(fr))
    finally:
        r.close()


def cards():
    """{label: lines} for every card, built from the evidence files.

    Keyed by label rather than by position, because a card whose evidence file is missing
    loses its number lines and a positional cut would then silently show the wrong card
    under the wrong beat. The text is kept short because `render.overlay` centres the block
    and a line wider than the frame is a line the viewer cannot read.
    """
    ev = dict(denoise=load("denoise_trace.json"), draw=load("policy_draw.json"),
              select=load("checkpoint_select.json"), test=load("policy_eval.json"))
    c = {}

    c["01 title"] = [("CATMOTION", 52),
                    ("a goal-conditioned diffusion policy for robotic drawing,", 26),
                    ("a simulated pen plotter, and the ink it actually laid down", 26)]

    c["02 finished page"] = [("THE FINISHED PAGE", 52),
                             ("four cameras, one sheet, ink painted onto the paper", 26)]

    c["03 the goal"] = [("THE ONLY INPUT", 52),
                        ("a 48 x 48 picture of a cat", 26),
                        ("no stroke names, no oracle ordering, no target path", 26)]

    head = [("THE MODEL", 52), ("one picture in, one 512-point pen program out", 26)]
    d = ev["denoise"]
    if d:
        head += [("ink F1 %.3f to %.3f, ink on the page %s units to %s"
                  % (d["f1_noise"], d["f1_final"], f(d["ink_units_noise"], 1),
                     f(d["ink_units_final"], 2)), 26)]
    c["04 reverse diffusion"] = head

    c["05 the path"] = [("512 POINTS, IN ORDER", 52),
                        ("x, y and a pen carried at plus or minus one,", 26),
                        ("so one noise scale fits all three channels", 26)]

    p = ev["draw"]
    line = [("THE NETWORK'S OWN PATH", 52),
            ("rasterised back to a picture and scored against its goal", 26)]
    if p:
        m = p["metrics"]
        line += [("ink F1 %s, %d of %d strokes, pen agreement %s"
                  % (f(m["f1"]), int(m["strokes"]), int(m["strokes_ref"]),
                     f(m["pen"])), 26)]
    c["05 the path scored"] = line

    c["06 the machine"] = [("THE MACHINE", 52),
                           ("a MuJoCo gantry plotter, 33 substeps per 60 Hz frame,", 26),
                           ("pen contact read from mj_contactForce", 26)]

    drew = [("THE NETWORK DREW THIS ONE", 52),
            ("the same planner, the same machine, the same camera gates", 26)]
    if p:
        drew += [("worst stroke registration %s px, centroid %s px"
                  % (f(p["registration_worst_px"], 2), f(p["centroid_px"], 1)), 26)]
    c["07 the policy's program"] = drew

    t = ev["test"]
    if t:
        s = t["test"]
        c["08 evaluation"] = [("HELD-OUT TEST, 32 GOALS", 52),
                              ("ink F1 %s   chamfer %s   ADE %s"
                               % (f(s["f1"][0]), f(s["chamfer"][0]), f(s["ade"][0])), 26),
                              ("%s strokes of %s   pen agreement %s"
                               % (f(s["strokes"][0], 1), f(s["strokes_ref"][0], 1),
                                  f(s["pen"][0])), 26)]

    k = ev["select"]
    if k:
        b, l = k["candidates"]["policy_best.pt"], k["candidates"]["policy_last.pt"]
        c["09 checkpoint"] = [("THE CHECKPOINT PROBLEM", 52),
                              ("epoch %d: val loss %s, ink F1 %s, %s strokes"
                               % (b["epoch"], f(b["val_loss"]), f(b["val"]["f1"]),
                                  f(b["val"]["strokes"], 1)), 26),
                              ("epoch %d: val loss %s, ink F1 %s, %s strokes"
                               % (l["epoch"], f(l["val_loss"]), f(l["val"]["f1"]),
                                  f(l["val"]["strokes"], 1)), 26),
                              ("the shipping choice was made on the drawing, not the loss", 26)]

    c["10 limitations"] = [("WHAT THIS IS NOT", 52),
                           ("no closed loop: the program is sampled once and run open loop", 26),
                           ("no physical plotter: dynamics, contact and camera are simulated", 26),
                           ("one cat in many styles, and no cross-category claim", 26)]

    c["11 closing"] = [("EVERY NUMBER ON THIS FILM", 52),
                       ("is in a log under out/, and src/verify.py fails the build", 26),
                       ("if the prose quotes a figure that is not in one", 26)]
    return c


# the cut, in the order a first-time viewer needs it. each entry is (kind, label, payload):
# a card is held for CARD frames, a still for STILL, a film for whatever it has.
CUT = [
    ("card", "01 title", TITLE),
    ("card", "02 finished page", CARD),
    ("still", "02 finished page", "render_stills_done.png"),
    ("card", "03 the goal", CARD),
    ("still", "03 the goal", "goal_sheet.png"),
    ("card", "04 reverse diffusion", INFO),
    ("film", "04 reverse diffusion", "cat_denoise.mp4"),
    ("card", "05 the path", CARD),
    ("still", "05 the path scored", "policy_sheet_hero.png"),
    ("card", "06 the machine", CARD),
    ("film", "06 the machine", "cat_film.mp4"),
    ("card", "07 the policy's program", INFO),
    ("film", "07 the policy's program", "policy_film_hero.mp4"),
    ("card", "08 evaluation", INFO),
    ("card", "09 checkpoint", INFO),
    ("card", "10 limitations", INFO),
    ("card", "11 closing", INFO),
]


def build(path=None, delivery=None, cut=CUT):
    import imageio

    path = path or os.path.join(DESIGN, NAME)
    delivery = delivery or os.path.join(DESIGN, NAME.replace(".mp4", "_delivery.mp4"))
    texts = cards()
    plan, frames = [], 0
    tmp = path + ".part.mp4"
    wr = imageio.get_writer(tmp, fps=MC.FPS, codec="libx264", macro_block_size=1,
                            pixelformat="yuv420p", quality=8)
    try:
        for kind, label, payload in cut:
            if label in texts and kind == "card":
                lines = texts[label]
                gen = (MC.card_frame(lines) for _ in range(payload))
            elif kind == "still":
                gen = still(payload)
            elif kind == "film":
                gen = film(payload)
            else:
                print("SKIP   %s: no card with that label" % label)
                continue
            n = 0
            for fr in gen:
                wr.append_data(fr)
                n += 1
            if not n:
                print("SKIP   %s %s produced no frames" % (kind, label))
                continue
            plan.append(dict(kind=kind, label=label, frames=int(n),
                             seconds=round(n / MC.FPS, 2)))
            frames += n
            print("SEG    %-6s %-24s %5d frames  %6.1f s"
                  % (kind, label[:24], n, n / MC.FPS))
    except BaseException:
        wr.close()
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    wr.close()
    if not plan:
        os.remove(tmp)
        raise SystemExit("nothing to cut - no source film or still exists")
    os.replace(tmp, path)

    t1 = time.time()
    MC.transcode(path, delivery)
    doc = dict(source="existing films and stills, nothing re-rendered",
               fps=MC.FPS, resolution=list(MC.SIZE), segments=plan,
               cards={k: [t for t, _ in v] for k, v in texts.items()},
               total_frames=int(frames), seconds=round(frames / MC.FPS, 2),
               mp4=os.path.relpath(path, ROOT).replace("\\", "/"),
               delivery=os.path.relpath(delivery, ROOT).replace("\\", "/"),
               master_bytes=os.path.getsize(path),
               delivery_bytes=os.path.getsize(delivery),
               transcode_sec=round(time.time() - t1, 1))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "demo_cut.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print("DEMO  %s  %d frames  %.1f s  %d segments"
          % (path, frames, frames / MC.FPS, len(plan)))
    return doc


if __name__ == "__main__":
    t0 = time.time()
    build()
    print("built in %.1f s" % (time.time() - t0))

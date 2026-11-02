"""One film, in the order the argument is actually made.

Part 1 is the model: a 48x48 picture and a pen program condensing out of noise. Part 2 is
the machine: a program executed by a simulated gantry plotter, with contact forces. Part 3
is both at once - the network's own program, drawn by the same machine. Each part already
exists as its own mp4; this only decodes them, puts a title card in front of each, and
re-encodes one continuous file at one resolution.

It reports what it found. A missing part is skipped and printed, never faked, and the
frame counts and durations it writes to out/master_cut.json are counted off the decoded
frames, not off whatever the parts claimed about themselves.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "wgl" if sys.platform == "win32" else "egl")

import json
import time

import numpy as np
from PIL import Image

import render

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# numbers to out/, pictures to design_out/ - the rule render.py and policy_film.py use
OUT = os.environ.get("CAT_OUT") or os.path.join(ROOT, "out")
DESIGN = os.environ.get("CAT_DESIGN") or os.path.join(ROOT, "design_out")

SIZE = (1920, 1080)
FPS = 30.0
CARD = 45          # frames of title card between parts

PARTS = [
    ("cat_denoise.mp4", "PART 1   THE MODEL",
     "one 48 x 48 picture in, one 512-point pen program out"),
    ("cat_film.mp4", "PART 2   THE MACHINE",
     "a gantry plotter executes the program, pen on paper, contact measured"),
    ("policy_film_hero.mp4", "PART 3   BOTH",
     "the network's own program, drawn by the same machine"),
]


def card_frame(lines, size=SIZE):
    a = np.zeros((size[1], size[0], 3), np.float32) + np.array([0.055, 0.062, 0.082], np.float32)
    # "mc": the block sits on the optical centre of the frame. A card is all picture, so
    # the bottom-anchored placement the captions use would park the only words on it.
    return (np.clip(render.overlay(a, [(t, s) for t, s in lines], "mc"), 0, 1) * 255
            ).astype(np.uint8)


def fit(frame, size=SIZE):
    """Any part, whatever it was rendered at, into the master's frame."""
    if frame.shape[:2] != (size[1], size[0]):
        frame = np.asarray(Image.fromarray(frame).resize(size, Image.LANCZOS))
    return frame


def transcode(src, dst, crf=20):
    """A copy of the master small enough to send, made from the same encoded frames.

    The archival cut is written at imageio's top quality and lands in the gigabytes,
    which is right for a file you keep and wrong for one you show somebody. This is a
    stream copy through x264 at a constant rate with the index moved to the front, so
    it plays while it downloads. Nothing is re-rendered: same frames, same length.
    """
    import subprocess

    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = dst + ".part.mp4"
    try:
        subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", tmp],
                       check=True)
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return dst


def build(path=None, parts=PARTS):
    import imageio
    path = path or os.path.join(DESIGN, "cat_master.mp4")
    found = [(n, h, s) for (n, h, s) in parts if os.path.isfile(os.path.join(DESIGN, n))]
    for n, _, _ in parts:
        if not os.path.isfile(os.path.join(DESIGN, n)):
            print("SKIP part %s: %s does not exist" % (n, os.path.join(DESIGN, n)))
    if not found:
        raise SystemExit("no part of the film exists yet - nothing to cut")
    plan, frames = [], 0
    tmp = path + ".part.mp4"        # imageio picks its backend from the extension
    wr = imageio.get_writer(tmp, fps=FPS, codec="libx264", macro_block_size=1,
                            pixelformat="yuv420p", quality=8)
    try:
        for name, head, sub in found:
            p = os.path.join(DESIGN, name)
            title = [(head, 52), (sub, 26)]
            for _ in range(CARD):
                wr.append_data(card_frame(title))
            n = 0
            for f in imageio.get_reader(p):
                wr.append_data(fit(np.asarray(f)))
                n += 1
            if not n:
                print("SKIP part %s: decoded 0 frames from %s" % (head, p))
                continue
            plan.append(dict(part=head, file=name, frames=int(n),
                             seconds=round(n / FPS, 2)))
            frames += n + CARD
            print("PART   %s  %d frames  %.1f s" % (head, n, n / FPS))
    except BaseException:
        wr.close()
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    wr.close()
    if not plan:
        os.remove(tmp)
        raise SystemExit("every part decoded zero frames - nothing to cut")
    os.replace(tmp, path)
    total = frames
    delivery = os.path.join(DESIGN, "cat_master_delivery.mp4")
    t1 = time.time()
    transcode(path, delivery)
    doc = dict(parts=plan, card_frames=CARD, total_frames=int(total),
               seconds=round(total / FPS, 2), resolution=list(SIZE), fps=FPS,
               mp4=os.path.relpath(path, ROOT).replace("\\", "/"),
               delivery=os.path.relpath(delivery, ROOT).replace("\\", "/"),
               master_bytes=os.path.getsize(path),
               delivery_bytes=os.path.getsize(delivery),
               transcode_sec=round(time.time() - t1, 1))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "master_cut.json"), "w") as f:
        json.dump(doc, f, indent=1)
    print("MASTER %s  %d frames  %.1f s  %d parts" % (path, total, total / FPS, len(plan)))
    return doc


if __name__ == "__main__":
    t0 = time.time()
    build()
    print("built in %.1f s" % (time.time() - t0))

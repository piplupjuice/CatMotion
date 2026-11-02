"""Measure the delivered videos rather than trust the code that wrote them.

A film is the part of this project most people watch and the part nobody checks. The
builder prints "293 frames", but that number comes from a counter inside the loop that
made the file: if the writer were interrupted, if the container lost its index, or if a
draft were renamed over a final cut, the counter would still say whatever it intended.
This reads the finished container back - frame count, size, rate, bitrate, digest - and
writes it as JSON, so every frame count and resolution quoted in the report is a property
of the file that exists, not of the code that hoped to make it.

A `.part.mp4` is never audited: the renderers write under that name and move it into
place only after the encoder closed, so a half film present at all means a run died.
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

import imageio_ffmpeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESIGN = os.path.join(ROOT, "design_out")
OUT = os.environ.get("CAT_OUT") or os.path.join(ROOT, "out")

STATS = re.compile(r"frame=\s*(\d+).*?time=(\d+):(\d+):(\d+(?:\.\d+)?)")
STREAM = re.compile(r"(\d{3,5})x(\d{3,5}).+?(\d+) kb/s, ([\d.]+) fps")

# one line of each kind that ffmpeg really printed. a reader whose pattern quietly stops
# matching reports nothing at all, which is the failure mode this tool exists to prevent,
# so the patterns are pointed at a fixture before they are pointed at a six-hundred-MB file
_SAMPLE = """  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 1920x1080, 12329 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
frame=  293 fps=0.0 q=-1.0 Lsize=N/A time=00:00:09.70 bitrate=N/A speed= 951x"""


def self_test():
    st, fs = STREAM.search(_SAMPLE), STATS.search(_SAMPLE)
    assert st and fs, "the reader's own patterns match nothing"
    assert st.groups() == ("1920", "1080", "12329", "30"), st.groups()
    assert (fs.group(1), fs.group(4)) == ("293", "09.70"), fs.groups()



def probe(exe, path):
    """(frames, fps, width, height, kb_s) read off the container by demuxing it.

    Stream copy with a null muxer walks every packet and decodes nothing, so the frame
    count is the container's own and the pass costs a second rather than a render.
    """
    r = subprocess.run([exe, "-hide_banner", "-i", path, "-map", "0:v:0", "-c", "copy",
                        "-f", "null", "-"], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("%s does not demux: %s" % (
            os.path.basename(path), r.stderr.strip().splitlines()[-1] if r.stderr else ""))
    st = STREAM.search(r.stderr)
    # ffmpeg answers with a stats line and then a muxer summary, so the frame count is
    # the last stats line in the whole block rather than simply its final line.
    fs = None
    for line in r.stderr.splitlines():
        m = STATS.search(line)
        if m:
            fs = m
    if not st or not fs:
        raise RuntimeError("%s: could not read the stream or the frame count" % path)
    h, m, s = int(fs.group(2)), int(fs.group(3)), float(fs.group(4))
    return dict(frames=int(fs.group(1)), fps=float(st.group(4)),
                width=int(st.group(1)), height=int(st.group(2)),
                kb_s=int(st.group(3)),
                container_sec=round(h * 3600 + m * 60 + s, 2))


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(buf), b""):
            h.update(block)
    return h.hexdigest()


def audit(paths, exe=None):
    exe = exe or imageio_ffmpeg.get_ffmpeg_exe()
    rows = {}
    for p in sorted(paths):
        d = probe(exe, p)
        d.update(bytes=os.path.getsize(p),
                 mb=round(os.path.getsize(p) / 1e6, 1),
                 seconds=round(d["frames"] / d["fps"], 2),
                 sha256=sha256(p)[:16],
                 mtime=os.path.getmtime(p))
        assert d["frames"] > 0 and d["kb_s"] > 0, "%s has no picture" % p
        assert abs(d["seconds"] - d["container_sec"]) < 1.0, (
            "%s: %d frames at %.2f fps is %.2f s but the container says %.2f s"
            % (os.path.basename(p), d["frames"], d["fps"], d["seconds"],
               d["container_sec"]))
        rows[os.path.relpath(p, ROOT).replace("\\", "/")] = d
        print("%-34s %5d frames  %dx%d  %.2f s  %6.1f MB  %5d kb/s  %s"
              % (os.path.basename(p), d["frames"], d["width"], d["height"],
                 d["seconds"], d["mb"], d["kb_s"], d["sha256"]))
    return rows


if __name__ == "__main__":
    self_test()
    paths = [p for p in glob.glob(os.path.join(DESIGN, "*.mp4"))
             if not p.endswith(".part.mp4")]
    assert paths, "no finished film in %s to measure" % DESIGN
    rows = audit(paths)
    os.makedirs(OUT, exist_ok=True)
    doc = dict(ffmpeg=imageio_ffmpeg.get_ffmpeg_version(), n_files=len(rows),
               total_mb=round(sum(r["mb"] for r in rows.values()), 1),
               files=rows)
    with open(os.path.join(OUT, "films.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    print("FILMS   %d files, %.1f MB, ffmpeg %s -> out/films.json"
          % (len(rows), doc["total_mb"], doc["ffmpeg"]))

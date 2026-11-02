"""Re-run every claim this project makes, and refuse to let prose outrun evidence.

Two separate jobs, because they fail in different ways:

* `checks` runs each module's own self-test in a subprocess and keeps the printed
  line. A module that asserts on itself cannot quietly rot; the line it prints is
  the number the report is allowed to quote.
* `audit` reads the markdown and every number in it, then demands that the same
  number appear in a machine-written log. A figure that is not in a log is a
  figure nobody measured, and this exits non-zero over it.

Nothing here invents a value. The only inputs are subprocess stdout and the JSON
files the training and rendering runs wrote themselves.
"""
import glob
import json
import os
import re
import subprocess
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
OUT = os.environ.get("CAT_OUT") or os.path.join(ROOT, "out")

# modules that only need numpy/torch, then the ones that need a GL context
SELF_TESTS = ["geometry", "art", "planner", "data", "model", "metrics"]
PHYSICS_TESTS = ["robot", "render"]

# files the finished project must contain, whatever the prose says about it
REQUIRED = [
    "src/geometry.py", "src/art.py", "src/planner.py", "src/robot.py",
    "src/render.py", "src/data.py", "src/model.py", "src/metrics.py",
    "src/train.py", "src/policy_film.py", "src/denoise_film.py", "src/ckpt_select.py",
    "src/master_cut.py", "src/film_audit.py", "src/verify.py",
]


def gl_backend():
    """Windows renders through wgl; Linux (Kaggle) has only egl. Honour an explicit
    MUJOCO_GL, otherwise pick by platform so the same file runs in both places."""
    return os.environ.get("MUJOCO_GL") or ("wgl" if sys.platform == "win32" else "egl")


def run_module(name, timeout=3600):
    """(ok, stdout) from one module's `__main__` block."""
    env = dict(os.environ, MUJOCO_GL=gl_backend(), PYTHONUNBUFFERED="1")
    try:
        p = subprocess.run([sys.executable, "-u", name + ".py"], cwd=SRC, env=env,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "%s: timed out after %d s" % (name, timeout)
    text = (p.stdout or "") + (p.stderr or "")
    return p.returncode == 0, text


def collect_logs():
    """Every number that exists in a machine-written log, plus the log lines."""
    nums, lines = [], []
    for path in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        try:
            with open(path) as f:
                blob = json.load(f)
        except (ValueError, OSError) as e:
            print("  ! %s unreadable: %s" % (path, e))
            continue

        def walk(x):
            if isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    walk(v)
            elif isinstance(x, bool):
                pass
            elif isinstance(x, (int, float)):
                nums.append(float(x))

        walk(blob)
        lines.append(os.path.basename(path))
    return nums, lines


NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
# a measurement is a number standing alone. "1080p", "v6", "2.15.0+cpu" and the digits
# inside a file name are identifiers, not figures anybody measured, and treating them as
# claims would train the reader to ignore the audit.
CLAIM = re.compile(r"(?<![A-Za-z0-9_.+-])" + NUMBER.pattern + r"(?![A-Za-z0-9_.])")
SKIP_LINE = re.compile(r"^\s*(?:```|\||https?://|!\[)")
INLINE_CODE = re.compile(r"`[^`]*`")
# a backtick is two different things in these documents. `out/policy_eval.json` and
# `policy_best.pt` name a file; `0.6905` and `3.69 M` state a measurement and only look
# tidy. Blank the names, keep the numbers - otherwise a report can put every figure it
# ever invented inside backticks and this audit would read a clean bill of health.
IDENTIFIER = re.compile(r"[/\\]|\.(?:%s)\b" % "|".join(
    ("py", "pt", "mp4", "json", "md", "txt", "png", "jpg", "ipynb", "yaml", "toml")))
# "section 2.6" and "§4" point at a heading in the same document; they are not quantities
# and no log could ever contain one. The same goes for the number a heading starts with.
SECTION = re.compile(r"\b(?:section|sections|§)\s*\d+(?:\.\d+)*", re.IGNORECASE)
HEADING_NUM = re.compile(r"^(\s*#{1,6}\s*)\d+(?:\.\d+)*\s")


def prose_numbers(path):
    """Numeric literals in the prose, with the line they came from.

    Table rows and fenced code are skipped: a table is quoting another document and
    code is a command, not a measurement. So are file names and section pointers. What
    is left is a measurement claim, however it is punctuated.
    """
    out, fence = [], False
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if line.startswith("```"):
                fence = not fence
                continue
            if fence or SKIP_LINE.match(line):
                continue
            line = SECTION.sub(" ", line)
            line = HEADING_NUM.sub(r"\1", line)
            text = re.sub(INLINE_CODE,
                          lambda m: " " if IDENTIFIER.search(m.group()) else m.group(),
                          line)
            for m in CLAIM.finditer(text):
                s = m.group()
                if re.match(r"^\d+$", s) and len(s) <= 2:
                    continue        # "23 strokes" is checked in full; "3" is prose noise
                out.append((ln, s))
    return out


def env_report():
    """The machine's own version numbers, written down rather than typed in.

    A report that says "MuJoCo 3.17" is quoting somebody's memory unless something
    measured it. This does, and hands the same numbers to the prose audit, so the
    documentation can state its environment without inventing it.
    """
    import platform
    info = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cores": os.cpu_count(),
        "numpy": __import__("numpy").__version__,
        "pillow": __import__("PIL.Image").__version__,
    }
    for name in ("torch", "mujoco", "imageio", "imageio_ffmpeg", "matplotlib"):
        try:
            info[name] = __import__(name).__version__
        except Exception as e:  # not installed here is a fact worth recording
            info[name] = "absent (%s)" % type(e).__name__
    # the encoder is a binary, not a package, so its version has to be read off the
    # executable that the films were actually written with.
    try:
        info["ffmpeg"] = __import__("imageio_ffmpeg").get_ffmpeg_version()
    except Exception as e:
        info["ffmpeg"] = "absent (%s)" % type(e).__name__
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "env.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, sort_keys=True)
    versions = [v for v in info.values() if isinstance(v, str)]
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", " ".join(versions))]
    return info, nums + [float(info["cores"])]


def rounds_to(value, text):
    """True if `value` rounds to the literal `text` at the literal's own precision."""
    dec = len(text.split(".")[1]) if "." in text else 0
    if "e" in text.lower():
        return abs(value - float(text)) <= 1e-9 * max(abs(value), 1.0)
    try:
        return abs(round(value, dec) - float(text)) <= 10.0 ** (-dec) / 2.0 + 1e-12
    except ValueError:
        return False


def audit(md_paths, log_nums, extra_text=""):
    """Each quoted number must be a correctly-rounded quote of a logged number."""
    pool = list(log_nums)
    for s in NUMBER.findall(extra_text):
        try:
            pool.append(float(s))
        except ValueError:
            pass
    bad, n_ok = [], 0
    for path in md_paths:
        for ln, text in prose_numbers(path):
            v = float(text)
            if any(rounds_to(p, text) for p in pool):
                n_ok += 1
            else:
                bad.append("%s:%d  %s" % (os.path.basename(path), ln, text))
    return n_ok, bad


def main():
    want_physics = "--fast" not in sys.argv and "--selftest" not in sys.argv
    # --selftest is what the Kaggle notebook runs: the modules proving themselves. It
    # deliberately skips the documentation audit, because at that point in a run the
    # reports have not been written yet - "no markdown found" is not a failed test.
    selftest = "--selftest" in sys.argv
    docs = [] if selftest else sorted(glob.glob(os.path.join(ROOT, "*.md")))

    transcript = []
    failures = []
    for name in (SELF_TESTS + (PHYSICS_TESTS if want_physics else [])):
        ok, text = run_module(name)
        last = [l for l in text.strip().splitlines() if l.strip()]
        print("%-9s %s  %s" % (name, "PASS" if ok else "FAIL",
                               last[0] if last else "(no output)"))
        transcript.append(text)
        if not ok:
            failures.append(name)

    missing = [] if selftest else [r for r in REQUIRED
                                   if not os.path.exists(os.path.join(ROOT, r))]
    for r in missing:
        print("MISSING %s" % r)

    log_nums, log_files = collect_logs()
    env, env_nums = env_report()
    print("ENV     python %s  torch %s  mujoco %s  %d cores"
          % (env["python"], env.get("torch"), env.get("mujoco"), env["cores"]))
    print("LOGS    %d numbers from %s" % (len(log_nums), ", ".join(log_files) or "none"))

    if docs:
        n_ok, bad = audit(docs, log_nums + env_nums, "\n".join(transcript))
        print("PROSE   %d numbers match a log, %d do not" % (n_ok, len(bad)))
        for b in bad[:40]:
            print("  unlogged  %s" % b)
    elif selftest:
        bad = []
        print("PROSE   skipped (--selftest: the reports are written after this stage)")
    else:
        bad = ["no markdown found"]
        print("PROSE   no documentation to audit")

    with open(os.path.join(OUT, "verify_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(transcript))

    if failures or missing or bad:
        print("VERIFY FAIL  self-tests %d  missing %d  unlogged numbers %d"
              % (len(failures), len(missing), len(bad)))
        return 1
    print("VERIFY PASS  %d self-tests, %d logged numbers, %d docs audited"
          % (len(transcript), len(log_nums), len(docs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

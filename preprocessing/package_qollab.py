"""Build dist/floodsight-q-improve.zip for Qollab's "Upload your own files" (checks Qollab's limits).

    .venv/bin/python preprocessing/package_qollab.py
"""

import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# Python project: main.py at the top level and NO index.html/main.js (those would make Qollab run it as JS).
INCLUDE_TOP = ["main.py", "README.md", "LICENSE", "requirements.txt"]
INCLUDE_DIRS = ["fsq", "data", "results", "preprocessing", "tests"]
SKIP = ("__pycache__", ".pyc", ".DS_Store")
MAX_FILES, MAX_BYTES = 100, 2 * 1024 * 1024


def files():
    out = [f for f in INCLUDE_TOP]
    for d in INCLUDE_DIRS:
        for base, _, names in os.walk(d):
            out += [os.path.join(base, n) for n in sorted(names) if not any(s in os.path.join(base, n) for s in SKIP)]
    return out


def main():
    fs = files()
    assert not any(f in fs for f in ("index.html", "main.js")), "would run as a JS project"
    big = [(f, os.path.getsize(f)) for f in fs if os.path.getsize(f) > MAX_BYTES]
    assert len(fs) <= MAX_FILES, f"{len(fs)} files > {MAX_FILES}"
    assert not big, f"files over 2 MB: {big}"
    os.makedirs("dist", exist_ok=True)
    with zipfile.ZipFile("dist/floodsight-q-improve.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in fs:
            z.write(f, f)
    total = sum(os.path.getsize(f) for f in fs)
    print(f"dist/floodsight-q-improve.zip: {len(fs)} files, {total / 1024:.0f} KB uncompressed, "
          f"largest {max(fs, key=os.path.getsize)} ({os.path.getsize(max(fs, key=os.path.getsize)) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

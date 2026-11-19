"""Keep every version of a scene: copy its latest preview render to the next free version number.

    python video/save_version.py video/scenes/s0_1_cold_open.py Scene_0_1 [-m "note"]

-> renders/scenes/s0_1_cold_open/s0_1_cold_open-v01.mp4, -v02.mp4, ...  (never overwrites)
   and appends a line to renders/scenes/s0_1_cold_open/versions.log
"""

import argparse
import datetime
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def latest_render(stem: str, cls: str) -> Path:
    candidates = list((ROOT / "media" / "videos" / stem).glob(f"*/{cls}.mp4"))
    if not candidates:
        raise SystemExit(f"no render found for {stem}/{cls} - render it with manim first")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_file")
    ap.add_argument("cls")
    ap.add_argument("-m", "--message", default="")
    a = ap.parse_args()

    stem = Path(a.scene_file).stem
    src = latest_render(stem, a.cls)
    out_dir = ROOT / "renders" / "scenes" / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 1
    while (out_dir / f"{stem}-v{n:02d}.mp4").exists():
        n += 1
    dst = out_dir / f"{stem}-v{n:02d}.mp4"
    shutil.copy2(src, dst)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    with open(out_dir / "versions.log", "a", encoding="utf-8") as f:
        f.write(f"v{n:02d}\t{datetime.datetime.now():%Y-%m-%d %H:%M}\t{src.parent.name}\t"
                f"after {commit}\t{a.message}\n")
    print(dst)


if __name__ == "__main__":
    main()

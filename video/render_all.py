"""Assemble the main video from the LATEST saved version of every scene, in story order.

    .venv/Scripts/python video/render_all.py

-> renders/render-NN.mp4 (next free number, never overwrites) + render-NN.txt listing the
   scene versions used. Scenes without a saved version are skipped, so a partial video works.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENES = ROOT / "renders" / "scenes"
ORDER = ["s0_1_cold_open", "s1_1_family", "s1_2_big_picture", "s1_3_block", "s1_4_attention",
         "s1_5_irope", "s1_6_moe", "s2_1_shrink", "s2_2_layers", "s2_3_data", "s3_1_targets",
         "s3_2_embed", "s3_3_loop", "s3_4_attn", "s3_5_ffn", "s3_6_logits", "s3_7_loss",
         "s3_8_backward", "s3_9_step", "s4_1_outro"]


def latest_version(stem: str):
    files = list((SCENES / stem).glob(f"{stem}-v*.mp4")) if (SCENES / stem).exists() else []
    if not files:
        return None
    return max(files, key=lambda p: int(re.search(r"-v(\d+)\.mp4$", p.name).group(1)))


def main():
    ffmpeg = shutil.which("ffmpeg") or sys.exit("ffmpeg not found on PATH")
    parts = [(s, latest_version(s)) for s in ORDER]
    used = [(s, p) for s, p in parts if p]
    if not used:
        sys.exit("no scene versions saved yet")

    n = 1
    while (ROOT / "renders" / f"render-{n:02d}.mp4").exists():
        n += 1
    out = ROOT / "renders" / f"render-{n:02d}.mp4"

    # normalize every scene to 1080p60 + 48 kHz stereo, then concatenate
    args = [ffmpeg, "-y", "-loglevel", "error"]
    for _, p in used:
        args += ["-i", str(p)]
    chains, labels = [], ""
    for i in range(len(used)):
        chains.append(f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
                      f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=60,setsar=1[v{i}]")
        chains.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo[a{i}]")
        labels += f"[v{i}][a{i}]"
    graph = ";".join(chains) + f";{labels}concat=n={len(used)}:v=1:a=1[v][a]"
    args += ["-filter_complex", graph, "-map", "[v]", "-map", "[a]",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "192k", str(out)]
    subprocess.run(args, check=True)

    report = [f"{out.name}"] + [f"  {s:20s} {p.name}" for s, p in used] + \
             [f"  {s:20s} (missing)" for s, p in parts if not p]
    out.with_suffix(".txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()

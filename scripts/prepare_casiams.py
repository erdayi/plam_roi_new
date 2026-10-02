# -*- coding: utf-8 -*-
"""CASIA-MS 扁平目录 -> ImageFolder(subject_hand ID).

命名: 001_l_460_01.jpg = 001号_左手_460nm_第01张
ID 规则: <subject>_<hand>, 全部 6 个光谱波段归入同一 ID (光谱差异=天然增广).
"""
import argparse
import shutil
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="E:/palm/data/CASIA-MS")
    ap.add_argument("--dst", default="E:/data/phaseA/casia_input")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)

    n = 0
    for f in sorted(src.glob("*.jpg")):
        m = re.match(r"(\d+)_(\w+)_(\w+)_(\d+)\.(jpg|JPG)", f.name)
        if not m:
            print("skip:", f.name)
            continue
        sid, hand = m.group(1), m.group(2)
        out = dst / f"{sid}_{hand}"
        out.mkdir(parents=True, exist_ok=True)
        target = out / f.name
        if not target.exists():
            shutil.copy2(f, target)
        n += 1
    ids = len(list(dst.iterdir()))
    print(f"organized {n} images into {ids} IDs -> {dst}")


if __name__ == "__main__":
    main()

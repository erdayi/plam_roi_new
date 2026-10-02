# -*- coding: utf-8 -*-
"""Phase A 数据准备: 把采集库整理成 ImageFolder(subject_hand ID).

适配 500w 布局: <src>/<subject>/{l_,r_,b_,c_}xxx.png
  - l_/r_ 前缀 → 左右手各算一个 ID (与 palm_roi 001_left/001_right 惯例一致)
  - 无前缀 → 整个 subject 一个 ID
输出: <dst>/<subject_hand>/*.png  (供 palm_roi CLI 批量裁 ROI, 裁完即训练用 ImageFolder)

用法:
  python scripts/prepare_phaseA.py --src "E:/个人资料/研究生资料/掌纹ROI/500w" \
      --dst "E:/data/phaseA/rois_input"
之后:
  cd E:/code/palm_roi
  python main.py -i E:/data/phaseA/rois_input -o E:/data/phaseA/rois_112 --roi_size 112
  python E:/code/roi-new/scripts/train_arcface.py --data_root E:/data/phaseA/rois_112
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--copy", action="store_true", help="复制而非软链/移动")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    n_files = n_ids = 0
    for subject_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        for f in subject_dir.iterdir():
            if not f.is_file() or f.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            hand = "left" if f.name.startswith(("l_", "L_")) else \
                   "right" if f.name.startswith(("r_", "R_")) else None
            class_id = f"{subject_dir.name}_{hand}" if hand else subject_dir.name
            out_dir = dst / class_id
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / f.name
            if args.copy:
                shutil.copy2(f, target)
            else:
                try:
                    target.symlink_to(f.resolve())
                except OSError:  # Windows 无符号链接权限时回退复制
                    shutil.copy2(f, target)
            n_files += 1
    n_ids = len(list(dst.iterdir()))
    print(f"organized {n_files} images into {n_ids} IDs -> {dst}")
    print("next: palm_roi CLI 批量裁 112x112 ROI, 然后 train_arcface.py --data_root <roi输出目录>")


if __name__ == "__main__":
    main()

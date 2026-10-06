# -*- coding: utf-8 -*-
"""GT 坐标落点自检: 把 json 谷点画到图上, 输出网格拼图供肉眼核查.

用法 (那台机器):
  python scripts/visualize_annotations.py --data_root D:/dataset/roi/MobileNet --n 20
输出: runs/gt_check/grid.jpg  (每格: 图缩略 + 红点=标注谷点)
判读: 红点应正好落在指缝底部; 飘到半空/图外/边缘的格子 = 该图 GT 有问题.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from palmroi_kpt.datasets.keypoint_dataset import walk_pairs, parse_annotation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="D:/dataset/roi/MobileNet")
    ap.add_argument("--out", default="runs/gt_check/grid.jpg")
    ap.add_argument("--n", type=int, default=20, help="抽查张数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = walk_pairs(args.data_root)
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.n]
    print(f"drawing {len(pairs)} samples")

    cell, cols = 256, 5
    rows = (len(pairs) + cols - 1) // cols
    grid = np.full((rows * cell, cols * cell, 3), 30, np.uint8)

    for i, (img_fp, json_fp) in enumerate(pairs):
        img = cv2.imread(str(img_fp))
        if img is None:
            continue
        H, W = img.shape[:2]
        k = parse_annotation(json_fp)                       # (2,2) 原图坐标
        s = cell / max(H, W)
        small = cv2.resize(img, (int(W * s), int(H * s)))
        canvas = np.full((cell, cell, 3), 0, np.uint8)
        canvas[: small.shape[0], : small.shape[1]] = small
        for x, y in k * s:
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 0, 255), -1)
        r, c = divmod(i, cols)
        grid[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = canvas
        # 同步保存单张放大图, 便于看不清时翻原图
        out_one = Path(args.out).parent / "single"
        out_one.mkdir(parents=True, exist_ok=True)
        vis = img.copy()
        for x, y in k:
            cv2.circle(vis, (int(x), int(y)), max(H, W) // 150, (0, 0, 255), -1)
        cv2.imwrite(str(out_one / f"{img_fp.stem}.jpg"), vis)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    print(f"grid -> {out} ; singles -> {out.parent / 'single'}")
    print("判读: 红点应落在指缝底部。飘到半空/边缘/图外的格子 = GT 可疑, "
          "记下文件名(lines in single/ 目录同名)。")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Phase A 裁 ROI: 绕过 palm_roi 的分类器, 强制走 WiLoR(完整手)分支.

复用 palm_roi 的 WilorDetector + ROIExtractor, 其余逻辑自己控制:
  <input>/<subject_hand>/*.jpg -> <output>/<subject_hand>/*_roi.png (112x112)

用法 (pytorch 环境):
  python scripts/crop_roi_wilor.py --input E:/data/phaseA/casia_input \
      --output E:/data/phaseA/casia_roi112 --roi_size 112 [--limit 2]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROI_NEW = Path(__file__).resolve().parents[1]
PALM_ROI = Path("E:/code/palm_roi")
sys.path.insert(0, str(ROI_NEW))
sys.path.insert(0, str(PALM_ROI))

from models.wilor_detector import WilorDetector          # noqa: E402
from core.roi_extractor import ROIExtractor              # noqa: E402
from config.settings import Config, get_config               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--roi_size", type=int, default=112)
    ap.add_argument("--limit", type=int, default=0, help="每类最多处理张数(测试用)")
    args = ap.parse_args()

    get_config()
    detector = WilorDetector(device="cuda" if sys.modules["torch"].cuda.is_available() else "cpu")
    extractor = ROIExtractor(alpha=0.15, beta=1.2)

    in_root, out_root = Path(args.input), Path(args.output)
    stats = {"ok": 0, "fail_kpt": 0, "fail_roi": 0, "total": 0}
    for class_dir in sorted(p for p in in_root.iterdir() if p.is_dir()):
        out_dir = out_root / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(f for f in class_dir.iterdir()
                       if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        if args.limit:
            files = files[: args.limit]
        for f in files:
            stats["total"] += 1
            img = cv2.imread(str(f))
            if img is None:
                stats["fail_kpt"] += 1
                continue
            kpt = detector.predict(img)
            if kpt is None or not getattr(kpt, "keypoints", None):
                stats["fail_kpt"] += 1
                continue
            roi = extractor.extract(img, kpt.keypoints)
            if not roi.success or roi.roi_image is None:
                stats["fail_roi"] += 1
                continue
            resized = cv2.resize(roi.roi_image, (args.roi_size, args.roi_size))
            cv2.imwrite(str(out_dir / f"{f.stem}_roi.png"), resized)
            stats["ok"] += 1
    print("stats:", stats)


if __name__ == "__main__":
    main()

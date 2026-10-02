# -*- coding: utf-8 -*-
"""定位线冒烟测试: 合成数据(图+json) -> 训练 3 epochs -> 评测, 全链路验证.

用法: python scripts/smoke_keypoint.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def make_synthetic(root: Path, n=40):
    """随机旋转/缩放画一只'合成手掌'(椭圆掌+4指), 按几何真值写 json."""
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = np.zeros((480, 640, 3), np.uint8)
        ang = rng.uniform(-180, 180)
        scale = rng.uniform(0.6, 1.1)
        cx, cy = rng.uniform(200, 440), rng.uniform(180, 320)
        # 以标准姿态定义: 掌心(0,0), 谷点(-70,-80)/(70,-80), 指向上
        pts_local = [[-70, -80], [70, -80]]  # 2 谷点 (与 MobileNet_Data 约定一致)
        th = np.deg2rad(ang)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) * scale

        def to_img(p):
            q = R @ np.array(p, float) + [cx, cy]
            return int(q[0]), int(q[1])

        palm = tuple(to_img((0, 20)))
        cv2.ellipse(img, palm, (int(90 * scale), int(110 * scale)), ang, 0, 360, (80, 80, 80), -1)
        for fx in (-60, -20, 20, 60):  # 4 根手指
            tip = to_img((fx, -170))
            base = to_img((fx, -70))
            cv2.line(img, base, tip, (90, 90, 90), int(34 * scale))
        ann = {"valleys": [list(to_img(v)) for v in pts_local]}
        cv2.imwrite(str(root / f"{i:05d}.jpg"), img)
        (root / f"{i:05d}.json").write_text(json.dumps(ann), encoding="utf-8")
    print(f"synthetic dataset: {n} pairs -> {root}")


def main():
    with tempfile.TemporaryDirectory() as td:
        data = Path(td) / "synthetic"
        make_synthetic(data)
        r1 = subprocess.run([PY, str(ROOT / "scripts" / "train_keypoint.py"),
                             "--data_root", str(data), "--out_dir", str(ROOT / "runs/smoke_kpt"),
                             "--epochs", "3", "--iters", "3", "--batch_size", "8",
                             "--num_workers", "0"], capture_output=True, text=True)
        print(r1.stdout[-1500:])
        if r1.returncode != 0:
            print(r1.stderr[-3000:])
            sys.exit("TRAIN FAILED")
        r2 = subprocess.run([PY, str(ROOT / "scripts" / "eval_keypoint.py"),
                             "--ckpt", str(ROOT / "runs/smoke_kpt/best.pt"),
                             "--data_root", str(data), "--vis", "4",
                             "--out_dir", str(ROOT / "runs/smoke_kpt/eval")],
                            capture_output=True, text=True)
        print(r2.stdout[-1200:])
        if r2.returncode != 0:
            print(r2.stderr[-3000:])
            sys.exit("EVAL FAILED")
        for fp in ["results.json", "results.md", "vis"]:
            assert (ROOT / "runs/smoke_kpt/eval" / fp).exists(), fp
        print("SMOKE TEST PASSED (train + eval + reports + vis)")


if __name__ == "__main__":
    main()

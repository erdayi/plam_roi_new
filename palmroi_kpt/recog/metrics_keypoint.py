# -*- coding: utf-8 -*-
"""定位指标: NLE(归一化定位误差) 与 SR(成功率).

NLE = ||pred - gt|| / d(valley1_gt, valley2_gt)   # 以两谷间距归一化, 尺度不变
SR@t = NLE < t 的样本比例 (t 默认 0.05 / 0.10 / 0.15)
输出: 逐点细分 + 总体, dict 形式 (JSON 可直接序列化).
"""
from __future__ import annotations

import numpy as np

DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15)
POINT_NAMES = ["valley1", "valley2", "center"]


def nle_metrics(pred_norm: np.ndarray, gt_norm: np.ndarray,
                thresholds=DEFAULT_THRESHOLDS) -> dict:
    """pred/gt: (N,3,2) 归一化坐标 (同顺序 valley1/valley2/center)."""
    pred = np.asarray(pred_norm, dtype=np.float64)
    gt = np.asarray(gt_norm, dtype=np.float64)
    d_v = np.linalg.norm(gt[:, 0] - gt[:, 1], axis=1)          # 两谷间距 (尺度基准)
    d_v = np.maximum(d_v, 1e-6)

    err = np.linalg.norm(pred - gt, axis=2) / d_v[:, None]      # (N,3)
    out = {"nle_mean": float(err.mean())}
    for i, name in enumerate(POINT_NAMES):
        out[f"nle_{name}"] = float(err[:, i].mean())
        for t in thresholds:
            out[f"sr{int(t*100)}_{name}"] = float((err[:, i] < t).mean())
    out["nle_all_points_max"] = float(err.max(axis=1).mean())   # 三点全对才算准的严格口径
    for t in thresholds:
        out[f"sr{int(t*100)}_all"] = float((err.max(axis=1) < t).mean())
    return out

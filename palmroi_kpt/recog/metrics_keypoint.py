# -*- coding: utf-8 -*-
"""定位指标: NLE(归一化定位误差) 与 SR(成功率), 支持无序关键点 (K=2 顺序不变匹配).

NLE = ||pred - gt|| / d(gt[0], gt[1])   # 以两谷间距归一化, 尺度不变
K=2 (无序谷点对): 对 identity/swap 两种配对取每样本误差较小者后再统计.
输出 dict 均为 JSON 可序列化.
"""
from __future__ import annotations

import numpy as np

DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15)


def nle_metrics(pred_norm: np.ndarray, gt_norm: np.ndarray,
                thresholds=DEFAULT_THRESHOLDS) -> dict:
    """pred/gt: (N,K,2) 归一化坐标. K=2 自动顺序不变匹配."""
    pred = np.asarray(pred_norm, dtype=np.float64)
    gt = np.asarray(gt_norm, dtype=np.float64)
    N, K, _ = gt.shape
    d_ref = np.maximum(np.linalg.norm(gt[:, 0] - gt[:, 1], axis=1), 1e-6)  # 谷间距为尺度基准

    err = np.linalg.norm(pred - gt, axis=2) / d_ref[:, None]               # (N,K)
    if K == 2:
        err_sw = err[:, ::-1]
        # 每样本选误差更小的配对 (以最大点误差为准)
        pick_swap = err_sw.max(axis=1) < err.max(axis=1)
        err = np.where(pick_swap[:, None], err_sw, err)

    out = {"nle_mean": float(err.mean())}
    for i in range(K):
        out[f"nle_p{i+1}"] = float(err[:, i].mean())
        for t in thresholds:
            out[f"sr{int(t*100)}_p{i+1}"] = float((err[:, i] < t).mean())
    out["nle_all_max"] = float(err.max(axis=1).mean())   # 全部点都准的严格口径
    for t in thresholds:
        out[f"sr{int(t*100)}_all"] = float((err.max(axis=1) < t).mean())
    return out


POINT_NAMES = {2: ["valley_a", "valley_b"], 3: ["valley1", "valley2", "center"]}

# -*- coding: utf-8 -*-
"""拓扑损失 v2: 三点三角形 (valley1/valley2/center), 旋转/镜像不变.

描述子 (对旋转、镜像、平移、等比缩放全部不变):
  - 成对距离比: d(V1,V2) : d(V1,C) : d(V2,C), 以总周长归一化 -> 3 维
  - 掌心处两谷向量夹角 cos(theta) -> 1 维 (cos 对旋转/镜像天然不变)
损失 = GT 与预测的形状描述子分布 (均值+方差) 对齐.
"""
from __future__ import annotations

import torch


def triangle_descriptor(kpts: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """(N,3,2) 归一化坐标 -> (N,4) 不变形状描述子."""
    v1, v2, c = kpts[:, 0], kpts[:, 1], kpts[:, 2]
    d12 = torch.norm(v1 - v2, dim=-1)
    d1c = torch.norm(v1 - c, dim=-1)
    d2c = torch.norm(v2 - c, dim=-1)
    per = d12 + d1c + d2c + eps
    ratio = torch.stack([d12 / per, d1c / per, d2c / per], dim=-1)
    cos_ang = torch.sum((v1 - c) * (v2 - c), dim=-1) / (d1c * d2c + eps)
    return torch.cat([ratio, cos_ang.unsqueeze(-1)], dim=-1)


class PalmTopologyLoss(torch.nn.Module):
    """三角形拓扑对齐: 预测形状分布向 GT 形状分布对齐 (均值 + 标准差)."""

    def __init__(self, lambda_mean: float = 1.0, lambda_std: float = 0.5):
        super().__init__()
        self.lambda_mean = lambda_mean
        self.lambda_std = lambda_std

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """pred/gt: (N,3,2) 归一化坐标."""
        if pred.shape[0] < 2:
            return (_triangle_mean(pred) - _triangle_mean(gt)).abs().sum()
        dp, dg = triangle_descriptor(pred), triangle_descriptor(gt)
        l_mean = (dp.mean(0) - dg.mean(0)).abs().sum()
        l_std = (dp.std(0) - dg.std(0)).abs().sum()
        return self.lambda_mean * l_mean + self.lambda_std * l_std


def _triangle_mean(kpts):
    return triangle_descriptor(kpts).mean(0)

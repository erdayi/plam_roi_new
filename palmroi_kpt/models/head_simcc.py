# -*- coding: utf-8 -*-
"""M2: SimCC 亚像素关键点头.

把关键点定位改造成 x/y 两个方向的分类问题 (SimCC, RTMPose 同款范式):
- 无热图解码、无 ARGMAX 量化误差, 分辨率可到亚像素 (bin < 1px);
- 与 PKLNet 的 global+local 回归头同接口可替换;
- 谷点/腕点作为直接监督目标 (不再经 21 点取中点).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SimCCKeypointHead(nn.Module):
    """SimCC 头: 输入 encoder 特征 (N,C,h,w), 输出每个关键点的 x-bin/y-bin logits.

    Args:
        in_channels: encoder 输出通道数.
        num_keypoints: 关键点数 (掌纹谷点方案 = 3 谷点 + 1 腕点 = 4).
        simcc_x_res / simcc_y_res: 输入图坐标划分的 bin 数, > 输入分辨率即亚像素.
        hidden_dims: MLP 中间层通道 (遵循原版 SimCC 的 MLP 设计).
    """

    def __init__(
        self,
        in_channels: int = 512,
        num_keypoints: int = 4,
        simcc_x_res: int = 256,
        simcc_y_res: int = 256,
        hidden_dims: tuple = (512, 128),
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.simcc_x_res = simcc_x_res
        self.simcc_y_res = simcc_y_res

        self.flatten = nn.Flatten(2)                      # (N,C,hw)
        self.transpose = lambda t: t.transpose(1, 2)      # (N,hw,C)
        layers, c = [], in_channels
        for h in hidden_dims:
            layers += [nn.Linear(c, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
            c = h
        self.mlp = nn.Sequential(*layers)
        self.head_x = nn.Linear(c, num_keypoints * simcc_x_res)
        self.head_y = nn.Linear(c, num_keypoints * simcc_y_res)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (simcc_x, simcc_y), 形状 (N, K, res)."""
        n, c, h, w = feat.shape
        feat = self.transpose(self.flatten(feat)).reshape(n * h * w, c)  # (N*hw, C)
        feat = self.mlp(feat)
        x = self.head_x(feat).view(n, h * w, self.num_keypoints, -1)
        y = self.head_y(feat).view(n, h * w, self.num_keypoints, -1)
        # 同一空间位置的两个 bin 特征取最大 (原版 SimCC 做法), (N,K,res)
        simcc_x = x.max(dim=1).values
        simcc_y = y.max(dim=1).values
        return simcc_x, simcc_y

    @torch.no_grad()
    def decode(self, simcc_x: torch.Tensor, simcc_y: torch.Tensor) -> torch.Tensor:
        """(N,K,res) logits -> (N,K,2) 亚像素坐标 (期望解码, 可微推理等价)."""
        px = _soft_argmax(simcc_x)   # (N,K) in [0, res_x)
        py = _soft_argmax(simcc_y)
        return torch.stack([px, py], dim=-1)

    def coordinate_scale(self, input_size: tuple[int, int]) -> tuple[float, float]:
        """bin 坐标 -> 原图坐标的缩放因子 (亚像素: res > input 即放大分辨率)."""
        return self.simcc_x_res / input_size[0], self.simcc_y_res / input_size[1]


def _soft_argmax(logits: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
    logp = torch.log_softmax(logits * beta, dim=-1)
    pos = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    return (logp.exp() * pos).sum(dim=-1)


def simcc_loss(
    simcc_x: torch.Tensor, simcc_y: torch.Tensor,
    gt_x: torch.Tensor, gt_y: torch.Tensor, sigma: float = 6.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """标准 SimCC 训练损失: GT 坐标高斯目标 vs 模型分布 (CE).

    gt_x/gt_y 为 bin 坐标 (float), (N,K). reduction: mean | none(逐样本, 供顺序不变匹配).
    sigma 默认 6 (RTMPose 系标配; 过小的 sigma 会让目标分布过尖、训练震荡).
    """
    loss = 0.0
    for simcc, gt, res in ((simcc_x, gt_x, simcc_x.shape[-1]), (simcc_y, gt_y, simcc_y.shape[-1])):
        idx = torch.arange(res, device=simcc.device).view(1, 1, -1)
        target = torch.exp(-((idx - gt.unsqueeze(-1)) ** 2) / (2 * sigma ** 2))
        target = target / (target.sum(dim=-1, keepdim=True) + 1e-8)
        logp = torch.log_softmax(simcc, dim=-1)
        per = -(target * logp).sum(dim=-1)                      # (N,K)
        loss = loss + per.mean(dim=-1)                          # (N,)
    loss = loss / 2
    return loss.mean() if reduction == "mean" else loss

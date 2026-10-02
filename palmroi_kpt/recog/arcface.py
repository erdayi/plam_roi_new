# -*- coding: utf-8 -*-
"""ArcFace 边际损失头(单卡轻量版,与 TopoFR/insightface 语义一致)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    """Additive Angular Margin: logits = s * cos(theta + m), 交叉熵.

    小类别数(掌纹库通常 <1000 ID)时建议 m=0.4, s=64 起步.
    """

    def __init__(self, in_features: int = 512, num_classes: int = 100,
                 s: float = 64.0, m: float = 0.4):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        w = F.normalize(self.weight, dim=1)
        x = F.normalize(emb, dim=1)
        cos = F.linear(x, w).clamp(-1 + 1e-7, 1 - 1e-7)          # (N,C)
        theta = torch.acos(cos)
        target_logit = torch.cos(theta + self.m)                # cos(theta+m), (N,C)
        onehot = F.one_hot(label, num_classes=cos.shape[1]).to(cos.dtype)
        logits = cos * self.s + (target_logit - cos) * onehot * self.s
        return F.cross_entropy(logits, label)

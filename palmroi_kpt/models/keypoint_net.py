# -*- coding: utf-8 -*-
"""关键点定位网络: backbone + (可选)分割先验 + SimCC 三点头.

每个组件的人脸 SOTA 出身见 docs/06_face_sota_transplant.md:
  backbone  <- 人脸预训练初始化 (M1, LVFace/TopoFR 权重可选)
  分支①分割 <- BiSeNet V2 式双路思想 (轻量版; 需 mask 标签, 默认关闭)
  头②SimCC  <- PIPNet/SimCC 坐标分类范式
"""
from __future__ import annotations

import torchvision
import torch
import torch.nn as nn

from palmroi_kpt.models.head_simcc import SimCCKeypointHead


class SegPriorHead(nn.Module):
    """轻量手掌分割先验头 (BiSeNet 精神: 低分辨率上下文 + 高分辨率细节融合).

    输出 input_size/4 的 mask logits; 分割特征与 backbone 特征拼接后喂 SimCC.
    需要每图一个同名 _mask.png 标签 (SAM2 离线生成), 无标签时整段关闭.
    """

    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1), nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.logits = nn.Conv2d(hidden, 1, 1)
        self.out_channels = hidden

    def forward(self, feat):
        f = self.conv(feat)
        return self.logits(f), f  # (mask logits, 分割特征)


class PalmKeypointNet(nn.Module):
    """backbone -> [seg prior] -> SimCC 头. 输出 (simcc_x, simcc_y), 各 (N,3,res)."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 num_keypoints: int = 3, input_size: int = 256,
                 simcc_res: int = 256, use_seg: bool = False):
        super().__init__()
        self.backbone_name = backbone
        self.use_seg = use_seg
        if backbone == "resnet18":
            m = torchvision.models.resnet18(
                weights="DEFAULT" if pretrained else None)
            self.feat = nn.Sequential(*list(m.children())[:-2])  # (N,512,h/32,w/32)
            c = 512
        elif backbone == "mobilenet_v3_small":
            m = torchvision.models.mobilenet_v3_small(
                weights="DEFAULT" if pretrained else None)
            self.feat = m.features
            c = 576
        else:
            raise ValueError(backbone)

        if use_seg:
            self.seg = SegPriorHead(c)
            head_in = c + self.seg.out_channels
        else:
            head_in = c

        self.head = SimCCKeypointHead(
            in_channels=head_in, num_keypoints=num_keypoints,
            simcc_x_res=simcc_res, simcc_y_res=simcc_res)

    def forward(self, x):
        f = self.feat(x)
        seg_logits = None
        if self.use_seg:
            seg_logits, f_seg = self.seg(f)
            f_seg_up = nn.functional.interpolate(f_seg, size=f.shape[2:], mode="bilinear",
                                                 align_corners=False)
            f = torch.cat([f, f_seg_up], dim=1)
        sx, sy = self.head(f)
        return sx, sy, seg_logits

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


class _IRFeatureExtractor(nn.Module):
    """iResNet 的空间特征提取: 取 conv1..layer4 的特征图 (bn2/fc 之前的 (N,512,h/16,w/16)),
    供 SimCC 头使用 (识别线的 embedding 输出对坐标回归无效)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        m = self.m
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.prelu(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        return x


class PalmKeypointNet(nn.Module):
    """backbone -> [seg prior] -> SimCC 头. 输出 (simcc_x, simcc_y), 各 (N,3,res)."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 num_keypoints: int = 3, input_size: int = 256,
                 simcc_res: int = 256, use_seg: bool = False):
        super().__init__()
        self.backbone_name = backbone
        self.use_seg = use_seg
        if backbone == "resnet18":
            m = self._build_resnet18(pretrained)
            self.feat = nn.Sequential(*list(m.children())[:-2])  # (N,512,h/32,w/32)
            c = 512
        elif backbone == "mobilenet_v3_small":
            m = self._build_mobilenet(pretrained)
            self.feat = m.features
            c = 576
        elif backbone == "iresnet18":
            # 人脸/掌纹识别预训练的 iResNet (insightface 语义, MIT, 已内嵌 vendored)。
            # 预训练权重经 --init_weights 注入 (如 Phase A 识别 checkpoint)。
            from palmroi_kpt.models.iresnet_backbone import iresnet18
            m = iresnet18(num_features=512, dropout=0.4, num_classes=1000)
            self.feat = _IRFeatureExtractor(m)
            c = 512
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

    @staticmethod
    def _safe_build(builder, pretrained: bool, what: str):
        """预训练权重下载: 走实验室代理; SSL 失败自动重试跳过证书校验; 仍失败则降级随机初始化."""
        from palmroi_kpt.netenv import apply_default_proxy
        apply_default_proxy()
        if not pretrained:
            return builder(weights=None)
        try:
            return builder(weights="DEFAULT")
        except Exception as e:
            print(f"[WARNING] {what} 预训练权重下载失败({type(e).__name__}), "
                  f"重试: 跳过 SSL 证书校验...")
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            try:
                return builder(weights="DEFAULT")
            except Exception as e2:
                print(f"[WARNING] 重试仍失败({type(e2).__name__}), 降级为随机初始化。")
                print("  手动解决: 把对应 .pth 放入 %USERPROFILE%\\.cache\\torch\\hub\\checkpoints\\")
                print("  resnet18 -> resnet18-f37072fd.pth")
                print("  mobilenet_v3_small -> mobilenet_v3_small-047dcff4.pth")
                return builder(weights=None)

    @classmethod
    def _build_resnet18(cls, pretrained: bool):
        return cls._safe_build(torchvision.models.resnet18, pretrained, "resnet18")

    @classmethod
    def _build_mobilenet(cls, pretrained: bool):
        return cls._safe_build(torchvision.models.mobilenet_v3_small, pretrained,
                               "mobilenet_v3_small")

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

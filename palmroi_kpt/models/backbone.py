# -*- coding: utf-8 -*-
"""M1: encoder + 面部/手部预训练权重加载.

迁移链: face-pretrained (InsightFace/AdaFace 骨干) -> hand (FoundHand/COCO-WholeBody)
-> palm fine-tune. 本模块只负责"权重能不能装进 encoder"这一层, 与具体数据集解耦.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision

logger = logging.getLogger(__name__)

#: 人脸识别预训练权重的典型参数名前缀 -> torchvision ResNet18 参数名映射示例.
#: InsightFace 的 backbone 命名与 torchvision 不同, 接入时在此维护映射表;
#: 名称对不上的层保持随机初始化并记录 (partial load).
_FACE_NAME_MAP = {
    # e.g. "conv1.weight" -> "conv1.weight", "layer1.0.conv1.weight" -> ...
}


def build_encoder(name: str = "resnet18", pretrained_imagenet: bool = True) -> nn.Module:
    """构建 encoder, 返回去掉分类头的卷积主干."""
    fn = {
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34,
        "mobilenet_v3_small": torchvision.models.mobilenet_v3_small,
    }[name]
    model = fn(weights="DEFAULT" if pretrained_imagenet else None)
    if name.startswith("resnet"):
        model = nn.Sequential(*list(model.children())[:-2])
    else:  # mobilenet
        model = model.features
    return model


def load_face_pretrained(encoder: nn.Module, weight_path: Optional[str]) -> tuple[nn.Module, int]:
    """尝试把人脸识别预训练权重装入 encoder (M1 第一段).

    返回 (encoder, 成功加载的参数量). 权重不存在或命名不匹配时安全降级为
    ImageNet 初始化, 并打印日志 —— 训练永远可以继续跑, 这是 partial load 的意义.
    """
    if not weight_path or not Path(weight_path).exists():
        logger.warning("face-pretrained weights not found at %r, keep ImageNet init", weight_path)
        return encoder, 0

    state = torch.load(weight_path, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    loaded = 0
    own = encoder.state_dict()
    for k, v in state.items():
        tgt = _FACE_NAME_MAP.get(k, k)
        if tgt in own and own[tgt].shape == v.shape:
            own[tgt] = v
            loaded += 1
    encoder.load_state_dict(own)
    logger.info("face-pretrained: loaded %d/%d tensors from %s", loaded, len(own), weight_path)
    return encoder, loaded

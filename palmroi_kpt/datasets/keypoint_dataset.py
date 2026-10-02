# -*- coding: utf-8 -*-
"""掌纹 3 点关键点数据集 (valley1/valley2/center 同名 json).

约定 (与用户提供的数据一致):
  - 图像任意常见格式, 同 stem 的 .json 标注:
      {"valley1": [x, y], "valley2": [x, y], "center": [x, y]}
  - 不区分左右手; 姿态任意 (横/竖/旋转) -> 训练增广含 ±180° 旋转, 标注同步变换;
  - 禁止水平翻转: 翻转会交换 valley1/valley2 的几何位置 (通道语义错位).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

POINT_NAMES = ["valley1", "valley2", "center"]


def rotate_point(pt, m):
    """关键点跟随图像仿射变换 (m: 2x3)."""
    x, y = pt
    return [m[0, 0] * x + m[0, 1] * y + m[0, 2],
            m[1, 0] * x + m[1, 1] * y + m[1, 2]]


class PalmKeypointDataset(Dataset):
    """读取 <root>/**/*.{jpg,png} + 同名 .json; 输出 256x256 图 + bin 坐标关键点.

    Args:
        root: 数据根目录 (递归扫描).
        input_size: 网络输入边长 (定位线默认 256).
        simcc_res: SimCC bin 分辨率 (>= input_size 即亚像素能力).
        train: True 时启用旋转/缩放增广.
        max_side: 原图过长边先缩到该值再处理 (加速).
    """

    def __init__(self, root: str, input_size: int = 256, simcc_res: int = 256,
                 train: bool = True, max_side: int = 1024):
        self.input_size, self.simcc_res, self.train = input_size, simcc_res, train
        self.max_side = max_side
        self.samples = []
        root = Path(root)
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                j = p.with_suffix(".json")
                if j.exists():
                    self.samples.append((p, j))
        if not self.samples:
            raise RuntimeError(f"no (image, json) pairs under {root}")

    def __len__(self):
        return len(self.samples)

    def _load(self, idx):
        img_fp, json_fp = self.samples[idx]
        img = cv2.imread(str(img_fp))
        ann = json.loads(Path(json_fp).read_text(encoding="utf-8"))
        kpts = np.array([ann[k] for k in POINT_NAMES], dtype=np.float32)  # (3,2)
        # 长边限制
        h, w = img.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            kpts *= scale
        return img, kpts

    def __getitem__(self, idx):
        img, kpts = self._load(idx)
        h, w = img.shape[:2]
        size = self.input_size

        if self.train:
            for _ in range(4):  # 变换后关键点出画则重采样
                ang = random.uniform(-180, 180)
                sc = random.uniform(0.75, 1.25)
                m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
                m[0, 2] += (size - w) / 2
                m[1, 2] += (size - h) / 2
                k = np.array([rotate_point(p, m) for p in kpts], dtype=np.float32)
                if (k >= 8).all() and (k <= size - 8).all():
                    break
            else:
                m = cv2.getRotationMatrix2D((w / 2, h / 2), 0, size / max(h, w))
                m[0, 2] += (size - w) / 2
                m[1, 2] += (size - h) / 2
                k = np.array([rotate_point(p, m) for p in kpts], dtype=np.float32)
            img = cv2.warpAffine(img, m, (size, size), borderValue=0)
        else:
            m = cv2.getRotationMatrix2D((w / 2, h / 2), 0, size / max(h, w))
            m[0, 2] += (size - w) / 2
            m[1, 2] += (size - h) / 2
            k = np.array([rotate_point(p, m) for p in kpts], dtype=np.float32)
            img = cv2.warpAffine(img, m, (size, size), borderValue=0)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
        k = np.clip(k, 0, size - 1)
        k_bin = torch.from_numpy(k * (self.simcc_res / size)).float()  # bin 坐标
        k_norm = torch.from_numpy(k / size).float()                    # 归一化坐标 (评测用)
        return img, k_bin, k_norm

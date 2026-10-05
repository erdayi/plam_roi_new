# -*- coding: utf-8 -*-
"""掌纹谷点关键点数据集 (MobileNet_Data 约定: {"valleys": [[x,y],[x,y]]}).

约定:
  - 图像任意常见格式, 同 stem .json 标注, 支持两种格式(自动识别):
      A) {"valleys": [[x,y],[x,y]]}                      # 2 谷点, 无序对 (主格式)
      B) {"valley1":[x,y],"valley2":[x,y],"center":[x,y]}  # 3 点 (兼容旧样本)
  - 不区分左右手 -> 训练用顺序不变匹配 (identity/swap 取损失较小者, 见 train 脚本);
  - 姿态任意 -> 训练增广含 ±180° 旋转, 标注同步变换;
  - 禁止水平翻转 (会交换两谷点语义).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # 掌纹采集存在 1 亿像素级原图, 关闭 PIL 防炸弹上限

EXCLUDE_DIRS = {"visualization"}


def parse_annotation(json_path: Path):
    """解析标注 -> (2,2) float 数组, 统一输出两谷点.

    两种已知格式均兼容 (混合数据集直接可训):
      A) {"valleys": [[x,y],[x,y]]}                        # 2 谷点
      B) {"valley1":..,"valley2":..,"center":..}           # 3 点 -> 取前两谷, center 忽略
    """
    ann = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if "valleys" in ann:
        k = np.array(ann["valleys"], dtype=np.float32)
    elif "valley1" in ann:
        k = np.array([ann["valley1"], ann["valley2"]], dtype=np.float32)
    else:
        raise KeyError(f"unknown annotation keys {list(ann.keys())} in {json_path}")
    if k.shape != (2, 2):
        raise ValueError(f"expect 2 valleys, got {k.shape} in {json_path}")
    return k


def walk_pairs(root: str):
    """递归收集 (图像, json) 对, 跳过 visualization 等排除目录."""
    root_p = Path(root)
    out = []
    for p in sorted(root_p.rglob("*")):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} and p.with_suffix(".json").exists():
            out.append((p, p.with_suffix(".json")))
    return out


class PalmKeypointDataset(Dataset):
    """输出 (img, k_bin(K,2), k_norm(K,2)); K 由数据自动确定并强校验一致性."""

    def __init__(self, root: str, input_size: int = 256, simcc_res: int = 256,
                 train: bool = True, max_side: int = 1024,
                 cache_side: int = 512, cache_dir: str = None):
        self.input_size, self.simcc_res, self.train = input_size, simcc_res, train
        self.max_side = max_side
        self.samples = walk_pairs(root)
        if not self.samples:
            raise RuntimeError(f"no (image, json) pairs under {root}")
        # ---- 预缩放缓存 (业界 decode-once-reuse 标准做法) ----
        # 首次启动把全数据集解码并统一缩到 cache_side 存 .npy 到本地;
        # 之后所有 epoch 直接 np.load, 免去重复 JPEG 解码 (大图场景提速 3-5 倍).
        self.cache_side = cache_side
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.kpts = []  # 缓存解析结果 (原图像素坐标)
        self.img_wh = []  # 原图 (w, h), 只读文件头, 供缓存坐标换算
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        k_dims = set()
        for img_fp, json_fp in self.samples:
            k = parse_annotation(json_fp)
            k_dims.add(len(k))
            self.kpts.append(k)
            try:
                with Image.open(img_fp) as im:
                    self.img_wh.append(im.size)  # PIL: (w, h)
            except Exception:
                img = cv2.imread(str(img_fp))
                self.img_wh.append((img.shape[1], img.shape[0]))
        if len(k_dims) != 1:
            raise RuntimeError(f"mixed annotation dims {k_dims} in {root}; "
                               f"数据集内 K 必须一致")
        self.num_keypoints = k_dims.pop()
        # 未标注坐标系按原图分辨率 -> 载入时才知道; 归一化在 __getitem__ 内完成

    def __len__(self):
        return len(self.samples)

    def _imread_scaled(self, img_fp: Path):
        """读图; 超大图 (>2x max_side) 用降分辨率解码, 防止单图打爆内存."""
        from PIL import Image
        try:
            with Image.open(img_fp) as im:  # 只读文件头, 获取尺寸
                w0, h0 = im.size
        except Exception:
            img = cv2.imread(str(img_fp))
            return img, (img.shape[0], img.shape[1]) if img is not None else (0, 0)
        if max(h0, w0) <= self.max_side * 2:
            img = cv2.imread(str(img_fp))
            return (img, (h0, w0)) if img is not None else (None, (h0, w0))
        # 超大图: 选择合适的降采样档位 (1/8 -> 1/4 -> 1/2), 解码内存降低 64/16/4 倍
        for flag, div in ((cv2.IMREAD_REDUCED_COLOR_8, 8),
                          (cv2.IMREAD_REDUCED_COLOR_4, 4),
                          (cv2.IMREAD_REDUCED_COLOR_2, 2)):
            if max(h0, w0) / div <= self.max_side * 2:
                img = cv2.imread(str(img_fp), flag)
                if img is not None:
                    return img, (h0, w0)
        img = cv2.imread(str(img_fp))
        return (img, (h0, w0)) if img is not None else (None, (h0, w0))

    def _cache_path(self, idx):
        stem = Path(self.samples[idx][0]).stem
        return self.cache_dir / f"{stem}_{self.cache_side}.npy"

    def _load(self, idx):
        img_fp, _ = self.samples[idx]
        kpts_orig = self.kpts[idx]        # 原图像素坐标 (只读)
        if self.cache_dir:
            cp = self._cache_path(idx)
            if cp.exists():               # 缓存命中: 免解码, 坐标按 原图->缓存图 比例换算
                img = np.load(cp)
                sx = img.shape[1] / self.img_wh[idx][0]
                sy = img.shape[0] / self.img_wh[idx][1]
                return img, kpts_orig * np.array([sx, sy], dtype=np.float32), img.shape[:2]
            # 缓存未命中: 解码 -> 坐标转缓存图坐标系 -> 存缓存
            img0, (h0, w0) = self._imread_scaled(img_fp)
            if img0 is None:
                raise RuntimeError(f"unreadable image: {img_fp}")
            h, w = img0.shape[:2]
            k0 = kpts_orig.copy()
            k0[:, 0] *= w / w0
            k0[:, 1] *= h / h0
            s = min(1.0, self.cache_side / max(h, w))
            if s < 1.0:
                img0 = cv2.resize(img0, (int(w * s), int(h * s)))
                k0 *= s
            np.save(cp, img0)
            return img0, k0, img0.shape[:2]
        # 无缓存模式 (原逻辑)
        kpts = kpts_orig.copy()
        img, (h0, w0) = self._imread_scaled(img_fp)
        if img is None:
            raise RuntimeError(f"unreadable image: {img_fp}")
        h, w = img.shape[:2]
        kpts[:, 0] *= w / w0
        kpts[:, 1] *= h / h0
        scale = min(1.0, self.max_side / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            kpts *= scale
            h, w = img.shape[:2]
        return img, kpts, (h, w)

    def __getitem__(self, idx):
        img, kpts, (h, w) = self._load(idx)
        size = self.input_size

        if self.train:
            for _ in range(4):  # 变换后关键点出画则重采样
                ang = random.uniform(-180, 180)
                sc = random.uniform(0.75, 1.25)
                m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
                m[0, 2] += (size - w) / 2
                m[1, 2] += (size - h) / 2
                k = kpts @ m[:, :2].T + m[:, 2]
                if (k >= 8).all() and (k <= size - 8).all():
                    break
            else:
                m = cv2.getRotationMatrix2D((w / 2, h / 2), 0, size / max(h, w))
                m[0, 2] += (size - w) / 2
                m[1, 2] += (size - h) / 2
                k = kpts @ m[:, :2].T + m[:, 2]
            img = cv2.warpAffine(img, m, (size, size), borderValue=0)
        else:
            m = cv2.getRotationMatrix2D((w / 2, h / 2), 0, size / max(h, w))
            m[0, 2] += (size - w) / 2
            m[1, 2] += (size - h) / 2
            k = kpts @ m[:, :2].T + m[:, 2]
            img = cv2.warpAffine(img, m, (size, size), borderValue=0)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
        k = np.clip(k, 0, size - 1)
        k_bin = torch.from_numpy(k * (self.simcc_res / size)).float()
        k_norm = torch.from_numpy(k / size).float()
        return img, k_bin, k_norm

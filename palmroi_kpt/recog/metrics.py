# -*- coding: utf-8 -*-
"""验证协议指标: EER / TAR@FAR(掌纹跨库 pair 协议通用)."""
from __future__ import annotations

import numpy as np
import torch


def eer_and_tar(same_scores: np.ndarray, diff_scores: np.ndarray,
                far_targets=(1e-2, 1e-3, 1e-4)) -> dict:
    """same/diff 余弦分数 -> EER 与 TAR@FAR.

    小验证集上 TAR@FAR=1e-4 可能无意义(负对太少), 返回 nan 由调用方忽略.
    """
    same = np.asarray(same_scores, dtype=np.float64)
    diff = np.asarray(diff_scores, dtype=np.float64)
    scores = np.concatenate([same, diff])
    labels = np.concatenate([np.ones_like(same), np.zeros_like(diff)])
    order = np.argsort(-scores)
    labels = labels[order]

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    tp = np.cumsum(labels)                     # 阈值取该分数时接受的正样本
    fp = np.cumsum(1 - labels)
    far = fp / max(n_neg, 1)
    frr = (n_pos - tp) / max(n_pos, 1)

    i = int(np.argmin(np.abs(far - frr)))
    eer = float((far[i] + frr[i]) / 2)
    out = {"eer": eer}
    for t in far_targets:
        idx = np.where(far <= t)[0]
        out[f"tar@far={t:g}"] = float(frr[idx[0]]) if len(idx) else float("nan")
        # TAR = 1 - FRR
        out[f"tar@far={t:g}"] = 1 - out[f"tar@far={t:g}"] if len(idx) else float("nan")
    return out


@torch.no_grad()
def evaluate_verification(encoder, dataset, device, max_ids: int = 0) -> dict:
    """从 ImageFolder 式数据集抽 pair 算 EER: 每类取样本两两同对, 跨类随机异对."""
    import random
    import torch

    encoder.eval()
    by_class: dict[int, list] = {}
    for i in range(len(dataset)):
        img, label = dataset[i]
        by_class.setdefault(label, []).append((i, img))
    if max_ids:
        by_class = dict(random.sample(list(by_class.items()), min(max_ids, len(by_class))))

    embs, labels = [], []
    for label, items in by_class.items():
        batch = torch.stack([img for _, img in items]).to(device)
        try:
            e = encoder(batch, phase="infer").cpu().numpy()   # TopoFR iResNet
        except TypeError:
            e = encoder(batch).cpu().numpy()                  # 普通 forward (ViT 等)
        embs.append(e)
        labels.extend([label] * len(items))
    embs = np.concatenate(embs)
    labels = np.asarray(labels)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)

    same, diff = [], []
    for a in range(len(embs)):
        for b in range(a + 1, len(embs)):
            s = float(embs[a] @ embs[b])
            (same if labels[a] == labels[b] else diff).append(s)
    if not same or not diff:
        return {"eer": float("nan")}
    return eer_and_tar(np.array(same), np.array(diff))

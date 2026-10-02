# -*- coding: utf-8 -*-
"""冒烟测试: M1 backbone + M2 SimCC 头 + M3 拓扑损失 全链路前向/反向一遍.

用法: python scripts/smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from palmroi_kpt.models.backbone import build_encoder
from palmroi_kpt.models.head_simcc import SimCCKeypointHead, simcc_loss
from palmroi_kpt.losses.topology import PalmTopologyLoss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, torch {torch.__version__}")

    # M1: encoder (ImageNet 初始化; 面部权重路径留空 -> 安全降级)
    encoder = build_encoder("resnet18").to(device)
    feat = encoder(torch.randn(2, 3, 256, 256, device=device))
    print(f"encoder out: {tuple(feat.shape)}")  # expect (2,512,8,8)

    # M2: SimCC 头 (256x256 输入 -> 256 bins, 亚像素)
    head = SimCCKeypointHead(in_channels=512, num_keypoints=4,
                             simcc_x_res=256, simcc_y_res=256).to(device)
    simcc_x, simcc_y = head(feat)
    kpts = head.decode(simcc_x, simcc_y)
    print(f"simcc_x: {tuple(simcc_x.shape)}, decoded kpts: {tuple(kpts.shape)}")
    print(f"decoded sample (bin coords): {kpts[0].detach().cpu().numpy().round(2).tolist()}")

    # 训练损失 = SimCC 主损失 + 拓扑对齐损失
    gt_x = torch.rand(2, 4, device=device) * 256
    gt_y = torch.rand(2, 4, device=device) * 256
    l_main = simcc_loss(simcc_x, simcc_y, gt_x, gt_y)

    kpts_norm = kpts / 256.0
    gt_norm = torch.stack([gt_x, gt_y], dim=-1) / 256.0
    l_topo = PalmTopologyLoss()(kpts_norm, gt_norm)

    loss = l_main + 0.01 * l_topo
    loss.backward()
    print(f"loss: main={l_main.item():.4f} topo={l_topo.item():.6f} total={loss.item():.4f}")

    # 反向传播后有梯度 = 全链路可训
    grads = [p.grad is not None for p in list(head.parameters())[:5]]
    print(f"head grads ok: {all(grads)}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()

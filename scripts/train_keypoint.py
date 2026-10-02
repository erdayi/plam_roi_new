# -*- coding: utf-8 -*-
"""定位线训练器: SimCC 三点谷点网络 (M2 主实验).

全配置化: 数据路径 / backbone / 分辨率 / 损失权重 全部命令行参数, 跨机器只需改参数.

用法:
  python scripts/train_keypoint.py --data_root E:/data/20k --out_dir runs/m2_v1
  python scripts/train_keypoint.py --data_root ... --backbone mobilenet_v3_small
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from palmroi_kpt.datasets.keypoint_dataset import PalmKeypointDataset
from palmroi_kpt.models.keypoint_net import PalmKeypointNet
from palmroi_kpt.models.head_simcc import simcc_loss
from palmroi_kpt.losses.topology import PalmTopologyLoss
from palmroi_kpt.recog.metrics_keypoint import nle_metrics


def split_train_val(root: str, val_fraction=0.1, seed=0):
    """按图像 stem 哈希切分, 跨次运行稳定 (同图永远同侧)."""
    ds = PalmKeypointDataset(root, train=False)
    tr_idx, va_idx = [], []
    for i, (img_fp, _) in enumerate(ds.samples):
        h = hash(Path(img_fp).stem) % 1000
        (va_idx if h < val_fraction * 1000 else tr_idx).append(i)
    return ds, tr_idx, va_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=r"D:\dataset\roi\MobileNet_Data",
                    help="图像+同名json 的数据根目录 (默认已写死实验室路径)")
    ap.add_argument("--val_root", default=None, help="独立验证集; 不给则从 data_root 按 10%% 切")
    ap.add_argument("--out_dir", default="runs/m2")
    ap.add_argument("--backbone", default="resnet18",
                    choices=["resnet18", "mobilenet_v3_small"])
    ap.add_argument("--input_size", type=int, default=256)
    ap.add_argument("--simcc_res", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--lambda_topo", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--iters", type=int, default=0, help=">0 时每 epoch 截断 (冒烟)")
    ap.add_argument("--save_every", type=int, default=5,
                    help="每 N 个 epoch 存 last.pt (覆盖写, 磁盘占用恒定; 1=最细粒度)")
    ap.add_argument("--no_resume", action="store_true", help="忽略 last.pt 从零重训")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    random.seed(0)
    print(f"device: {device}, torch {torch.__version__}")

    base_train = PalmKeypointDataset(args.data_root, args.input_size, args.simcc_res, train=True)
    K = base_train.num_keypoints
    print(f"num_keypoints = {K}")
    if args.val_root:
        train_ds = base_train
        val_ds = PalmKeypointDataset(args.val_root, args.input_size, args.simcc_res, train=False)
    else:
        _, tr_idx, va_idx = split_train_val(args.data_root)
        print(f"split: train {len(tr_idx)} / val {len(va_idx)}")
        train_ds = torch.utils.data.Subset(base_train, tr_idx)
        val_ds = torch.utils.data.Subset(
            PalmKeypointDataset(args.data_root, args.input_size, args.simcc_res, train=False), va_idx)
    print(f"train {len(train_ds)} / val {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    model = PalmKeypointNet(args.backbone, pretrained=True, num_keypoints=K,
                            input_size=args.input_size, simcc_res=args.simcc_res).to(device)
    topo = PalmTopologyLoss() if K == 3 else None  # 三角形拓扑需要 3 点

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    csv_fp = out / "train_log.csv"
    if not csv_fp.exists():
        with open(csv_fp, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["epoch", "loss", "topo", "nle_mean", "nle_p1", "nle_p2",
                 "sr5_all", "sr10_all", "sr15_all"])

    # ---- 断点续训: last.pt 每 epoch 自动保存, 重跑同命令自动恢复 ----
    start_epoch, best_sr = 0, -1.0
    last_fp = out / "last.pt"
    if last_fp.exists() and not args.no_resume:
        ck = torch.load(last_fp, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        best_sr = ck["best_sr"]
        print(f"[resume] 从 epoch {ck['epoch']} 恢复 (best SR@10 {best_sr:.4f}), "
              f"继续到 {args.epochs}")
        if args.epochs <= ck["epoch"]:
            print("[resume] 已达目标 epochs, 无需继续。加 --epochs 更大值或 --no_resume 重训。")
            return

    import time
    for ep in range(start_epoch, args.epochs):
        model.train()
        tot_loss, tot_topo, n = 0.0, 0.0, 0
        t0 = time.time()
        n_total = len(train_dl)
        for it, (img, k_bin, k_norm) in enumerate(train_dl):
            if args.iters and it >= args.iters:
                break
            img = img.to(device, non_blocking=True)
            k_bin = k_bin.to(device)
            k_norm = k_norm.to(device)
            sx, sy, _ = model(img)
            gt_x, gt_y = k_bin[..., 0], k_bin[..., 1]
            l_id = simcc_loss(sx, sy, gt_x, gt_y, sigma=args.sigma, reduction="none")
            if K == 2:  # 无序谷点对: identity/swap 逐样本取较小者
                l_sw = simcc_loss(sx, sy, gt_y, gt_x, sigma=args.sigma, reduction="none")
                l_main = torch.minimum(l_id, l_sw).mean()
            else:
                l_main = l_id.mean()
            pred_norm = model.head.decode(sx, sy) / args.simcc_res
            if topo is not None:
                l_t = topo(pred_norm, k_norm)
                loss = l_main + args.lambda_topo * l_t
            else:
                l_t = torch.zeros((), device=device)
                loss = l_main
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss, tot_topo, n = tot_loss + l_main.item(), tot_topo + l_t.item(), n + 1
            if (it + 1) % 50 == 0 or it + 1 == n_total:  # 逐迭代进度
                dt = time.time() - t0
                print(f"  [ep {ep:03d}] it {it+1}/{n_total} "
                      f"loss {tot_loss/n:.3f} {n/dt:.1f} img/s", flush=True)
        sched.step()

        # 验证
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for img, k_bin, k_norm in val_dl:
                sx, sy, _ = model(img.to(device, non_blocking=True))
                p = model.head.decode(sx, sy) / args.simcc_res
                preds.append(p.cpu().numpy())
                gts.append(k_norm.numpy())
        m = nle_metrics(np.concatenate(preds), np.concatenate(gts))
        with open(csv_fp, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, f"{tot_loss/max(n,1):.4f}", f"{tot_topo/max(n,1):.4f}"]
                                   + [f"{m[k]:.4f}" for k in
                                      ["nle_mean", "nle_p1", "nle_p2", "sr5_all", "sr10_all", "sr15_all"]])
        print(f"ep {ep:03d} loss {tot_loss/max(n,1):.4f} topo {tot_topo/max(n,1):.4f} "
              f"NLE {m['nle_mean']:.4f} SR@10 {m['sr10_all']:.4f}")
        if m["sr10_all"] > best_sr:
            best_sr = m["sr10_all"]
            torch.save({"model": model.state_dict(), "args": vars(args), "num_keypoints": K,
                        "epoch": ep}, out / "best.pt")
        # 每 epoch 快照 (断点续训用; 含优化器/调度器状态)
        if (ep + 1) % args.save_every == 0 or ep == args.epochs - 1:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": ep, "best_sr": best_sr,
                        "args": vars(args), "num_keypoints": K}, last_fp)
    print(f"best SR@10 {best_sr:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()

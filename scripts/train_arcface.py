# -*- coding: utf-8 -*-
"""Phase A 单卡 ArcFace 训练器(掌纹 ROI 验证基线).

- backbone 直接复用 TopoFR 的 iResNet(insightface 语义, 512 维嵌入);
- 输入 128x128 ROI ImageFolder; 严禁水平翻转(左右手是不同 ID);
- 小 ID 数场景: m=0.4, s=64 起步;
- 每轮做验证集 pair 评测 (EER / TAR@FAR).

用法:
  # 真实数据 (ImageFolder: data_root/<subject_hand>/*.jpg)
  python scripts/train_arcface.py --data_root E:/data/palm_rois --out_dir runs/phaseA
  # 冒烟 (内存合成数据, 验证训练环路)
  python scripts/train_arcface.py --synthetic
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "TopoFR"))

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from backbones.iresnet import iresnet18
from palmroi_kpt.recog.arcface import ArcFace
from palmroi_kpt.recog.metrics import evaluate_verification

# 掌纹 ROI 的增广: 禁止 hflip! 轻量色彩/几何扰动即可
TRAIN_TF = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
    transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
VAL_TF = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


class SyntheticPalm(Dataset):
    """冒烟用: 随机纹理+类相关偏置的假掌纹, 验证训练环路."""

    def __init__(self, n_classes=20, per_class=10, size=112, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.size, self.items = size, []
        self.templates = torch.randn(n_classes, 3, size, size, generator=g)
        for c in range(n_classes):
            for _ in range(per_class):
                self.items.append((c, torch.randn(3, size, size, generator=g) * 0.2 + self.templates[c]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        c, img = self.items[i]
        return img, c


def stratified_split(dataset, val_fraction=0.2, min_train=1, seed=0):
    """按类分层切 train/val 的索引 (ImageFolder.targets)."""
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for i, t in enumerate(dataset.targets):
        by_class.setdefault(t, []).append(i)
    tr, va = [], []
    for c, idxs in by_class.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_val = int(len(idxs) * val_fraction) if len(idxs) >= min_train + 1 else 0
        va += idxs[:n_val]
        tr += idxs[n_val:]
    return tr, va


class SubsetDataset(Dataset):
    def __init__(self, base, indices, transform):
        self.base, self.indices, self.transform = base, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, label = self.base.samples[self.indices[i]]
        from PIL import Image
        img = self.transform(Image.open(path).convert("RGB"))
        return img, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root")
    ap.add_argument("--out_dir", default="runs/phaseA")
    ap.add_argument("--synthetic", action="store_true", help="内存合成数据冒烟")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.4)
    ap.add_argument("--scale", type=float, default=64.0)
    ap.add_argument("--backbone", default="iresnet18", choices=["iresnet18", "lvface_s"],
                    help="iresnet18=TopoFR iResNet; lvface_s=LVFace ViT-S(ICCV 2025 SOTA)")
    ap.add_argument("--init_weights", default=None,
                    help="M1 人脸预训练权重: TopoFR iResNet .pth / LVFace .pt")
    ap.add_argument("--iters", type=int, default=0, help=">0 时每 epoch 截断迭代数(冒烟)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, torch {torch.__version__}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        train_ds = SyntheticPalm()
        val_ds = SyntheticPalm(seed=1)
    else:
        base = ImageFolder(args.data_root)
        tr_idx, va_idx = stratified_split(base)
        print(f"classes: {len(base.classes)}, train: {len(tr_idx)}, val: {len(va_idx)}")
        train_ds = SubsetDataset(base, tr_idx, TRAIN_TF)
        val_ds = SubsetDataset(base, va_idx, VAL_TF)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, drop_last=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_classes = len(set(t for _, t in train_ds))
    if args.backbone == "iresnet18":
        encoder = iresnet18(num_features=512, dropout=0.4, num_classes=num_classes).to(device)
        if args.init_weights:
            sd = torch.load(args.init_weights, map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd)
            own = encoder.state_dict()
            hit = sum(1 for k in sd if k in own and own[k].shape == sd[k].shape)
            for k in sd:
                if k in own and own[k].shape == sd[k].shape:
                    own[k] = sd[k]
            encoder.load_state_dict(own)
            print(f"face-pretrained init: loaded {hit} tensors")
    else:  # lvface_s
        import importlib.util
        lv_dir = Path(__file__).resolve().parents[1] / "third_party" / "LVFace" / "backbones"
        spec = importlib.util.spec_from_file_location(
            "lvface_backbones", lv_dir / "__init__.py",
            submodule_search_locations=[str(lv_dir)])
        lv_pkg = importlib.util.module_from_spec(spec)
        sys.modules["lvface_backbones"] = lv_pkg
        spec.loader.exec_module(lv_pkg)
        encoder = lv_pkg.get_model("vit_s", num_features=512).to(device)
        if args.init_weights:
            sd = torch.load(args.init_weights, map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd)
            encoder.load_state_dict(sd, strict=True)
            print(f"LVFace-S weights loaded (strict)")
    head = ArcFace(512, num_classes=num_classes, s=args.scale, m=args.margin).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()),
                            lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_eer = 1.0
    for ep in range(args.epochs):
        encoder.train()
        total, n = 0.0, 0
        for it, (img, label) in enumerate(train_dl):
            if args.iters and it >= args.iters:
                break
            img, label = img.to(device), label.to(device)
            if args.backbone == "iresnet18":
                emb = encoder(img, phase="infer")  # TopoFR iResNet: infer 返回嵌入
            else:
                emb = encoder(img)
            loss = head(emb, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        sched.step()

        encoder.eval()
        with torch.no_grad():
            for img, label in val_dl:
                _ = encoder(img.to(device))
                break  # 仅冒烟: 全量验证交给 evaluate_verification
        m = evaluate_verification(encoder, val_ds, device)
        print(f"ep {ep:03d} loss {total / max(n, 1):.4f} eer {m['eer']:.4f}")
        if m["eer"] < best_eer:
            best_eer = m["eer"]
            torch.save({"encoder": encoder.state_dict(), "head": head.state_dict(),
                        "backbone": args.backbone,
                        "classes": getattr(train_ds, "classes", None)}, out / "best.pt")
    print(f"best EER {best_eer:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()

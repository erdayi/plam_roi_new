# -*- coding: utf-8 -*-
"""跨域评测: 用 CASIA 训练的 checkpoint 在 500w ROI 上测验证 EER.

用法:
  python scripts/eval_crossdomain.py --ckpt runs/phaseA_casia_r18/best.pt \
      --data_root E:/data/phaseA/green_roi112
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "TopoFR"))

import torch
from torchvision import transforms, datasets

from backbones.iresnet import iresnet18
from palmroi_kpt.recog.metrics import evaluate_verification

VAL_TF = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/phaseA_casia_r18/best.pt")
    ap.add_argument("--data_root", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    backbone = ckpt.get("backbone", "iresnet18")
    if backbone == "lvface_s":
        import importlib.util
        lv_dir = Path(__file__).resolve().parents[1] / "third_party" / "LVFace" / "backbones"
        spec = importlib.util.spec_from_file_location(
            "lvface_backbones", lv_dir / "__init__.py",
            submodule_search_locations=[str(lv_dir)])
        lv_pkg = importlib.util.module_from_spec(spec)
        sys.modules["lvface_backbones"] = lv_pkg
        spec.loader.exec_module(lv_pkg)
        encoder = lv_pkg.get_model("vit_s", num_features=512).to(device)
    else:
        n_cls = ckpt["encoder"]["weight"].shape[0]
        encoder = iresnet18(num_features=512, dropout=0.4, num_classes=n_cls).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    print(f"loaded {args.ckpt} (backbone={backbone})")

    ds = datasets.ImageFolder(args.data_root, transform=VAL_TF)
    print(f"probe set: {len(ds)} images, {len(ds.classes)} IDs")
    m = evaluate_verification(encoder, ds, device)
    for k, v in m.items():
        print(f"{k}: {v:.4f}" if v == v else f"{k}: nan")


if __name__ == "__main__":
    main()

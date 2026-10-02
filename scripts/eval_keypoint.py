# -*- coding: utf-8 -*-
"""定位线评测: 加载训练好的 checkpoint, 输出 NLE/SR 指标 + JSON/Markdown 报告 + 可视化.

用法:
  python scripts/eval_keypoint.py --ckpt runs/m2/best.pt --data_root E:/data/20k_val \
      --vis 24 --out_dir runs/m2/eval
输出:
  eval/results.json   # 全部指标 (可入库/画图)
  eval/results.md     # 人读表格
  eval/vis/*.jpg      # GT(绿) vs 预测(红) 对照图
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from palmroi_kpt.datasets.keypoint_dataset import PalmKeypointDataset
from palmroi_kpt.models.keypoint_net import PalmKeypointNet
from palmroi_kpt.recog.metrics_keypoint import nle_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default="D:/dataset/roi/MobileNet_Data")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--vis", type=int, default=24, help="可视化张数")
    args = ap.parse_args()
    out_dir = Path(args.out_dir or Path(args.ckpt).parent / "eval")
    (out_dir / "vis").mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    targs = ckpt["args"]
    ds = PalmKeypointDataset(args.data_root, targs["input_size"], targs["simcc_res"], train=False)
    K = ckpt.get("num_keypoints", ds.num_keypoints)
    model = PalmKeypointNet(targs["backbone"], pretrained=False, num_keypoints=K,
                            input_size=targs["input_size"], simcc_res=targs["simcc_res"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dl = DataLoader(ds, batch_size=64, num_workers=2)
    preds, gts = [], []
    vis_saved = 0
    with torch.no_grad():
        for bi, (img, k_bin, k_norm) in enumerate(dl):
            sx, sy, _ = model(img.to(device))
            p = model.head.decode(sx, sy) / targs["simcc_res"]
            preds.append(p.cpu().numpy())
            gts.append(k_norm.numpy())
            # 可视化 (从原始文件取图, 画原始分辨率坐标)
            if vis_saved < args.vis:
                for j in range(img.shape[0]):
                    if vis_saved >= args.vis:
                        break
                    idx = bi * dl.batch_size + j
                    if idx >= len(ds):
                        break
                    img_fp, json_fp = ds.samples[idx]
                    raw = cv2.imread(str(img_fp))
                    ann = json.loads(Path(json_fp).read_text(encoding="utf-8"))
                    raw = cv2.resize(raw, (targs["input_size"],) * 2)
                    scale = targs["simcc_res"] / targs["input_size"]
                    for i in range(K):
                        gx = int(k_norm[j, i, 0] * targs["input_size"])
                        gy = int(k_norm[j, i, 1] * targs["input_size"])
                        px = int(p[j, i, 0] / scale)
                        py = int(p[j, i, 1] / scale)
                        cv2.circle(raw, (gx, gy), 6, (0, 255, 0), -1)   # GT 绿
                        cv2.circle(raw, (px, py), 6, (0, 0, 255), -1)   # 预测 红
                    cv2.imwrite(str(out_dir / "vis" / f"{Path(img_fp).stem}.jpg"), raw)
                    vis_saved += 1

    m = nle_metrics(np.concatenate(preds), np.concatenate(gts))
    (out_dir / "results.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    rows = ["| 指标 | 数值 |", "|---|---|"]
    for k, v in m.items():
        rows.append(f"| {k} | {v:.4f} |")
    (out_dir / "results.md").write_text(
        f"# 定位评测报告\n\n- ckpt: {args.ckpt}\n- data: {args.data_root} "
        f"({len(ds)} images)\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(m, indent=1))
    print(f"report -> {out_dir / 'results.md'} ; vis x{vis_saved}")


if __name__ == "__main__":
    main()

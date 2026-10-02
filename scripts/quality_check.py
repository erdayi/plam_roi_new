# -*- coding: utf-8 -*-
"""GT 质检 harness: 对已有 3 点标注做三引擎体检.

引擎:
  E1 几何有效性 (零成本, 必跑): 三点构成合理三角形 (谷点在上/掌心在下、边长比 sane)
  E2 WiLoR 预标对比 (可选, 需 GPU): WiLoR 21 点 -> 推算 2 谷点(指缝中点) + 掌心(手掌框中心),
     与已有标注算像素距离
  E3 一致性分级: E2 距离 < agree_px 直接入库; >= agree_px 进修正队列

输出: qc_report.json + qc_report.md + disagree_list.txt (待人工修正队列)

用法:
  python scripts/quality_check.py --data_root E:/data/20k --out_dir E:/data/20k_qc
  python scripts/quality_check.py --data_root ... --wilor   # 加跑 E2
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np


def geometric_validity(kpts: dict) -> tuple[bool, str]:
    """E1: 三点几何合理性. kpts: {valley1:[x,y], valley2:[x,y], center:[x,y]}."""
    v1, v2, c = (np.array(kpts[k], float) for k in ("valley1", "valley2", "center"))
    d12 = np.linalg.norm(v1 - v2)
    d1c = np.linalg.norm(v1 - c)
    d2c = np.linalg.norm(v2 - c)
    if min(d12, d1c, d2c) < 5:
        return False, "points_too_close"
    # 谷点间距应与谷-心距离同量级 (0.3 ~ 3.0)
    ratio = d12 / max((d1c + d2c) / 2, 1e-6)
    if not (0.3 < ratio < 3.0):
        return False, f"bad_ratio_{ratio:.2f}"
    # 谷点应在掌心上方 (y 更小; 允许 ±60° 倾斜 -> 用点积判)
    up = (c - v1) + (c - v2)          # 合向量指向掌心反方向
    if up[1] <= 0:
        return False, "valleys_not_above_center"
    return True, "ok"


def wilor_prelabel(img, detector):
    """E2: WiLoR 21 点 -> (valley1, valley2, center) 候选.

    谷点 = 掌纹流水线同款: kp5-kp9 与 kp13-kp17 的中点 (MediaPipe 序).
    掌心 = 21 点均值.
    返回 dict 或 None.
    """
    k = detector.predict(img)
    if k is None or not getattr(k, "keypoints", None):
        return None
    kp = np.array(k.keypoints, float)
    valley1 = ((kp[5] + kp[9]) / 2).tolist()
    valley2 = ((kp[13] + kp[17]) / 2).tolist()
    center = kp.mean(axis=0).tolist()
    return {"valley1": valley1, "valley2": valley2, "center": center}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=r"D:\datasetoi\MobileNet_Data")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--wilor", action="store_true", help="加跑 WiLoR 预标对比 (需 GPU)")
    ap.add_argument("--agree_px", type=float, default=15.0, help="E2 一致判定阈值 (像素)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    root = Path(args.data_root)
    pairs = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} and p.with_suffix(".json").exists():
            pairs.append(p)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"{len(pairs)} labeled images")

    detector = None
    if args.wilor:
        sys.path.insert(0, "E:/code/palm_roi")
        from models.wilor_detector import WilorDetector
        from config.settings import get_config
        get_config()
        detector = WilorDetector(device="cuda")

    stats = {"total": len(pairs), "geo_ok": 0, "geo_bad": 0, "bad_reasons": {},
             "wilor_compared": 0, "agree": 0, "disagree": 0}
    disagree = []
    for i, img_fp in enumerate(pairs):
        ann = json.loads(img_fp.with_suffix(".json").read_text(encoding="utf-8"))
        ok, reason = geometric_validity(ann)
        if ok:
            stats["geo_ok"] += 1
        else:
            stats["geo_bad"] += 1
            stats["bad_reasons"][reason] = stats["bad_reasons"].get(reason, 0) + 1
            disagree.append(f"{img_fp}\tE1:{reason}")
            continue
        if detector is not None:
            img = cv2.imread(str(img_fp))
            pre = wilor_prelabel(img, detector)
            if pre is None:
                continue
            stats["wilor_compared"] += 1
            d = max(np.linalg.norm(np.array(ann[k]) - np.array(pre[k]))
                    for k in ("valley1", "valley2", "center"))
            if d <= args.agree_px:
                stats["agree"] += 1
            else:
                stats["disagree"] += 1
                disagree.append(f"{img_fp}\tE2:max_dist={d:.0f}px")
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(pairs)}")

    (out / "qc_report.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    (out / "disagree_list.txt").write_text("\n".join(disagree), encoding="utf-8")
    md = ["# GT 体检报告", "",
          f"- 总量: {stats['total']}",
          f"- 几何有效: {stats['geo_ok']} ({stats['geo_ok']/max(stats['total'],1):.1%})",
          f"- 几何异常: {stats['geo_bad']} ({json.dumps(stats['bad_reasons'], ensure_ascii=False)})"]
    if detector is not None:
        md += [f"- WiLoR 对比: {stats['wilor_compared']} 张, "
               f"一致 {stats['agree']}, 分歧 {stats['disagree']} "
               f"({stats['disagree']/max(stats['wilor_compared'],1):.1%})",
               f"- 修正队列: {out / 'disagree_list.txt'}"]
    (out / "qc_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

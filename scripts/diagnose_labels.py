# -*- coding: utf-8 -*-
"""标注格式诊断: 扫描数据集里所有 json, 统计键名组合分布, 找出格式变体.

用法:
  python scripts/diagnose_labels.py                # 用默认数据路径
  python scripts/diagnose_labels.py --data_root D:/xxx
输出: 键名组合分布 (数量/占比/示例文件) —— 粘贴回对话用于适配解析器.
"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path

DEFAULT_ROOT = "D:/dataset/roi/MobileNet_Data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DEFAULT_ROOT)
    ap.add_argument("--max_samples", type=int, default=3, help="每种格式打印的示例数")
    args = ap.parse_args()

    sig = Counter()
    samples = {}
    parse_errors = 0
    files = glob.glob(str(Path(args.data_root) / "**" / "*.json"), recursive=True)
    print(f"scanning {len(files)} jsons under {args.data_root}")
    for f in files:
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            sig["__PARSE_ERROR__"] += 1
            samples.setdefault("__PARSE_ERROR__", []).append(f)
            continue
        if isinstance(d, dict):
            s = tuple(sorted(d.keys()))
        else:
            s = (f"__TYPE__:{type(d).__name__}",)
        sig[s] += 1
        samples.setdefault(s, []).append(f)

    print(f"\ntotal: {len(files)}")
    for s, c in sig.most_common(15):
        print(f"\n[{c} 个, {c/len(files):.1%}] keys = {s}")
        for ex in samples[s][: args.max_samples]:
            print("   example:", ex)
            try:
                print("   content :", Path(ex).read_text(encoding="utf-8")[:200])
            except Exception as e:
                print("   read err:", e)


if __name__ == "__main__":
    main()

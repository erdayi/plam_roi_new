# -*- coding: utf-8 -*-
"""数据集打包脚本(拷到另一台电脑运行, 无第三方依赖).

把多个来源的手掌图像 + 三点标注文件打包成分卷 zip + manifest 清单, 便于拖拽传输.

用法 (每个数据集根目录跑一次, name 标记来源):
  python pack_dataset.py --src D:/数据集A --name A --out D:/to_transfer
  python pack_dataset.py --src D:/数据集B --name B --out D:/to_transfer
生成:
  D:/to_transfer/manifest_A.jsonl     # 清单: 每张图一行 (相对路径/标注文件/大小)
  D:/to_transfer/pkg_A.zip.001 ...    # 分卷压缩包 (拖拽传输用)

传输到本机后放 E:/data/incoming/, 解压合并即可.
"""
import argparse
import json
import zipfile
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
LABEL_EXT = {".npy", ".json", ".txt", ".csv", ".xml"}
CHUNK_BYTES = 1500 * 1024 * 1024  # 1.5GB/卷


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="数据集根目录")
    ap.add_argument("--name", required=True, help="来源标记 (如 hfut/bjtu)")
    ap.add_argument("--out", required=True, help="打包输出目录")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_fp = out / f"manifest_{args.name}.jsonl"

    files = []  # (abs_path, arcname)
    n_img, n_lbl, n_orphan = 0, 0, 0
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src).as_posix()
        if p.suffix.lower() in IMG_EXT:
            files.append((p, f"{args.name}/{rel}"))
            n_img += 1
        elif p.suffix.lower() in LABEL_EXT:
            files.append((p, f"{args.name}/{rel}"))
            n_lbl += 1

    # 找没有标注文件的图 (orphan, 供人工确认)
    label_stems = {Path(a).stem for _, a in files if Path(a).suffix.lower() in LABEL_EXT}
    orphans = [a for _, a in files
               if Path(a).suffix.lower() in IMG_EXT and Path(a).stem not in label_stems]
    n_orphan = len(orphans)

    with open(manifest_fp, "w", encoding="utf-8") as f:
        for abs_path, arcname in files:
            f.write(json.dumps({
                "dataset": args.name,
                "path": arcname,
                "bytes": abs_path.stat().st_size,
                "is_label": Path(arcname).suffix.lower() in LABEL_EXT,
            }, ensure_ascii=False) + "\n")

    # 分卷 zip (zip64, 逐卷写)
    vol, written, zf = 1, 0, None
    for abs_path, arcname in files:
        if zf is None:
            fp = out / f"pkg_{args.name}.zip.{vol:03d}"
            zf = zipfile.ZipFile(fp, "w", zipfile.ZIP_STORED, allowZip64=True)
            written = 0
        zf.write(abs_path, arcname)
        written += abs_path.stat().st_size
        if written >= CHUNK_BYTES:
            zf.close()
            zf, vol = None, vol + 1
    if zf:
        zf.close()

    print(f"dataset={args.name}: images={n_img}, labels={n_lbl}, "
          f"images_without_label={n_orphan}")
    print(f"volumes: {vol}, manifest: {manifest_fp}")
    if n_orphan:
        print("注意: 有图像没有对应标注文件, 名单见 manifest 中 is_label=false 的行")


if __name__ == "__main__":
    main()

# roi-new: 掌纹 ROI 关键点定位(SimCC 三点谷点网络)

> 人脸 SOTA 工具链向掌纹 ROI 的系统移植。设计依据见 `docs/`(移植表 / 实验结论 / 指标规格)。

## 目录

```
palmroi_kpt/            # 核心包
  datasets/keypoint_dataset.py   # 图像+同名json(valley1/valley2/center) 数据集, ±180°旋转增广
  models/keypoint_net.py         # backbone + (可选)分割先验 + SimCC 三点头
  models/head_simcc.py           # SimCC 坐标分类头 (PIPNet/SimCC 范式)
  losses/topology.py             # 三角形拓扑损失(旋转/镜像不变)
  recog/metrics_keypoint.py      # NLE / SR@{5,10,15}% 指标
  recog/arcface.py, metrics.py   # 识别线(Phase B 联合训练用)
scripts/
  train_keypoint.py     # 定位训练器 (全参数可配置)
  eval_keypoint.py      # 评测: results.json/md + GT/预测可视化
  quality_check.py      # GT 体检: 几何有效性 + WiLoR 对比 + 修正队列
  prepare_casiams.py    # CASIA-MS 整理为 ImageFolder
  pack_dataset.py       # 跨机器打包传输 (分卷 zip + manifest)
  smoke_keypoint.py     # 合成数据全链路冒烟测试
docs/                   # 设计文档与实验结论
```

## 快速开始

```bash
# 1) 冒烟 (合成数据, 无需真实数据)
python scripts/smoke_keypoint.py

# 2) 训练 (数据: <root>/**/xxx.jpg + 同名 xxx.json 三点标注)
python scripts/train_keypoint.py --data_root E:/data/20k --out_dir runs/m2_v1 \
    --backbone resnet18 --epochs 60 --batch_size 64

# 3) 评测 (NLE/SR 报告 + 可视化)
python scripts/eval_keypoint.py --ckpt runs/m2_v1/best.pt --data_root E:/data/20k

# 4) GT 质检 (可选 WiLoR 对比)
python scripts/quality_check.py --data_root E:/data/20k --out_dir E:/data/20k_qc [--wilor]
```

## 输出指标规格

| 指标 | 定义 | 输出位置 |
|---|---|---|
| NLE(归一化定位误差) | `‖pred−gt‖ / d(valley1,valley2)` | 逐点(valley1/2/center)+总体 |
| SR@{5,10,15}% | NLE 小于阈值的样本比例 | 含"三点全对"严格口径 `sr*_all` |
| 训练曲线 | 每 epoch loss/topo/NLE/SR | `train_log.csv` |
| 评测报告 | 全指标 JSON + 人读 MD + GT绿/预测红对照图 | `eval/results.json\|md`、`eval/vis/` |

## 训练前的数据约定

- 图像任意常见格式,同名 `.json` 标注:`{"valley1":[x,y],"valley2":[x,y],"center":[x,y]}`(像素坐标)
- 不区分左右手、姿态任意(横/竖/旋转)——训练增广已含 ±180° 旋转,标注同步变换
- 先跑 `quality_check.py` 出体检报告,分歧队列人工修正后再冻结 GT 训练

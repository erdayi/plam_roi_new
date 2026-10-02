# 基于现成人脸识别代码改造掌纹方案(已拉库验证,v2 补 2025-2026 SOTA 核查)

> 结论:**可行,且是最快出活的路**。推荐底座 = **TopoFR(NeurIPS 2024 官方代码)**,它本身就是 insightface arcface_torch 的升级版,自带拓扑对齐损失(PTSA)——等于一次性拿到"SOTA 人脸训练框架 + 可迁移到掌纹的创新损失"。
> 已克隆到 `third_party/`:`TopoFR/`、`insightface/`(arcface_torch)、`SapiensID/`(CVPR 2025,可选进阶)。

## 〇、2025-2026 SOTA 核查(回应"TopoFR 是不是过时了")

2026-10 时点的人脸识别格局(已查 CVPR 2026 / ICCV 2025 / NeurIPS 2025 / InsightFace 官方月报):

| 方法 | 出处 | 地位 | 训练代码 |
|---|---|---|---|
| **[LVFace](https://github.com/bytedance/LVFace)** | **ICCV 2025 Highlight**(字节) | **当前基准 SOTA**(IJB-C TAR@1e-4=97.70,ViT-B/S/T/L 权重已放) | ❌ 仅推理(MIT);训练基于 arcface_torch 生态 |
| [TopoFR](https://arxiv.org/abs/2410.10587) | NeurIPS 2024 | 被 LVFace 超越,但仍是**最完整的可训练开源框架** | ✅ 完整 |
| [SapiensID](http://openaccess.thecvf.com/content/CVPR2025/papers/Kim_SapiensID_Foundation_for_Human_Recognition_CVPR_2025_paper.pdf) / 2.0 | CVPR 2025 | 免对齐基础模型范式 | 部分(依赖 Sapiens,重) |
| TransFace++ | TPAMI 2026 | ViT 方向延续 | — |
| SteerFace / IQ 评估 / 跨光谱蒸馏 | 2026-05 arXiv | 都在数据侧创新,**无新训练框架/损失** | — |

**修订后的双底座策略**:
1. **训练框架仍用 TopoFR**——LVFace 只放了推理代码,而它自己的训练也是 arcface_torch 生态;框架层(2024 底的 TopoFR)并不过时,过时的只是"把 ResNet50+ArcFace 当 SOTA"这个说法。
2. **LVFace-S/B 权重(非商业研究许可,学术可用)拿来做两件事**:①作为掌纹 encoder 的初始化(M1 迁移链的第一段从 InsightFace-ResNet 升级为 2025 SOTA ViT);②作为识别侧的 SOTA 对照线。
3. 论文写作时引用链:ArcFace → AdaFace → TopoFR → SapiensID → **LVFace**,证明我们挂靠的是最新谱系。

## 一、为什么 TopoFR 合适(对比 2023 老代码)

| 仓库 | 年份 | 问题/优势 |
|---|---|---|
| CCNet | TIFS 2023 | 你说得对:识别机制偏老,且只是识别骨干,训练框架一般 |
| arcface-pytorch | 2020-2021 | 太老,弃用 |
| **insightface arcface_torch** | **持续维护** | 人脸识别事实标准训练框架;TopoFR 的底座 |
| **TopoFR** | **NeurIPS 2024** | ⭐ 推荐:新 SOTA + 拓扑对齐损失可直接迁移掌纹关键点论文 |
| SapiensID | CVPR 2025 | 最新免对齐范式;依赖 Sapiens 基础模型(重),Phase C 再考虑 |

## 二、TopoFR 代码体检(已拉库确认)

```
TopoFR/
├── train.py / losses.py / partial_fc.py   # ArcFace 训练框架,数据无关 ✅直接用
├── dataset.py          # 第 43-49 行: else 分支 = torchvision ImageFolder
│                       #   → 掌纹库按 "subject_id/xxx.jpg" 建目录即可直接喂 ✅
├── backbones/          # iresnet.py (iResNet18/34/50/100), mobilefacenet ✅直接用
├── configs/            # ms1mv2_r50.py 等, 改数据路径/类别数/分辨率即可 🔧小改
├── topology.py / persistent_homology.py / GUM.py   # PTSA 拓扑对齐 + 难样本挖掘
│                       #   纯嵌入空间操作, 数据无关 ✅直接用(也是论文引用点)
└── eval/               # IJBC 验证协议 → 改成掌纹跨库 pair 协议 🔧中改
```

**要新写的只有一件事:掌纹没有"对齐好的 112×112 裁片"** ——人脸代码假定输入已对齐(RetinaFace 5 点裁剪),掌纹的对应物就是"谷点→坐标系→ROI"。这正是我们已有的关键点前端,两阶段接上。

## 三、两阶段落地

### Phase A:纯识别适配(1-2 周,先拿基线)
1. 用现有 palm_roi 流水线(WiLoR 关键点→谷点→透视变换)把 IITD/CASIA/绿幕图批量裁成 128×128 ROI,按 `subject_id/img.jpg` 建 ImageFolder。
2. `configs/palm_iitd_r50.py`:数据路径、`num_classes`(IITD 235 类)、`image_size`(112→128,backbone 全卷积天然支持)、batch/margin 微调(小类别数时 m 适当降到 0.4、s 64 起步)。
3. `dataset.py` transform 改:去人脸专属增广(无横向翻转——**掌纹左右手不能翻**),加轻量色温/模糊增广。
4. 训练 + 用 `eval/` 改的跨库 pair 协议出 EER/TAR@FAR。
→ **产出**:SOTA 级掌纹识别基线(对比 BOCV/CCNet),同时验证"人脸代码吃掌纹"这条路真的通。

### Phase B:接关键点前端,联合训练(论文核心)
- 把 Phase A 的"离线裁 ROI"换成**可微前端**:`encoder(关键点头)→ 仿射对齐 → TopoFR encoder → ArcFace`,端到端。
- TopoFR 的 PTSA 拓扑损失同时作用于**嵌入空间**(原设计)和**关键点四边形**(我们的 M3),双拓扑对齐是论文的独特故事:"人脸的拓扑对齐思想,在掌纹中同时约束特征与几何"。
- 联合损失:`L = L_arcface + L_ptsa + λ·L_simcc + μ·L_kpt_topo + γ·L_seg`(分割头可选)。

### Phase C(可选加分):SapiensID 式免对齐
- 若 Phase B 结果好,引 SapiensID 做讨论/对照:"人脸已免对齐,掌纹联合框架的极限在哪"。

## 四、风险

- TopoFR 训练脚本假定多卡 + RecordIO 高性能管线;单卡跑 ImageFolder 分支要改 DataLoader 配置(已确认 ImageFolder 分支存在,改动小)。
- 掌纹类内方差远大于人脸(姿态/光照),margin 超参要重调;Phase A 先扫一组。
- iResNet 在 128×128 输入的下采样步幅与 112 略有差异,注意关键点前端输出分辨率对齐。

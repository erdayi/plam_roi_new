# Pipeline 设计:基于 PKLNet 血统 baseline 的人脸技术迁移 + 掌纹微调

> 骨架来源:**沿用 `E:\code\palm_roi` 现有 baseline**(ResNet18 分流 → WiLoR/PartialPalmNet 关键点 → 谷点 → 透视变换 ROI),**不引入 MediaPipe**。本仓库在骨架之上做四个创新模块,目标产出:掌纹 ROI 关键点提取的高质量论文(冲击 TIFS / Pattern Recognition 级)。

## 一、现有 baseline 的问题诊断(= 创新点的依据)

| # | 问题 | 现状 | 后果 |
|---|---|---|---|
| P1 | **谷点是间接推断的** | 谷点 = kp5-kp9、kp13-kp17 关键点的中点 | 手部关键点误差在取中点时放大;网络从未直接见过"谷点"这个目标 |
| P2 | **域差距** | WiLoR 在通用手部数据集(ARCTIC/HAND)上预训练 | 绿幕/非受控掌纹图像的分布(俯视、张开、固定距离)与训练域不同,21 点在掌纹场景下未对齐优化 |
| P3 | **无结构约束** | 各关键点独立回归 | 谷点间距/角度等强拓扑先验被浪费,异常帧(遮挡、截断)易出离谱点 |
| P4 | **泛化未验证** | 只在自采绿幕库上闭环 | 跨库(IITD/CASIA/X-Palm)表现未知,而"泛化性"正是顶刊关心点 |

## 二、创新模块设计(四件套,每件都可单独消融)

### M1 面部→手部→掌纹 预训练初始化(借力人脸,解决 P2)
- encoder 用三段式权重初始化:面部关键点数据(WFLW,98 点)→ 手部(FoundHand-10M 子集 或 COCO-WholeBody 21 点)→ 掌纹微调。
- 具体实现: ResNet18/MobileNetV3 encoder 先加载 InsightFace 骨干权重(人脸识别预训练, 学到强边缘/纹理表征), 再在 FoundHand 关键点上热身, 最后掌纹微调。
- **论文故事**: "人脸识别预训练表征对掌纹关键点的迁移增益"——LAFS 的反向应用, 从没人做过。

### M2 SimCC 亚像素关键点头(升级 PKLNet 的回归头, 解决 P1)
- 把 PKLNet 的 global+local 回归融合替换/并联为 SimCC 式坐标分类(x-bin / y-bin 两分类), 亚像素精度, 免热图解码, 实时性好。
- 直接监督 3 个谷点 + 1 个腕点(不再经 21 点取中点), 输出即坐标系锚点。
- 可保留热图分支做双头一致性(训练时互蒸馏, 推理只用 SimCC, 零额外开销)。

### M3 关键点图拓扑对齐损失(把 TopoFR 的 PTSA 搬进关键点空间, 解决 P3)
- 谷点 V1/V2/V3 + 腕点 W 构成稳定四边形; 定义归一化形状描述子(成对距离比 + 对角角度), 对 GT 与预测强制分布一致。
- 实现: `losses/topology.py` 的 `PalmTopologyLoss`(已写好可跑), 与 TopoFR 引用关系直接可写。

### M4 跨域泛化协议(解决 P4, 也是论文主实验)
- 训练: 绿幕自采 + IITD Touchless(PalmKit 自动 GT)+ 生成增强(RPG-Palm/Canny2Palm 可选)。
- 评测: 跨库协议 train-on-A/test-on-B(IITD↔CASIA↔X-Palm 多光谱↔手机), 指标 NLE/SR + 下游识别 EER、TAR@FAR=1e-4(接 palm_roi 的 BOCV/EBOCV 管线)。
- 对照基线: PKLNet 原版、WiLoR 零样本、MediaPipe 零样本、传统谷点切线法、(可选)SAM2 零样本科胚分割。

## 三、目录结构

```
roi-new/
├── docs/                    # 本文档 + 调研
├── palmroi_kpt/             # 核心包
│   ├── models/
│   │   ├── backbone.py      # encoder + 面部/手部预训练加载 (M1)
│   │   └── head_simcc.py    # SimCC 亚像素关键点头 (M2)
│   ├── losses/
│   │   └── topology.py      # 掌纹关键点图拓扑对齐损失 (M3)
│   └── datasets/            # PalmKit GT 转换 + 跨库 loader (M4, 数据就位后填)
├── baselines/               # 对照: 传统切线法 / PKLNet / WiLoR 零样本封装
├── configs/                 # 训练配置
└── scripts/                 # smoke_test.py 冒烟测试 / 数据准备
```

## 四、里程碑(建议 8-10 周节奏)

| 周 | 里程碑 | 验收标准 |
|---|---|---|
| W1-2 | PalmKit 对 IITD/CASIA 生成谷点 GT; 跑通 SimCC 头 + 拓扑损失的 smoke test | GT 可视化抽查; 冒烟测试通过 |
| W3-4 | M1-M3 组合训练 v1(单库 IITD) | NLE/SR 超 WiLoR 零样本与中点法 |
| W5-6 | 跨库协议 + 对照基线补齐 | 主表 + 消融表成型 |
| W7-8 | 下游识别闭环(BOCV/EBOCV EER) | 证明 ROI 质量提升 → 识别收益 |
| W9-10 | 写作 | 初稿 |

## 五、风险与备选

- 面部权重迁移增益不显著 → 消融如实报告, 主卖点转 M2+M3(结构创新不依赖迁移)。
- PalmKit 环境难复现 → 用 mediapipe/人工半自动标注 500 张 + 迭代自训练。
- X-Palm 数据申请周期长 → 主实验用 IITD/CASIA, X-Palm 作补充。

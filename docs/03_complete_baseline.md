# 完整掌纹 Baseline 调研与统一多任务方案(v3)

> 修正:v2 的 M1-M4 是"在你 palm_roi 流水线上做增量",但那条流水线(3D 姿态估计→后处理→校验)本身不可端到端训练。
> 本版重新调研**完整可训练的掌纹 baseline**,并把语义分割、人脸识别方法作为训练组件嫁接进去。

## 一、现有"完整 baseline"盘点(谁最接近可端到端)

| 方案 | 出处 | 组成 | 可训练性 | 代码 |
|---|---|---|---|---|
| CCNet | IEEE TIFS 2023 | 竞争机制 + 多阶纹理特征识别 | 仅识别(输入是裁好的 ROI) | ✅ [官方](https://github.com/Zi-YuanYang/CCNet) |
| ArcFace-Palm | Geometric Synthesis / Canny2Palm 用作基线 | ArcFace 识别 | 仅识别 | ✅ [arcface_torch](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/README.md) / [arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch) |
| ROI3Net | 电子与信息学报 2025 | MobileOne+SimCC 定位 → 3 尺度 ROI → Gabor 融合识别 | 定位与识别**分离训练**再融合 | ❌ 无公开代码 |
| One-Stage MPN | ITC-CSCC 2023 | MobileNetV3 关键点 + CosFace 识别 | 一体化,但极轻量、会议级 | ❌ |
| BPFNet | ICONIP 2021 | 检测 + 对齐 + 双模态融合 | 部分联合 | ❌ |
| **空白** | — | **分割 + 关键点 + ROI + 识别 联合训练的统一框架** | — | **无人做过** |

**关键判断**:定位+识别一体化的最好工作(ROI3Net、MPN)都停在期刊/会议中档,而 TIFS 级的掌纹工作(CCNet 等)全部假设 ROI 已裁好。**"统一多任务"这个生态位是空的**——人脸领域恰好有直接可抄的先例:**RetinaFace 就是"检测 + 5 点关键点 + 识别特征"的多任务联合训练**(CVPR 2020,被引数千),这个范式从没被搬进掌纹。

## 二、方案:PalmMT — 掌纹统一多任务网络(拟名)

```
输入原图 (绿幕 / 非受控)
   │
   ┌─────────────┴──────────────┐
   │   共享 encoder (M1: 人脸→手→掌纹 预训练初始化)   │
   └─────────────┬──────────────┘
   ├─ Head A  语义分割: 手掌/背景 mask (Dice+CE)
   │           → 分割边界先验喂给关键点头 (谷点在手掌轮廓凹陷处)
   ├─ Head B  SimCC 关键点头: 3 谷点 + 腕点直接监督 (亚像素)
   │           + 拓扑对齐损失 (四边形形状分布约束)
   ├─ 可微 ROI 对齐: 关键点参数化仿射变换 → grid_sample 裁标准 ROI
   │           (特征层 ROI 对齐, 端到端可导)
   └─ Head C  识别嵌入: ArcFace margin loss (与 CCNet 可互换)

联合损失: L = λ_seg·L_seg + λ_kpt·L_simcc + λ_topo·L_topo + λ_id·L_arcface
```

**为什么联合训练会涨点(论文的理论故事)**:
1. 分割→关键点:手掌轮廓是谷点定位的几何先验(你的 A-10 论文已验证分割有效,但只当预处理用);多任务让共享特征同时"看见"边界和纹理。
2. 识别→关键点(核心创新):识别损失对"定位不确定性"最敏感——ROI 偏 1 个像素,类内方差就增大。端到端反传时,ArcFace 梯度会**自动把关键点拉向对识别最有利的位置**,这是"取中点"/单监督关键点永远做不到的。人脸领域 LAFS(CVPR 2024)证明过关键点-识别互益,掌纹无人验证。
3. 可微 ROI 对齐替代"透视变换后处理":pipeline 里不可导的环节变成网络的一部分。

## 三、与 v2 四模块的关系

v2 的 M1(迁移预训练)、M2(SimCC)、M3(拓扑损失)全部保留,成为 PalmMT 的三个子模块;**新增 Head A(分割)与 Head C(ArcFace)+ 可微 ROI 对齐**,把"流水线"升级成"一个网络"。

## 四、训练与评测协议

- **数据**:绿幕自采(有)+ IITD Touchless + CASIA(PalmKit 生成谷点 GT)+ 可选 RPG-Palm/Canny2Palm 生成增强。
- **识别 GT**:公开库自带身份标签;绿幕库按采集者标 ID。
- **主表**(跨库泛化):train IITD → test CASIA / X-Palm,报 NLE、SR、EER、TAR@FAR=1e-4。
- **对照**:①你现在 WiLoR+BOCV 流水线 ②PKLNet 复现 ③CCNet(ROI 给定,识别上限)④ROI3Net 复现 ⑤"分割/识别损失拆掉"的消融(证明多任务必要性)。
- **消融表**:−分割头 −ArcFace −拓扑损失 −迁移初始化,四行。
- **指标目标**:识别侧对齐 CCNet 报告水平;定位侧 NLE 低于 PKLNet;跨库退化幅度(variance)显著小于 baseline——泛化是主卖点。

## 五、风险

- 联合训练不稳定 → λ 预热策略(先定位收敛再放开 ID 损失),渐进联合。
- 显存:特征层 grid_sample 额外开销小(ROI 只有 128×128),ResNet18 级 encoder 无压力。
- 若 ArcFace 梯度对关键点影响过强 → 停梯度隔离一版做对照(论文里也是一组消融)。

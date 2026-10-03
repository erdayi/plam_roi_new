# 可迁移模块调研报告:语义分割 / 目标检测 / 关键点 / 生物特征(2025-2026)

> 调研时间:2026-10。来源:arXiv、CVPR/ICCV/ICLR/NeurIPS/IJCB 官方信息与检索验证。
> 目的:为掌纹 ROI 谷点定位网络(roi-new)寻找可迁移复用的 SOTA 模块。
> 标注:✅=已验证存在(检索命中 venue/代码);⚠️=单源信息,使用前需二次核对。

## 一、语义分割(解决"分割先验分支"和"自动打标老师")

| 工作 | 出处 | 核心内容 | 对掌纹 ROI 的迁移方式 |
|---|---|---|---|
| **SAM 3**("Segment Anything with Concepts") | Meta,2025-11 ✅ | 可提示**概念**分割(文本+示例提示),统一图像/视频的检测+分割+跟踪;自带 EfficientViT 高效变体(ES-EV-S 约 60FPS,精度损失 ~10%) | **离线标注老师首选**:输入文本概念 "palm/hand" 即可批量给 2 万张打手掌 mask,喂给我们的分割分支(SAM2 都不用装了) |
| EoMT("Your ViT is Secretly an Image Segmentation Model") | CVPR 2025 ✅ | 去掉 Mask2Former 的重型 decoder,纯 ViT + 轻量 query/mask 头达到通用分割 SOTA | 架构启示:我们的"backbone+轻量头"路线与 SOTA 演进方向一致;引用其"简化头不减精度"的论据 |
| EfficientViT-SAM / EdgeSAM / MobileSAM | 2024-2025 ✅ | SAM 蒸馏出的轻量系(EdgeSAM: prompt-in-the-loop 蒸馏,面向端侧) | 若 SAM3 太重,用 EfficientViT-SAM 当蒸馏老师的替代;EdgeSAM 的蒸馏方法可参考 |
| MaskDINO / OneFormer | 持续有效 | query-based 通用分割基准 | related work 引用,非直接复用 |

**结论**:分割先验分支的"老师"用 **SAM3(2025-11 新鲜出)**,"学生"用我们已规划的 BiSeNet 式轻量头——老师和学生的代差越小越好,2025-11 的 SAM3 比 2023 的 SAM2 值得等。

## 二、目标检测(解决"复杂背景手掌检测"层)

| 工作 | 出处 | 核心内容 | 迁移方式 |
|---|---|---|---|
| **D-FINE** | ICLR 2025 ✅ | 把 DETR 边框回归重新定义为细粒度分布细化,实时 SOTA | 若 L0 检测层需要升级(复杂背景误检时),手掌检测器直接换 D-FINE 预训练权重微调 |
| **DEIM** | CVPR 2025 ✅ | DETR 改进匹配(Dense O2O + MAL),超越 YOLO 系 | **训练策略借鉴**:匹配思想对 SimCC 的坐标分类训练有参考价值(引用级) |
| RF-DETR | Roboflow 2025 ✅(Apache 2.0) | NAS 优化的实时检测 Transformer,带关键点检测 preview | 备选检测底座;关键点 preview 值得跟进 |
| YOLOv12/v13 | 2025 ✅ | 注意力中心化 YOLO | 同上,备选 |

**结论**:维持现有 ResNet18 门控不变(已验证够用),**检测层的升级是后备项**——等 M2 主实验做完,如果复杂背景误检成为误差主因,再上 D-FINE。

## 三、关键点/姿态估计(我们的直接赛道)

| 工作 | 出处 | 核心内容 | 迁移方式 |
|---|---|---|---|
| Sapiens-2B | Meta,持续 SOTA ✅ | 人类视觉基础模型(2D pose/分割/深度/法线),Humans-5K 超 DWPose-L +7.1 AP | 21 点手部教师(替代/并列 WiLoR 蒸馏源);太重(2B),只当离线老师 |
| **RTMW**(MMPose) | 2024-2025 ✅ | 实时全身(身体+手+脸+脚)关键点,工业级 | **蒸馏老师的轻量替代**;MMPose 生态成熟,微调到谷点工程量小 |
| VLPose | 2025 ⚠️ | 视觉-语言模型做姿态,COCO +3.74% | 趋势引用:VLM 进关键点领域 |
| HashPose | OpenReview ⚠️ | 内存高效实时关键点(1/256 头显存) | 部署阶段参考 |
| ICCV 2025 soft-argmax 修正 | ICCV 2025 ✅ | 重审热图回归的 soft-argmax,结构化学习目标达 SOTA | **直接采纳进 SimCC 损失**(已在移植表) |
| WildHand(半监督手部关键点) | ScienceDirect ✅ | 大规模无标注数据半监督手部关键点 | **和我们"自训练扩充"路线同构**,方法可平移到掌纹谷点 |
| PACL(伪标签自动课程) | OpenReview ✅ | 半监督关键点的伪标签自动课程学习 | 我们"M2 给 CASIA 自动打标"应加课程/置信度筛选 |

**结论**:关键点头范式(SimCC)选对了,但有两个增强:①蒸馏老师可选 RTMW(轻)或 Sapiens(重);②自训练务必加**置信度筛选**(uncertainty-aware pseudo-label,综述见 PACL 一脉),这是文献验证过的必要组件。

## 四、其他生物特征(同赛道异模态的互相印证)

| 工作 | 出处 | 内容 | 对我们的印证/复用 |
|---|---|---|---|
| FingerVeinSyn-5M | arXiv 2025-06 ✅ | 5 百万张合成指静脉、5 万身份,最大合成指静脉库 | **"合成数据解决生物特征数据稀缺"已是顶会级共识**——直接支撑我们的合成背景/RPG-Palm 路线 |
| 背侧手静脉:contour-guided 凹陷点几何 ROI | 2025 ✅ | 用轮廓引导的凹陷点几何做精确 ROI | **方法同构**:手背静脉的"凹陷点定位 ROI"和我们的"谷点定位 ROI"是同一个数学问题,他们的几何校验规则可借来加强 quality_check |
| 多半径融合旋转 LBP 指静脉 | 2025-11 ✅ | 静脉线分割 + 识别 | related work 引用 |
| X-Palm / GBU-Palm / RPG-Palm / Canny2Palm | 2023-2026(掌纹侧,见 docs/01) | 数据集与生成 | 已纳入计划 |

**结论**:其他模态(指静脉/手背静脉)2025 年的工作确认了两件事:①合成数据是生物特征的主流补数据手段;②"几何关键点→ROI"仍是跨模态的主流 ROI 范式——我们的路线在更大坐标系里是顺势的。

## 五、可迁移模块优先级矩阵(汇总)

| 优先级 | 模块 | 来源 | 融入点 | 当前状态 |
|---|---|---|---|---|
| P0 | 谷点 GT 质检+自训练置信度筛选 | PACL/WildHand | quality_check / 自训练扩充 | harness 已写,置信度筛选待加 |
| P0 | SAM3 离线打手掌 mask | Meta 2025-11 | 分割分支的监督标签 | 待 SAM3 开放后接入 |
| P1 | RTMW 蒸馏谷点 | MMPose | 谷点头蒸馏(替代/并列 WiLoR) | 待 M2 基线出数后做 |
| P1 | ICCV25 soft-argmax 损失修正 | ICCV 2025 | SimCC 损失 | 计划内 |
| P2 | D-FINE 手掌检测 | ICLR 2025 | L0 检测层(仅当误检成主因) | 后备 |
| P2 | MagFace 质量门控 | CVPR 2021 沿用 | 低质量帧拒识 | 远期 |

## 六、对当前正在跑的实验的影响

**不影响,不中断**。正在跑的 M2 基线(resnet18+SimCC,2 万张)是消融表参照系,以上所有模块都是它之上的增量。本轮调研的 actionable 输出:①下一批实验清单更新如上;②related work 素材入 docs/01;③SAM3 发布信息修正了"用 SAM2 当老师"的旧计划。

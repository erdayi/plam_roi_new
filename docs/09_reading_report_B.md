# B 轮深度阅读报告:直接同题工作与可借鉴方法(2024-2026,49 篇)

> **方法**:20 组聚焦检索(arXiv+OpenAlex)→ 234 篇入池(2024-2026)→ 按六组精选 49 篇(12 篇带代码线索)→ 下载 28 篇 arXiv 原文逐篇读摘要+结论 + Springer 页面补读 + 代码仓库核查。
> **本轮目标**:回答"围绕当前基线(NLE 0.296 / SR@10 70.7%),该参考谁、避让谁、借什么"。
> 语料:`docs/survey2026b/`(pool 234 篇 / selected 49 / readings.json / pdfs 28 篇)。

## 一、直接同题工作(必引必比,定差异化的依据)

### 1.1 KCNet — 《ROI Extraction for Palm in the Wild》(Springer LNCS 2025)⭐⭐⭐ 最接近的同题工作

- **方法**:top-down **关键点分类网络**(KCNet)做**掌静脉** wild ROI 提取——与我们的 SimCC 坐标分类**同一范式**;
- **数据贡献**:给 SCUT_PV_v1、CASIA、TJ_PV、VERA 五库标注关键点 GT,并**首个构建野外掌静脉库 SCUT_PV_Wild**;
- **对我们的三重意义**:
  1. **范式验证**:关键点分类→生物特征 ROI 在 2025 年已发表且"remarkable results"——我们的路线站得住;
  2. **必须差异化**:他们=掌**静脉**+单模型+无训练创新栈;我们=掌**纹**+SimCC 亚像素+顺序不变匹配+拓扑约束+迁移初始化+LUPI 分割先验+2 万真实多姿态数据——论文里要有一张"vs KCNet"的对比行;
  3. **方法论印证**:他们"给多个公开库标注 GT"与我们的 PalmKit/标注审计计划完全同构——多人做过说明可行,且多库 GT 是进入顶刊的标配工作量。
- 全文在 Springer 付费墙内(DOI: 10.1007/978-981-96-1068-6_13),校园网获取。

### 1.2 Noncontact Palm Vein ROI via Improved Lightweight HRNet(2024)

- HRNet 关键点定位 → 掌静脉 ROI;重构残差块+深度可分离卷积做轻量化,面向低成本嵌入式;
- **启示**:①HRNet 高分辨率表征在静脉 ROI 有效——可作为我们 backbone 消融的候选;②"轻量化改造(重残差块+DSConv)"是可引用的工程配方;③它发在较好期刊说明"关键点→掌生物特征 ROI+轻量化"这个组合的发表窗口仍在。

### 1.3 Geometry-Guided Affine-Invariant ROI Extraction for Palm Biometrics(2026, Springer)

- 无 arXiv 版(DOI: 10.1109/icigp68997.2026.11619844 同会议系列);标题表明做仿射不变的掌纹 ROI——与我们的"任意姿态鲁棒"目标重叠,全文待校园网获取后精读;已知信息不足以判断机制(之前 X-08 待核对的就是它)。

### 1.4 组内既有(B-04 免谷点 ROI、AMCOA 机器人采集)——你的 PPT/收藏已覆盖,不赘述。

## 二、方法论迁移(每条都能落到我们的网络/训练里)

| 来源 | 方法 | 迁移到 roi-new | 优先级 |
|---|---|---|---|
| **PECC**(2026) | 坐标分类加**位置编码**直接嵌入关键点特征 + Filtering Amplified Attention | SimCC 头升级:位置编码进 head,治"SimCC 忽略坐标间空间关系"(ViTCC 同批评述原话) | P1(下一版头) |
| **ViTCC**(2024, ACM) | ViT 做坐标分类 backbone | backbone 消融已含 ViT 路线(LVFace-S),此文佐证方向 | ☆引用 |
| **GKDT**(ECCV 2026) | 通用关键点 Transformer;**代码 [AlanLuSun/General-Keypoint-Detection](https://github.com/AlanLuSun/General-Keypoint-Detection) + [MegaKPT 数据集](https://huggingface.co/datasets/changshenglu/MegaKPT)(HF)** | ①零样本对比行("通用大模型 vs 领域专用");②MegaKPT 可做 M1 迁移链的手部预训练源 | P1 |
| **DINOKey**(2026) | 点预测用 **L1 + focal + 平均 Hausdorff 距离**三损失(局部精度+全局一致性) | SimCC 损失可加 Hausdorff 项约束两谷点联合分布 | P2 |
| **Hand Visibility Detector**(2026, AIST) | **逐关键点可见性估计**(专用检测头) | 质量门控再升级:谷点可见性头(遮挡帧输出低可见性) | P2 |
| **Keypoint Self-Consistency "Meta Pose"**(2026) | 用**关键点几何自洽性**检测估计失败(轻量,免额外模型) | 我们已有拓扑损失,推理时把同款几何自洽分数当**免训练失败检测器**——几乎零成本 | **P0(下次 eval 就加)** |
| **Transformation-Isomorphic Latent Space**(2025) | 构造与变换同构的潜空间,回归任务收敛更快 | M1 预训练阶段的表示学习目标(替代/增强普通 ArcFace 预训练) | P2 |
| **PhysAstro-Pose**(2026) | 半监督姿态里加**结构拓扑+物理约束** | 佐证我们拓扑约束路线;其 SSL 框架可用于自训练扩充 | ☆ |

## 三、训练策略(跨数据集/多任务,正对我们的泛化主线)

| 来源 | 策略 | 借鉴点 |
|---|---|---|
| **FreqFLD**(2026) | **All-in-One**:一个模型统一多个人脸关键点数据集 | 我们的多库混训协议直接对标("all-in-one 掌纹谷点") |
| **EFLD**(2024) | **cross-format training**:混合多个公开数据集训练提升鲁棒性 + 边缘部署 | 混合数据训练配方;轻量 head 设计 |
| **FaceLift**(2024, Flawless AI) | 半监督 3D 关键点;处理**不同数据集 GT 定义不一致**问题 | 我们跨库时谷点定义可能有差异——GT 调和(harmonization)要有意识做 |
| **Cephalometric 域对齐**(MICCAI 2024 Challenge) | 域对齐 + 伪影增广策略提升跨源泛化 | 跨库训练的增广参考 |
| **AHMAD**(2026, INSAIT) | 多任务(稠密+稀疏预测)混合训练 + 辅助蒸馏 | 我们的分割+关键点(+识别)多任务参考 |
| **Accuracy Compensation**(2026) | 轻量化网络的精度补偿策略 | 部署阶段参考 |

## 四、代码可用清单(已核实仓库内容,2026-10 更新)

| 资源 | 地址 | 核实结果 | 用途 |
|---|---|---|---|
| **GKDT**(ECCV 2026) | github.com/AlanLuSun/General-Keypoint-Detection(已克隆 third_party/gkdt) | ✅ 完整训练/评测框架 + **GKDT-L/H 模型已放**(DINOv3 底座);**MegaKPT=29 库统一、130 万+实例,含 300W/OneHand/HInt/Hand X-ray**;支持 visual/text prompt 零样本,OneHand PCK@0.1 达 92-97 | ①零样本对比行;②M1 迁移链预训练源;③continual learning 接口可参考 |
| **Hand Visibility Detector**(2026) | github.com/ryhara/hand_visibility_detector(已克隆 third_party/) | ✅ 完整训练+推理+HF 模型+Demo;**基于 WiLoR-mini 构建**(与我们现有 WiLoR 管线同源!) | 谷点可见性头参考;质量门控;可零成本试跑 |
| SimCC 官方 | github.com/leeyegy/SimCC | 头实现对照 | 对照 |
| MMPose(RTMPose/RTMW) | github.com/open-mmlab/mmpose | 蒸馏老师/工程基线 | P1 |
| TopoFR / LVFace / SapiensID | 已克隆 third_party/ | M1 权重与对照 | 已用 |
| PKLNet 官方(组内) | github.com/xuliangcs/pklnet | 组内前作代码 | 对比基线 |
| RobustPalmRoi | github.com/leosocy/RobustPalmRoi | 手机图掌 ROI 开源实现 | related |

**KCNet 补充核实**:作者林浩恒等,华南理工大学(与贾伟组不同单位);CCBR 2024(论文集 LNCS 15352, 2025);**5 个关键点**,引 SimCC/RTMPose/Lite-HRNet(坐标分类+轻量 HR 谱系确认);无公开代码;全文付费(校园网可下)。SCUT_PV_v1 即该组知名掌静脉库。

## 五、取长补短行动项(合并进实验计划)

| # | 行动 | 来源 | 排期 |
|---|---|---|---|
| A1 | eval 加**几何自洽失败检测**(谷点三角形自洽分,免训练) | Meta Pose | 下次 eval |
| A2 | SimCC 头加**位置编码**(PECC 式) | PECC | M3 后的头升级 |
| A3 | **KCNet 对比行**+论文差异化段落(掌纹 vs 掌静脉、训练创新栈) | KCNet CCBR 2024/LNCS 2025(华南理工) | 写作期 |
| A4 | **GKDT 零样本行** + MegaKPT 手部预训练源(替代/并列 FoundHand) | GKDT ECCV 2026(代码已克隆核实) | M1 扩展 |
| A5 | 多库混训协议对标 FreqFLD/EFLD 的 all-in-one 配方 | 两者 | M4 跨库阶段 |
| A6 | **Hand Visibility Detector 试跑**(WiLoR-mini 同源,谷点可见性门控) | AIST 2026(代码已克隆核实) | 质量门控阶段 |
| A7 | 校园网获取 KCNet、Geometry-Guided、GKDT 全文精读 | 三篇 | 近期 |

## 六、规模与结论

- 池:234 篇(2024-2026);精选:49 篇;PDF 下载精读:28 篇;Springer 页面补读:1 篇;代码核查:GKDT(有)、ViTCC(无)、PECC(未放出)。
- **一句话结论**:我们的"SimCC 谷点 + 迁移初始化 + 结构约束"路线在 2025-2026 的版图里**恰好站在刚被验证的风口上**(KCNet 同范式发表、PECC/ViTCC 改进坐标分类、GKDT 把关键点推向通用大模型)——差异化空间完整存在:领域(掌纹 wild RGB)+ 训练创新栈 + 跨模态迁移,三者组合无人做过。

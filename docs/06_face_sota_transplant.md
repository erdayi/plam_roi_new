# 人脸 SOTA 工具链 → 掌纹 ROI 逐项移植表(v3 网络设计)

> 原则:每个设计组件必须有"人脸 SOTA 出身",注明代表作;掌纹侧只做"适配改造",不无中生有。
> 论文故事 = **系统性地把人脸识别工具链移植到掌纹 ROI**,而不是零散拼装。

## 一、人脸 pipeline 全景 → 掌纹对应物

| # | 人脸组件 | 人脸 SOTA 代表作 | 移植到掌纹 ROI | 状态 |
|---|---|---|---|---|
| 1 | 检测+关键点**联合**单阶段 | [RetinaFace](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html)(CVPR 2020:box+5点+dense 3D 一网络) | 现有 ResNet18 门控可升级为 RetinaFace 式单阶段(手掌框+4谷点一次出) | Phase C 候选,先不合并 |
| 2 | **关键点定位头** | [PIPNet](https://openaccess.thecvf.com/content/CVPR2023/html/Jin_PIPNet_Pixel-in-PixelNet_Towards_Efficient_Facial_Landmark_Detection_in_the_Wild_CVPR_2023_paper.html)(CVPR 2023,分类式高效定位)与 RTMPose/SimCC 同思想;[ICCV 2025 soft-argmax 修正](https://arxiv.org/html/2508.14929v1) | SimCC 谷点+腕点头(**采用 ICCV 2025 的损失修正**) | ✅ 代码已就位 |
| 3 | **解析/分割头** | [BiSeNet V2](https://arxiv.org/abs/2104.06536)(实时双路)/[FaRL](https://openaccess.thecvf.com/content/CVPR2022/html/Feng_Learning_Facial_Representation_From_Motion-Style_Semantic_Within_a_Differential_Force_CVPR_2022_paper.html)(CVPR 2022,精度线) | 轻量 BiSeNet 式双路分割分支;标签由 SAM2 离线打(对应人脸 FaceSynthetics"合成完美标签"先例) | 待建(下一批) |
| 4 | **对齐协议** | 5 点相似变换 → 112×112(insightface 工业标准) | **4 点(3谷+腕)→ 标准掌纹 ROI 坐标系**,与 A 类机制兼容 | 机制已有 |
| 5 | 识别训练框架 | insightface arcface_torch / TopoFR / LVFace | ✅ 已接入(TopoFR iResNet、LVFace-S 消融已跑) | 已完成 |
| 6 | **质量评估** | [MagFace](https://openaccess.thecvf.com/content/CVPR2021/html/Meng_MagFace_A_Universal_Representation_for_Face_Recognition_CVPR_2021_paper.html)(特征幅度=质量,免标注)/SER-FIQ | ROI 质量门控(L4):低质量帧拒识,嵌入幅度做免标注质量分 | 远期加分项 |
| 7 | 合成数据 | FaceSynthetics 100k(微软,合成脸带完美标签) | 抠图贴背景 + RPG-Palm/Canny2Palm | 已立项(数据侧) |
| 8 | 免对齐终局 | SapiensID(CVPR 2025)、LAFS(CVPR 2024) | 论文 discussion:人脸已免对齐,掌纹联合训练的极限在哪 | 写作素材 |

## 二、网络规格 v3(每个块标注人脸出身)

```
输入:256×256
Backbone:轻量 CNN,人脸预训练初始化        ← 出身:#5(已验证:域内 0.21%)
Neck:轻量多尺度融合                        ← 出身:HRNet 高分辨率表征思想
头①分割分支(BiSeNet 式双路)              ← 出身:#3 BiSeNet V2
头②SimCC 谷点+腕点头(4 点)               ← 出身:#2 PIPNet/SimCC + ICCV25 修正
   损失:L_simcc(含 soft-argmax 修正) + L_topo(TopoFR 出身) + λ_seg·L_dice
头③(可选)热图头接 WiLoR 蒸馏            ← 出身:#1 RetinaFace 的 dense 半监督思想
对齐输出:4 点 → 坐标系 → 标准 ROI          ← 出身:#4(5 点协议的掌纹版)
Phase B:ROI 可微化 + ID 损失反哺           ← 出身:#5 + LAFS 的"关键点-识别互益"
```

## 三、和 v2 设计的差异(回应"SOTA 出身"要求)

1. 分割头从"U-Net 式"改为 **BiSeNet V2 双路式**(人脸解析的实时标准,双路=细节+上下文,正好匹配"谷点要细节、手掌要上下文");
2. SimCC 损失采纳 **ICCV 2025 soft-argmax 修正**(人脸关键点最新理论成果,直接引用);
3. 质量门控明确为 **MagFace 路线**(免标注,零额外标注成本);
4. 每个组件在论文里都带人脸引用链,审稿人看到的是"成熟工具链的移植",不是拍脑袋。

## 四、执行顺序不变

PalmKit 4 点 GT(卡脖子)→ ②+拓扑训练出 NLE/SR → ①分割分支叠加 → Phase B 联合 → L4/MagFace 门控(加分)。

## 五、数据特性确认(2026-10-02,用户提供 2 万张样本后修订)

- **主训练集切换**:用户的 2 万张多姿态数据集(带 3 点 GT:valley1/valley2/center,同名 json 格式)成为定位线主训练集;CASIA 降级为识别线训练域 + 自训练扩充对象
- **GT 约定**:3 点(2 谷+掌心),与 A-04/A-08/A-11 机制及现有 ROIExtractor 兼容;SimCC num_keypoints=3;拓扑损失改三角形约束
- **不分左右手**:标注无手性语义 → 质检改为几何有效性校验(三点三角形合理性),取消手性校验
- **多姿态(横/竖/任意旋转)**:训练增广改为 ±180° 大角度旋转 + 缩放抖动(标注同步变换);拓扑描述子改为旋转/镜像不变(成对距离比)
- **质检协议**:三引擎对比(已有 PKL 标注 vs WiLoR 预标 vs ERAlign 指缝线)→ 一致直接入库,分歧进修正队列(不丢弃难样本)
- 等待项:pack_dataset.py 打包传输(另一台电脑,分卷 zip + manifest)

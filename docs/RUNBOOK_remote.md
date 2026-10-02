# 远程执行手册(另一台电脑上的操作序列)

> 循环模式:你在这台电脑执行 → 把指定结果粘贴回对话 → 我更新代码推 GitHub → 你 `git pull` 进入下一步。
> 每步都标了【粘贴什么】。

## Step 0 环境准备(一次性)

```bash
git clone <仓库地址> roi-new
cd roi-new
pip install -r requirements.txt
```

验证环境(合成数据,不需要真实数据,2 分钟):

```bash
python scripts/smoke_keypoint.py
```

【粘贴】最后 5 行输出(应看到 `SMOKE TEST PASSED`)。

## Step 1 GT 体检(不训练,CPU 即可)

```bash
python scripts/quality_check.py --data_root <你的2万数据集路径> --out_dir qc1
```

产出 `qc1/qc_report.md`(几何有效性:三点三角形合理性、边长比、朝向)。

【粘贴】`qc1/qc_report.md` 全文 + `qc1/disagree_list.txt` 的前 30 行。
→ 我判断 GT 质量,决定是否需要修正队列/如何修。

## Step 2 主训练:M2 定位网络(质检通过后)

```bash
python scripts/train_keypoint.py --data_root <你的2万数据集路径> --out_dir runs/m2_v1 --epochs 60
```

- 默认配置:resnet18 backbone / 256×256 输入 / batch 64 / AdamW lr=1e-3
- GPU 上预计几小时(2 万张 × 60 epochs);中断了直接重跑同一命令(每轮都有 best.pt 保存,但注意当前版本不续训,建议一次跑完)

【粘贴】`runs/m2_v1/train_log.csv` 全文(就 60 行)。

## Step 3 评测出报告

```bash
python scripts/eval_keypoint.py --ckpt runs/m2_v1/best.pt --data_root <数据路径> --vis 24
```

【粘贴】`runs/m2_v1/eval/results.md` 全文 + `eval/vis/` 里 2-3 张你肉眼觉得最差的可视化截图描述。

→ 我拿到 NLE/SR 数字后:判断是否达到可入主表的水平、定位误差集中在哪个点、下一步是调参还是上分割分支。

## Step 4 消融(可选,主训练出数后再说)

```bash
python scripts/train_keypoint.py --data_root ... --backbone mobilenet_v3_small --out_dir runs/m2_mobilenet
python scripts/train_keypoint.py --data_root ... --lambda_topo 0 --out_dir runs/m2_no_topo
```

## 注意事项

- 数据约定:每张图同名 `.json`(`valley1/valley2/center`,像素坐标);缺标注的图会被自动跳过——**缺多少个先跑 Step 1 的体检就知道**
- 训练时 Ctrl+C 中断不会丢已存 checkpoint,但续训需手动加 `--epochs` 补齐,当前版本不做断点续训
- Windows 下 num_workers 默认 4,如遇内存不足加 `--num_workers 0`

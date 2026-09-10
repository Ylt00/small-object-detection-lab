# 真实数据基线：COCO8

COCO8 是 Ultralytics 从 COCO train2017 中取出的前 8 张图片，4 张用于训练、4 张用于验证。它包含 80 类 COCO 标签，适合验证真实数据链路，但不能用来证明算法优劣。基线默认微调官方 YOLO11n 预训练权重；如果本机不存在，Ultralytics 会自动下载。

## 1. 准备环境

当前项目已验证环境：Python 3.10、PyTorch CPU、Ultralytics。

```powershell
conda activate dl
$env:PYTHONPATH = "src"
```

## 2. 下载并校验数据

```powershell
python -m sodlab.cli prepare-coco8 --output data/raw
```

命令会：

1. 下载官方 `coco8.zip`；
2. 校验固定 SHA256；
3. 防止 ZIP 路径穿越；
4. 解压到 `data/raw/coco8`；
5. 生成 `data/raw/coco8/data.yaml`。

如果自动下载被网络阻断，可以先用 GitHub CLI 下载，再导入本地压缩包：

```powershell
gh release download v0.0.0 `
  --repo ultralytics/assets `
  --pattern coco8.zip `
  --dir data/downloads

python -m sodlab.cli prepare-coco8 `
  --output data/raw `
  --archive data/downloads/coco8.zip `
  --force
```

## 3. 校验

```powershell
python -m sodlab.cli validate --data data/raw/coco8/data.yaml
```

预期结果：8 张图片、8 个标签文件、训练集 4 张、验证集 4 张，错误数为 0。

## 4. CPU 基线训练

```powershell
python -m sodlab.cli train --config configs/coco8-cpu.yaml
```

训练输出位于 `runs/train/coco8-cpu/`。重点观察：

- `results.csv`：每个 epoch 的 loss 和 mAP；
- `weights/last.pt`：最后一个 epoch 的权重；
- `weights/best.pt`：验证指标最好的权重。

## 5. 评估

```powershell
python -m sodlab.cli evaluate `
  --weights runs/train/coco8-cpu/weights/best.pt `
  --data data/raw/coco8/data.yaml `
  --imgsz 320 `
  --device cpu `
  --project runs/val `
  --name coco8-cpu
```

`runs/val/coco8-cpu/metrics.json` 是机器可读指标，适合后续汇总成论文表格。

## 6. 为什么不能直接写进论文

- 训练集只有 4 张图片，严重过拟合；
- 验证集只有 4 张图片，指标方差极大；
- 数据量不足以比较模型或模块；
- COCO8 的定位是调试和教学。

完成本流程后，下一步应选择至少数百到数千张图片的真实数据集，固定训练、验证、测试划分，再开始方法与消融研究。


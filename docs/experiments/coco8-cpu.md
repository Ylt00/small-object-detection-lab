# COCO8 CPU baseline

日期：2026-09-10

## 目的

验证真实 COCO 图片、YOLO 标签、预训练权重、CPU 训练、评估和推理的完整链路。结果不用于论文比较。

## 环境

- CPU：Intel Core i5-1155G7
- Python：3.10.21
- PyTorch：2.14.0+cpu
- Ultralytics：8.4.140
- 模型：YOLO11n 官方预训练权重
- 输入尺寸：320
- Batch：4
- Epoch：10
- Seed：42

## 数据

- 训练集：4 张
- 验证集：4 张
- 目标总数：30
- 小目标：4
- 中目标：12
- 大目标：14

## 结果

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.592 | 0.650 | 0.679 | 0.434 |

## 复现命令

```powershell
python -m sodlab.cli prepare-coco8 `
  --output data/raw `
  --archive data/downloads/coco8.zip `
  --force

python -m sodlab.cli validate --data data/raw/coco8/data.yaml
python -m sodlab.cli train --config configs/coco8-cpu.yaml

python -m sodlab.cli evaluate `
  --weights runs/train/coco8-cpu/weights/best.pt `
  --data data/raw/coco8/data.yaml `
  --imgsz 320 `
  --device cpu
```

## 结论

链路工作正常，预训练模型可以在 CPU 上快速微调。验证集只有 4 张图片，单类指标波动极大，不能据此判断模型能力。下一阶段需要换取更大的真实数据集。

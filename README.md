# Small Object Detection Lab

[![CI](https://github.com/Ylt00/small-object-detection-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Ylt00/small-object-detection-lab/actions/workflows/ci.yml)

[GitHub 仓库](https://github.com/Ylt00/small-object-detection-lab) · [v0.1.0 Release](https://github.com/Ylt00/small-object-detection-lab/releases/tag/v0.1.0)

一个面向研究生入门与科研复现的“小目标检测”项目模板。目标不是堆积模型代码，而是建立一条可靠的工作流：

**数据准备 → 数据校验 → 基线训练 → 指标评估 → 消融实验 → GitHub 发布**

> 当前阶段：`v0.1` 工程骨架与无网络烟雾测试。真实数据集和研究改进将在后续迭代中接入。

## 为什么从小目标检测开始

小目标检测有明确的研究问题、丰富的公开数据集和可解释的改进方向，适合形成连续的论文与开源项目。典型挑战包括：

- 目标像素少，经过下采样后特征容易消失；
- 多尺度变化大，固定特征层难以兼顾；
- 背景干扰强，密集目标容易被漏检；
- 标注成本高，数据分布容易长尾。

## 当前功能

- 可复现地生成 YOLO 格式的合成目标检测数据；
- 校验图像、标签、类别 ID 和归一化边界框；
- 统计小、中、大目标的分布；
- 使用 Ultralytics YOLO 运行 CPU 烟雾训练；
- 在测试集上评估并导出指标；
- 对图片、目录或视频运行推理；
- 通过 GitHub Actions 自动执行单元测试和数据校验。

## 快速开始

推荐使用已经验证过的 Conda 环境 `dl`：

```powershell
conda activate dl
$env:PYTHONPATH = "src"

python -m sodlab.cli generate --output data/synthetic
python -m sodlab.cli validate --data data/synthetic/data.yaml
python -m sodlab.cli train --config configs/smoke.yaml
```

训练结果写入 `runs/train/smoke/`。烟雾训练只用于确认代码路径正确，不代表真实检测精度。

评估最后一次训练得到的权重：

```powershell
python -m sodlab.cli evaluate `
  --weights runs/train/smoke/weights/last.pt `
  --data data/synthetic/data.yaml `
  --device cpu `
  --name smoke-eval
```

推理：

```powershell
python -m sodlab.cli predict `
  --weights runs/train/smoke/weights/last.pt `
  --source data/synthetic/images/test `
  --name smoke-predict
```

也可以一次执行完整冒烟流程：

```powershell
.\scripts\run_smoke.ps1
```

如果 `dl` 环境路径不同，可通过 `-Python` 或 `SODLAB_PYTHON` 指定解释器。

也可以正式安装本项目：

```powershell
python -m pip install -e ".[train]"
sodlab --help
```

## 项目结构

```text
small-object-detection-lab/
├─ configs/                 # 可复现实验配置
├─ docs/                    # 学习路线、GitHub 指南、研究计划
├─ experiments/             # 保留实验索引，不提交大文件
├─ scripts/                 # 一键执行脚本
├─ src/sodlab/              # Python 包
├─ tests/                   # 标准库 unittest
└─ .github/workflows/       # 持续集成
```

## 数据约定

项目统一使用 Ultralytics YOLO 检测数据格式。每张图片对应一个同名 `.txt` 标签文件：

```text
class_id center_x center_y width height
```

其中四个坐标均为相对于图像宽高的 `0~1` 浮点数。原始数据、处理后数据和模型权重不提交到 Git；使用数据准备脚本和配置文件保证可复现性。

## 学习与开发顺序

1. 阅读 `docs/learning-plan.md`，按周完成任务；
2. 按 `docs/github-first-publish.md` 将仓库发布到 GitHub；
3. 每完成一个实验，在 `docs/research-log.md` 记录配置、现象和结论；
4. 接入真实数据后，先复现基线，再做单一变量的消融实验。

## 发布代码时的边界

GitHub 适合发布代码、配置、文档和小型可复现结果。数据集、私有数据、模型大文件、账号令牌和导师未授权的代码不应被提交。学术论文还需要与导师确定研究问题、数据合规性、评价协议和投稿目标。

## License

MIT



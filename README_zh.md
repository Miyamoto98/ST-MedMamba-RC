[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

# ST-MedMamba-RC: 基于 MedMamba 的直肠癌分期

## 项目概览

本项目旨在利用深度学习模型对直肠癌进行分期，数据来源于 T2 加权 (T2W) 和弥散加权 (DWI) 磁共振成像 (MRI) 扫描。模型采用了基于 Vision Mamba 的架构，集成了基于 MedSAM2 的图像编码器、自定义的记忆模块 (Memory Blocks) 和 MambaFusion 模块，用于鲁棒的特征提取与融合，最终通过分类头进行分期。

## 项目结构

本项目的主要目录和文件如下：

```
ST-MedMamba-RC/
├── checkpoints/
│   └── MedSAM2_latest.pt  # 预训练的 MedSAM2 图像编码器权重
│   └── best_model.pth     # 微调后的模型检查点 (训练后生成)
├── data/                  # 包含病人数据和 labels.csv
│   ├── patient_001/
│   │   ├── patient_001_t2w.nii.gz
│   │   └── patient_001_dwi.nii.gz
│   ├── ...
│   └── labels.csv
├── model/                 # 模型定义 (RectalCancerStagingModel, MambaFusion, MemoryBlock)
├── sam2/                  # MedSAM2 相关模块 (ImageEncoder, etc.)
├── utils/                 # 工具函数 (预处理, 辅助脚本)
├── inference.py           # 用于对单个病人进行推理的脚本
├── train.py               # 模型训练脚本
├── demo_train.py          # 修改后的训练脚本，用于演示/测试目的
├── requirements.txt       # Python 依赖项
├── README.md              # 本英文版 README 文件
└── README_zh.md           # 本中文版 README 文件
```

## 设置与安装

### 前提条件

*   Python 3.8+
*   `pip` (Python 包管理器)

### 克隆仓库

```bash
git clone https://github.com/your_repo/ST-MedMamba-RC.git
cd ST-MedMamba-RC
```

### 安装依赖

使用 `pip` 安装所有必需的 Python 包：

```bash
pip install -r requirements.txt
```

### 预训练检查点

从 [Hugging Face](https://huggingface.co/wanglab/MedSAM2) 下载预训练的 `MedSAM2_latest.pt` 图像编码器权重，并将其放置在 `checkpoints/` 目录下。此检查点作为模型中图像编码器的初始权重。

## 数据集准备

本项目期望的数据集结构是：每个病人的 MRI 扫描（T2W 和 DWI）组织在其各自的子目录中，并且标签在 `labels.csv` 文件中提供。

### 目录结构

请按以下方式组织您的病人数据：

```
data/
├── patient_001/
│   ├── patient_001_t2w.nii.gz
│   └── patient_001_dwi.nii.gz
├── patient_002/
│   ├── patient_002_t2w.nii.gz
│   └── patient_002_dwi.nii.gz
├── ...
└── labels.csv
```

*   **`data/`**：所有病人数据的根目录。
*   **`patient_XXX/`**：每个子目录应以唯一的 `patient_id` 命名（例如 `patient_001`，`patient_002`）。
*   **影像文件**：在每个 `patient_XXX` 文件夹内，放置 T2W 和 DWI 的 NIfTI 文件。文件名应分别包含 `t2w` (或 `T2W`, `se1`) 和 `dwi` (或 `DWI`, `se7`) 关键字，并以 `.nii.gz` 结尾。例如：`patient_001_t2w.nii.gz` 和 `patient_001_dwi.nii.gz`。

### `labels.csv` 格式

在 `data/` 目录下直接创建一个 `labels.csv` 文件。此文件应包含两列：`patient_id` 和 `label`。`patient_id` 必须与您的病人子目录的名称完全匹配。

`labels.csv` 示例：

```csv
patient_id,label
patient_001,1
patient_002,3
patient_003,0
...
patient_N,2
```

## 模型训练

要训练模型，请运行 `train.py` 脚本。脚本将自动从 `data/` 目录加载数据，将其分割为训练集和验证集，并保存表现最佳的模型检查点。

```bash
python train.py [OPTIONS]
```

### 主要训练选项

*   `--data_root`：数据集的根目录 (默认值: `data/`)。
*   `--labels_csv`：包含标签的 CSV 文件路径 (默认值: `data/labels.csv`)。
*   `--epochs`：训练的 epoch 数量 (默认值: `50`)。
*   `--batch_size`：训练的批次大小 (默认值: `1`)。
*   `--lr`：学习率 (默认值: `1e-4`)。
*   `--val_split`：用于验证的数据比例 (默认值: `0.2`)。
*   `--patience`：早停的耐心值 (默认值: `5`)。如果验证损失在此数量的 epoch 内没有改善，训练将停止。
*   `--pretrained_checkpoint`：用于微调的预训练检查点路径 (默认值: `checkpoints/MedSAM2_latest.pt`)。

**示例：**

```bash
python train.py --epochs 100 --batch_size 2 --lr 5e-5 --patience 10
```

表现最佳的模型（基于验证损失）将保存为 `checkpoints/best_model.pth`。

## 运行推理

要对特定病人运行推理，请使用 `inference.py` 脚本并提供 `--patient_id` 参数。

```bash
python inference.py --patient_id <您的病人ID> [OPTIONS]
```

### 主要推理选项

*   `--patient_id`：**必需。** 要进行推理的病人ID (例如 `patient_001`)。
*   `--data_root`：数据集的根目录 (默认值: `data/`)。
*   `--checkpoint`：模型检查点文件路径 (默认值: `checkpoints/best_model.pth`)。
*   `--slice_batch_size`：模型内处理 3D 切片的批次大小 (默认值: `1`)。

**示例：**

```bash
python inference.py --patient_id patient_001
python inference.py --patient_id patient_005 --checkpoint checkpoints/my_custom_model.pth
```

脚本将打印预测的类别以及输入最后一个批次的 Softmax 概率。

## 许可证

本项目采用 [知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh)。详情请参阅 [LICENSE](LICENSE) 文件。
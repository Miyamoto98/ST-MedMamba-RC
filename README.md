[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

# ST-MedMamba-RC: Rectal Cancer Staging with MedMamba

## Overview

This project implements a deep learning model for rectal cancer staging using T2-weighted (T2W) and Diffusion-weighted (DWI) Magnetic Resonance Imaging (MRI) scans. The model leverages a Vision Mamba-based architecture, incorporating MedSAM2-based image encoders, custom Memory Blocks, and a MambaFusion module for robust feature extraction and fusion, followed by a classification head for staging.

## Project Structure

The key directories and files in this project are:

```
ST-MedMamba-RC/
├── checkpoints/
│   └── MedSAM2_latest.pt  # Pre-trained MedSAM2 image encoder weights
│   └── best_model.pth     # Fine-tuned model checkpoint (generated after training)
├── data/                  # Contains patient data and labels.csv
│   ├── patient_001/
│   │   ├── patient_001_t2w.nii.gz
│   │   └── patient_001_dwi.nii.gz
│   ├── ...
│   └── labels.csv
├── model/                 # Model definitions (RectalCancerStagingModel, MambaFusion, MemoryBlock)
├── sam2/                  # MedSAM2 related modules (ImageEncoder, etc.)
├── utils/                 # Utility functions (preprocessing, auxiliary scripts)
├── inference.py           # Script for running inference on a single patient
├── train.py               # Script for training the model
├── demo_train.py          # Modified training script for demonstration/testing purposes
├── requirements.txt       # Python dependencies
├── README.md              # This English README file
└── README_zh.md           # Chinese README file
```

## Setup & Installation

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Cloning the Repository

```bash
git clone https://github.com/your_repo/ST-MedMamba-RC.git
cd ST-MedMamba-RC
```

### Installing Dependencies

Install all required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### Pre-trained Checkpoint

Download the pre-trained `MedSAM2_latest.pt` image encoder weights from [Hugging Face](https://huggingface.co/wanglab/MedSAM2) and place it in the `checkpoints/` directory. This checkpoint serves as the initial weights for the image encoders in our model.

## Dataset Preparation

The project expects a structured dataset where each patient's MRI scans (T2W and DWI) are organized into their own subdirectory, and labels are provided in a `labels.csv` file.

### Directory Structure

Organize your patient data as follows:

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

*   **`data/`**: The root directory for all your patient data.
*   **`patient_XXX/`**: Each subdirectory should be named with a unique `patient_id` (e.g., `patient_001`, `patient_002`).
*   **Image Files**: Inside each `patient_XXX` folder, place the T2W and DWI NIfTI files. The filenames should contain `t2w` (or `T2W`, `se1`) and `dwi` (or `DWI`, `se7`) respectively, and end with `.nii.gz`. For example: `patient_001_t2w.nii.gz` and `patient_001_dwi.nii.gz`.

### `labels.csv` Format

Create a `labels.csv` file directly under the `data/` directory. This file should contain two columns: `patient_id` and `label`. The `patient_id` must exactly match the names of your patient subdirectories.

Example `labels.csv`:

```csv
patient_id,label
patient_001,1
patient_002,3
patient_003,0
...
patient_N,2
```

## Training the Model

To train the model, run the `train.py` script. The script will automatically load data from the `data/` directory, split it into training and validation sets, and save the best performing model checkpoint.

```bash
python train.py [OPTIONS]
```

### Key Training Options

*   `--data_root`: Root directory of the dataset (default: `data/`).
*   `--labels_csv`: Path to the CSV file with labels (default: `data/labels.csv`).
*   `--epochs`: Number of training epochs (default: `50`).
*   `--batch_size`: Batch size for training (default: `1`).
*   `--lr`: Learning rate (default: `1e-4`).
*   `--val_split`: Fraction of data to use for validation (default: `0.2`).
*   `--patience`: Patience for early stopping (default: `5`). If validation loss does not improve for this many epochs, training stops.
*   `--pretrained_checkpoint`: Path to a pre-trained checkpoint for fine-tuning (default: `checkpoints/MedSAM2_latest.pt`).

**Example:**

```bash
python train.py --epochs 100 --batch_size 2 --lr 5e-5 --patience 10
```

The best model (based on validation loss) will be saved as `checkpoints/best_model.pth`.

## Running Inference

To run inference on a specific patient, use the `inference.py` script and provide the `--patient_id` argument.

```bash
python inference.py --patient_id <YOUR_PATIENT_ID> [OPTIONS]
```

### Key Inference Options

*   `--patient_id`: **Required.** ID of the patient to run inference on (e.g., `patient_001`).
*   `--data_root`: Root directory of the dataset (default: `data/`).
*   `--checkpoint`: Path to the model checkpoint file (default: `checkpoints/best_model.pth`).
*   `--slice_batch_size`: Batch size for processing 3D slices within the model (default: `1`).

**Example:**

```bash
python inference.py --patient_id patient_001
python inference.py --patient_id patient_005 --checkpoint checkpoints/my_custom_model.pth
```

The script will print the predicted class and the softmax probabilities for the last batch of the input.

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License](https://creativecommons.org/licenses/by-nc-sa/4.0/). See the [LICENSE](LICENSE) file for details.

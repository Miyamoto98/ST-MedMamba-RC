
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import os
import argparse
import glob

# To ensure all modules are importable, add the project root to PYTHONPATH
# For example, run `export PYTHONPATH=$PYTHONPATH:/path/to/MedSAM2-main`
# or install the project as a package.

from model.model import RectalCancerStagingModel
from utils.preprocessing import preprocess_patient_mri

def run_inference():
    # --- Configuration ---
    num_classes = 4
    d_model = 256
    img_size = 64
    slice_depth = 64
    target_size = (slice_depth, img_size, img_size)
    
    parser = argparse.ArgumentParser(description='Run inference with a MedSAM2 model.')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='Path to the checkpoint file')
    parser.add_argument('--patient_id', type=str, required=True, help='ID of the patient to run inference on (e.g., patient_001)')
    parser.add_argument('--data_root', type=str, default='data', help='Root directory of the dataset')
    parser.add_argument('--slice_batch_size', type=int, default=1, help='Batch size for processing 3D slices')
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    slice_batch_size = args.slice_batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model Initialization ---
    print("Initializing model...")
    model = RectalCancerStagingModel(num_classes=num_classes, d_model=d_model)
    model.to(device)

    # --- Load Checkpoint ---
    print(f"Loading model checkpoint from {checkpoint_path}...")
    try:
        # Load the state dictionary from the fine-tuned model
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        # Load the state dict into the model
        model.load_state_dict(state_dict)
        
        print("Fine-tuned model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        print("Please ensure the checkpoint is correctly placed in the 'checkpoints' folder or provide the correct path using --checkpoint.")
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval() # Set model to evaluation mode

    # --- Load NIfTI Data ---
    # --- Find and Load NIfTI Data ---
    patient_dir = os.path.join(args.data_root, args.patient_id)
    print(f"Searching for patient data in: {patient_dir}")

    if not os.path.isdir(patient_dir):
        print(f"Error: Patient directory not found at {patient_dir}")
        return

    try:
        # Use glob to find files, making it robust to exact naming
        t2w_files = glob.glob(os.path.join(patient_dir, '*t2w*.nii.gz')) + glob.glob(os.path.join(patient_dir, '*T2W*.nii.gz')) + glob.glob(os.path.join(patient_dir, '*se1*.nii.gz'))
        dwi_files = glob.glob(os.path.join(patient_dir, '*dwi*.nii.gz')) + glob.glob(os.path.join(patient_dir, '*DWI*.nii.gz')) + glob.glob(os.path.join(patient_dir, '*se7*.nii.gz'))

        if not t2w_files:
            print(f"Error: No T2W file found in {patient_dir}")
            return
        if not dwi_files:
            print(f"Error: No DWI file found in {patient_dir}")
            return

        t2w_nifti_path = t2w_files[0]
        dwi_nifti_path = dwi_files[0]

        print(f"Loading T2W NIfTI file from {t2w_nifti_path}...")
        t2w_nifti_img = nib.load(t2w_nifti_path)
        t2w_volume_raw = t2w_nifti_img.get_fdata().astype(np.float32)
        print(f"T2W NIfTI file loaded. Original shape: {t2w_volume_raw.shape}")

        print(f"Loading DWI NIfTI file from {dwi_nifti_path}...")
        dwi_nifti_img = nib.load(dwi_nifti_path)
        dwi_volume_raw = dwi_nifti_img.get_fdata().astype(np.float32)
        print(f"DWI NIfTI file loaded. Original shape: {dwi_volume_raw.shape}")

    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return

    # --- Preprocessing ---
    print("Starting preprocessing...")
    preprocessed_t2w, preprocessed_dwi = preprocess_patient_mri(
        t2w_volume_raw, dwi_volume_raw, target_size=target_size
    )
    print(f"Preprocessing complete. T2W shape: {preprocessed_t2w.shape}, DWI shape: {preprocessed_dwi.shape}")

    # --- Convert to PyTorch Tensors and add batch/channel dims ---
    t2w_batch = torch.from_numpy(preprocessed_t2w).unsqueeze(0).unsqueeze(0) # Add B and C
    dwi_batch = torch.from_numpy(preprocessed_dwi).unsqueeze(0).unsqueeze(0) # Add B and C
    
    t2w_batch = t2w_batch.to(device)
    dwi_batch = dwi_batch.to(device)

    # --- Inference ---
    print("Performing inference...")
    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = model(t2w_batch, dwi_batch, slice_batch_size=slice_batch_size)
    
    # Get predicted class
    predicted_class = torch.argmax(outputs, dim=1).item()

    print(f"Inference successful! Output logits: {outputs.cpu().numpy()}")
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    try:
        run_inference()
    except ImportError as e:
        print(f"\nError: {e}")
        print("Please ensure that the project root (containing 'sam2' and 'model' folders) is in your PYTHONPATH.")
        print("You can add it by running: export PYTHONPATH=$PYTHONPATH:/path/to/MedSAM2-main")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

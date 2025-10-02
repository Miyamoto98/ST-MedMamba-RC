
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import os
import argparse

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
    parser.add_argument('--checkpoint', type=str, default='checkpoints/MedSAM2_latest.pt', help='Path to the checkpoint file')
    parser.add_argument('--t2w_path', type=str, default='data\Ⅲ\SE1.nii.gz', help='Path to the T2W NIfTI file')
    parser.add_argument('--dwi_path', type=str, default='data\Ⅲ\SE7.nii.gz', help='Path to the DWI NIfTI file')
    parser.add_argument('--slice_batch_size', type=int, default=1, help='Batch size for processing 3D slices')
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    t2w_nifti_path = args.t2w_path
    dwi_nifti_path = args.dwi_path
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
        full_checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # If the state_dict is nested under a 'model' key, extract it
        if isinstance(full_checkpoint, dict) and 'model' in full_checkpoint:
            full_checkpoint = full_checkpoint['model']

        # Extract image_encoder state_dict
        image_encoder_state_dict = {}
        for key, value in full_checkpoint.items():
            if key.startswith('image_encoder.'):
                image_encoder_state_dict[key.replace('image_encoder.', '')] = value
        
        # Load image_encoder weights into t2w_encoder and dwi_encoder
        # Use strict=False because the ImageEncoder might not have all keys from the full checkpoint's image_encoder
        model.t2w_encoder.load_state_dict(image_encoder_state_dict, strict=False)
        model.dwi_encoder.load_state_dict(image_encoder_state_dict, strict=False)
        
        print("Image encoder weights loaded successfully into t2w_encoder and dwi_encoder.")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        print("Please ensure the checkpoint is correctly placed in the 'checkpoints' folder.")
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval() # Set model to evaluation mode

    # --- Load NIfTI Data ---
    print(f"Loading T2W NIfTI file from {t2w_nifti_path}...")
    try:
        t2w_nifti_img = nib.load(t2w_nifti_path)
        t2w_volume_raw = t2w_nifti_img.get_fdata().astype(np.float32)
        print(f"T2W NIfTI file loaded. Original shape: {t2w_volume_raw.shape}")
    except FileNotFoundError:
        print(f"Error: T2W NIfTI file not found at {t2w_nifti_path}")
        return
    except Exception as e:
        print(f"Error loading T2W NIfTI file: {e}")
        return

    print(f"Loading DWI NIfTI file from {dwi_nifti_path}...")
    try:
        dwi_nifti_img = nib.load(dwi_nifti_path)
        dwi_volume_raw = dwi_nifti_img.get_fdata().astype(np.float32)
        print(f"DWI NIfTI file loaded. Original shape: {dwi_volume_raw.shape}")
    except FileNotFoundError:
        print(f"Error: DWI NIfTI file not found at {dwi_nifti_path}")
        return
    except Exception as e:
        print(f"Error loading DWI NIfTI file: {e}")
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

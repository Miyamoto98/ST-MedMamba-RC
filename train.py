import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nibabel as nib
import os
import tqdm
import argparse

# To ensure all modules are importable, add the project root to PYTHONPATH
# For example, run `export PYTHONPATH=$PYTHONPATH:/path/to/MedSAM2-main`
# or install the project as a package.

from model.model import RectalCancerStagingModel
from utils.preprocessing import preprocess_patient_mri

def train_model():
    # --- Configuration ---
    num_classes = 4
    d_model = 256
    
    parser = argparse.ArgumentParser(description='Train a MedSAM2-based rectal cancer staging model.')
    parser.add_argument('--img_size', type=int, default=64, help='Image size for preprocessing (H, W)')
    parser.add_argument('--slice_depth', type=int, default=64, help='Slice depth for preprocessing (D)')
    parser.add_argument('--t2w_path', type=str, default='data\\Ⅲ\\SE1.nii.gz', help='Path to the T2W NIfTI file')
    parser.add_argument('--dwi_path', type=str, default='data\\Ⅲ\\SE7.nii.gz', help='Path to the DWI NIfTI file')
    parser.add_argument('--slice_batch_size', type=int, default=1, help='Batch size for processing 3D slices within the model')
    parser.add_argument('--pretrained_checkpoint', type=str, default='checkpoints/MedSAM2_latest.pt', help='Path to a pre-trained checkpoint for fine-tuning')
    args = parser.parse_args()

    img_size = args.img_size
    slice_depth = args.slice_depth
    target_size = (slice_depth, img_size, img_size)
    
    t2w_nifti_path = args.t2w_path
    dwi_nifti_path = args.dwi_path
    slice_batch_size = args.slice_batch_size
    pretrained_checkpoint_path = args.pretrained_checkpoint
    
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "model_checkpoint.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    # Model expects (B, C, D, H, W). Our preprocessed data is (D, H, W).
    t2w_batch = torch.from_numpy(preprocessed_t2w).unsqueeze(0).unsqueeze(0) # Add B and C
    dwi_batch = torch.from_numpy(preprocessed_dwi).unsqueeze(0).unsqueeze(0) # Add B and C
    
    # Dummy label for a single patient, assuming a classification task with num_classes=4
    # We'll use label 0 for demonstration.
    labels = torch.tensor([0]).to(device) 

    # --- Model Initialization ---
    print("Initializing model...")
    model = RectalCancerStagingModel(num_classes=num_classes, d_model=d_model)
    model.to(device)

    # --- Load Pre-trained Checkpoint (for fine-tuning) ---
    if pretrained_checkpoint_path:
        print(f"Loading pre-trained checkpoint from {pretrained_checkpoint_path} for fine-tuning...")
        try:
            full_checkpoint = torch.load(pretrained_checkpoint_path, map_location=device)
            
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
            
            print("Pre-trained image encoder weights loaded successfully into t2w_encoder and dwi_encoder.")
        except FileNotFoundError:
            print(f"Error: Pre-trained checkpoint file not found at {pretrained_checkpoint_path}")
            print("Please ensure the checkpoint is correctly placed in the 'checkpoints' folder.")
            return
        except Exception as e:
            print(f"Error loading pre-trained checkpoint: {e}")
            return

    # --- Loss Function and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Training Loop (Single Epoch for demonstration) ---
    print("Starting training (single epoch for verification)...")
    model.train() # Set model to training mode
    
    # Wrap the single training step in tqdm for progress indication
    for _ in tqdm.tqdm(range(1), desc="Training Epoch"): # Single iteration for demonstration
        t2w_batch = t2w_batch.to(device)
        dwi_batch = dwi_batch.to(device)

        optimizer.zero_grad()
        outputs = model(t2w_batch, dwi_batch, slice_batch_size=slice_batch_size)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Training complete. Loss: {loss.item():.4f}")

    # --- Save Checkpoint ---
    print(f"Saving model checkpoint to {checkpoint_path}...")
    torch.save(model.state_dict(), checkpoint_path)
    print("Checkpoint saved successfully.")

if __name__ == "__main__":
    try:
        train_model()
    except ImportError as e:
        print(f"\nError: {e}")
        print("Please ensure that the project root (containing 'sam2' and 'model' folders) is in your PYTHONPATH.")
        print("You can add it by running: export PYTHONPATH=$PYTHONPATH:/path/to/MedSAM2-main")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import nibabel as nib
import os
import tqdm
import argparse
import glob
import pandas as pd

from model.model import RectalCancerStagingModel
from utils.preprocessing import preprocess_patient_mri

class PatientVolumeDataset(Dataset):
    """
    PyTorch Dataset for loading patient MRI volumes (T2W and DWI) from a directory structure.
    It reads labels from a provided CSV file.
    """
    def __init__(self, data_root, labels_csv, target_size):
        self.data_root = data_root
        self.target_size = target_size
        
        # --- Load Labels from CSV ---
        try:
            labels_df = pd.read_csv(labels_csv)
            self.labels_map = pd.Series(labels_df.label.values, index=labels_df.patient_id).to_dict()
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Labels CSV file not found at {labels_csv}")

        # --- Find data files and align with labels ---
        self.file_pairs = []
        self.labels = []
        
        all_patient_dirs = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])

        for patient_id in all_patient_dirs:
            if patient_id not in self.labels_map:
                print(f"Warning: Patient ID {patient_id} from directory is not in labels.csv. Skipping.")
                continue

            patient_dir = os.path.join(data_root, patient_id)
            t2w_files = glob.glob(os.path.join(patient_dir, '*t2w*.nii.gz')) + glob.glob(os.path.join(patient_dir, '*T2W*.nii.gz')) + glob.glob(os.path.join(patient_dir, '*se1*.nii.gz'))
            dwi_files = glob.glob(os.path.join(patient_dir, '*dwi*.nii.gz')) + glob.glob(os.path.join(patient_dir, '*DWI*.nii.gz')) + glob.glob(os.path.join(patient_dir, '*se7*.nii.gz'))
            
            if t2w_files and dwi_files:
                self.file_pairs.append((t2w_files[0], dwi_files[0]))
                self.labels.append(self.labels_map[patient_id])
            else:
                print(f"Warning: Could not find T2W or DWI file in {patient_dir}. Skipping patient {patient_id}.")

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        t2w_path, dwi_path = self.file_pairs[idx]
        
        # Load NIfTI images
        t2w_img = nib.load(t2w_path).get_fdata().astype(np.float32)
        dwi_img = nib.load(dwi_path).get_fdata().astype(np.float32)
        
        # Preprocess
        preprocessed_t2w, preprocessed_dwi = preprocess_patient_mri(
            t2w_img, dwi_img, target_size=self.target_size
        )
        
        # Convert to PyTorch Tensors and add channel dim
        t2w_tensor = torch.from_numpy(preprocessed_t2w).unsqueeze(0) # (1, D, H, W)
        dwi_tensor = torch.from_numpy(preprocessed_dwi).unsqueeze(0) # (1, D, H, W)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return t2w_tensor, dwi_tensor, label

def train_model():
    # --- Configuration & Argument Parsing ---
    parser = argparse.ArgumentParser(description='Train a MedSAM2-based rectal cancer staging model.')
    parser.add_argument('--data_root', type=str, default='data/', help='Root directory of the dataset')
    parser.add_argument('--labels_csv', type=str, default='data/labels.csv', help='Path to the CSV file with labels')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training (number of patients)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of data to use for validation')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of staging classes')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--img_size', type=int, default=32, help='Image size for preprocessing (H, W)')
    parser.add_argument('--slice_depth', type=int, default=32, help='Slice depth for preprocessing (D)')
    parser.add_argument('--slice_batch_size', type=int, default=1, help='Batch size for processing 3D slices within the model')
    parser.add_argument('--pretrained_checkpoint', type=str, default='checkpoints/MedSAM2_latest.pt', help='Path to a pre-trained checkpoint for fine-tuning')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    args = parser.parse_args()

    target_size = (args.slice_depth, args.img_size, args.img_size)
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset and DataLoader ---
    print("Setting up dataset...")
    try:
        dataset = PatientVolumeDataset(data_root=args.data_root, labels_csv=args.labels_csv, target_size=target_size)
    except Exception as e:
        print(e)
        return

    if len(dataset) == 0:
        print("No data loaded. Please check the --data_root and --labels_csv paths and their contents.")
        return

    # Handle case with very few samples where val_split might result in 0
    if len(dataset) < 2:
        train_dataset = dataset
        val_dataset = None
        train_size = len(dataset)
        val_size = 0
    else:
        val_size = int(len(dataset) * args.val_split)
        if val_size == 0: # Ensure at least one validation sample if possible
            val_size = 1 
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        val_loader = None
    print(f"Dataset loaded: {train_size} training samples, {val_size} validation samples.")

    # --- Model Initialization ---
    print("Initializing model...")
    model = RectalCancerStagingModel(num_classes=args.num_classes, d_model=args.d_model)
    model.to(device)

    # --- Load Pre-trained Checkpoint (for fine-tuning) ---
    if args.pretrained_checkpoint and os.path.exists(args.pretrained_checkpoint):
        print(f"Loading pre-trained checkpoint from {args.pretrained_checkpoint} for fine-tuning...")
        try:
            full_checkpoint = torch.load(args.pretrained_checkpoint, map_location=device)
            if isinstance(full_checkpoint, dict) and 'model' in full_checkpoint:
                full_checkpoint = full_checkpoint['model']

            image_encoder_state_dict = {k.replace('image_encoder.', ''): v for k, v in full_checkpoint.items() if k.startswith('image_encoder.')}
            
            model.t2w_encoder.load_state_dict(image_encoder_state_dict, strict=False)
            model.dwi_encoder.load_state_dict(image_encoder_state_dict, strict=False)
            print("Pre-trained image encoder weights loaded successfully.")
        except Exception as e:
            print(f"Error loading pre-trained checkpoint: {e}. Training from scratch.")
    else:
        print("No pre-trained checkpoint found or specified. Training from scratch.")

    # --- Loss Function and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    print("Starting training...")

    for epoch in range(args.epochs):
        # -- Training Phase --
        model.train()
        train_loss = 0.0
        train_loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for t2w_batch, dwi_batch, labels in train_loop:
            t2w_batch, dwi_batch, labels = t2w_batch.to(device), dwi_batch.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(t2w_batch, dwi_batch, slice_batch_size=args.slice_batch_size)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # -- Validation Phase --
        if not val_loader:
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, No validation set to evaluate.")
            continue

        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        last_batch_outputs = None
        val_loop = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")

        with torch.no_grad():
            for t2w_batch, dwi_batch, labels in val_loop:
                t2w_batch, dwi_batch, labels = t2w_batch.to(device), dwi_batch.to(device), labels.to(device)
                
                outputs = model(t2w_batch, dwi_batch, slice_batch_size=args.slice_batch_size)
                last_batch_outputs = outputs
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                val_loop.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        if last_batch_outputs is not None:
            # Calculate softmax probabilities for the last batch
            last_batch_probs = torch.softmax(last_batch_outputs, dim=1) * 100
            # Round to 2 decimal places for cleaner printing
            last_batch_probs_rounded = torch.round(last_batch_probs * 100) / 100
            print(f"  └─ Last Batch Probabilities (%): {last_batch_probs_rounded.cpu().numpy()}")

        # -- Save Best Model & Early Stopping --
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path} with validation loss: {best_val_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {args.patience} epochs with no improvement.")
                break

    print("Training complete.")
    print(f"Best model saved at {best_model_path}")

if __name__ == "__main__":
    try:
        train_model()
    except ImportError as e:
        print(f"\nError: {e}")
        print("Please ensure that the project root is in your PYTHONPATH.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

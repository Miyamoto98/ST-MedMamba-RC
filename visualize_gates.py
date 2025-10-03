"""
This script provides an ADVANCED visualization for the fusion gate weights
from the BGAMF module, suitable for publication figures.

It performs the following steps:
1.  Creates a synthetic grayscale MRI image to serve as a background.
2.  Initializes the BGAMF model and creates dummy input tensors.
3.  Runs a forward pass to obtain the low-resolution (e.g., 16x16) gate weights.
4.  Upsamples the gate map to the original MRI image size (e.g., 256x256) using
    bilinear interpolation.
5.  Overlays the upsampled gate heatmap with transparency onto the synthetic MRI image.
6.  Saves the resulting visualization to 'gate_overlay_visualization.png'.

This visualization intuitively shows how the model prioritizes different modalities
over specific anatomical structures.
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

# Ensure the model module can be imported
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.mamba_fusion import BGAMF

def create_synthetic_mri(size: int = 256) -> np.ndarray:
    """
    Generates a simple synthetic MRI slice with geometric shapes.
    """
    print(f"Creating a synthetic MRI image of size {size}x{size}...")
    image = np.zeros((size, size), dtype=np.float32)
    
    # Draw a large circle for a brain-like structure
    cx, cy = size // 2, size // 2
    r = size // 2 - 10
    y, x = np.ogrid[-cx:size-cx, -cy:size-cy]
    mask = x*x + y*y <= r*r
    image[mask] = 0.3

    # Draw smaller, brighter circles to simulate lesions/tumors
    lesion1_y, lesion1_x, lesion1_r = size // 4, size // 4, size // 16
    mask1 = (x - (lesion1_x - cx))**2 + (y - (lesion1_y - cy))**2 <= lesion1_r**2
    image[mask1] = 0.9

    lesion2_y, lesion2_x, lesion2_r = size // 2, int(size * 0.75), size // 12
    mask2 = (x - (lesion2_x - cx))**2 + (y - (lesion2_y - cy))**2 <= lesion2_r**2
    image[mask2] = 0.7
    
    return image

def generate_and_save_gate_overlay(
    d_model: int = 128,
    feature_map_size: int = 16,
    mri_size: int = 256,
    dropout: float = 0.1,
    save_path: str = "gate_overlay_visualization.png"
):
    """
    Generates and saves a visualization of the fusion gate overlaid on a synthetic MRI.
    """
    seq_len = feature_map_size * feature_map_size
    
    # 1. Create a synthetic MRI background
    mri_image = create_synthetic_mri(mri_size)

    # 2. Initialize the BGAMF model
    print("Initializing BGAMF model...")
    model = BGAMF(d_model=d_model, dropout=dropout)
    model.eval()

    # 3. Create dummy input tensors and get the gate
    print(f"Creating dummy input tensors of shape (1, {seq_len}, {d_model})...")
    f_t2w = torch.randn(1, seq_len, d_model)
    f_dwi = torch.randn(1, seq_len, d_model) * 0.5 + 0.1
    
    print("Running forward pass to get the gate weights...")
    with torch.no_grad():
        _, gate = model(f_t2w, f_dwi, return_gate=True)

    # 4. Process and upsample the gate
    # Gate shape: (B, L, C) -> (1, 256, 128)
    gate_map = gate[0].mean(dim=-1) # Shape: (256,)
    gate_map_2d = gate_map.reshape(1, 1, feature_map_size, feature_map_size) # Shape: (1, 1, 16, 16)
    
    print(f"Upsampling gate map from {feature_map_size}x{feature_map_size} to {mri_size}x{mri_size}...")
    gate_map_upsampled = F.interpolate(gate_map_2d, size=(mri_size, mri_size), mode='bilinear', align_corners=False)
    gate_map_upsampled = gate_map_upsampled.squeeze().cpu().numpy() # Shape: (256, 256)

    # 5. Generate and save the overlay plot
    print(f"Generating and saving overlay visualization to '{save_path}'...")
    plt.figure(figsize=(10, 10))
    
    # Plot the base MRI image in grayscale
    plt.imshow(mri_image, cmap='gray')
    
    # Overlay the heatmap with transparency
    plt.imshow(gate_map_upsampled, cmap='viridis', alpha=0.6)
    
    plt.title("BGAMF Fusion Gate Overlay", fontsize=16)
    plt.axis('off') # Hide axes for a cleaner look
    
    # Add a colorbar to show the mapping of colors to gate weights
    cbar = plt.colorbar()
    cbar.set_label('Gate Weight (0 -> DWI, 1 -> T2W)', rotation=270, labelpad=20)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Done.")

if __name__ == "__main__":
    D_MODEL = 128
    FEATURE_MAP_SIZE = 16 # The size of the feature map before flattening (e.g., 16x16)
    MRI_SIZE = 256 # The size of the original MRI image

    generate_and_save_gate_overlay(
        d_model=D_MODEL,
        feature_map_size=FEATURE_MAP_SIZE,
        mri_size=MRI_SIZE
    )
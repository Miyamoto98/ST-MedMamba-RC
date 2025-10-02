
import torch
import torch.nn as nn

# Import the MedSAM2 encoder and its components
from sam2.modeling.backbones.image_encoder import ImageEncoder
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine

# Import our custom modules
from .memory_block import MemoryBlock
from .mamba_fusion import MambaFusion

def get_medsam_encoder(d_model: int = 256):
    """
    Instantiates the MedSAM2 image encoder with parameters from its config.
    """
    encoder = ImageEncoder(
        scalp=1,
        trunk=Hiera(
            embed_dim=96, num_heads=1, stages=[1, 2, 7, 2],
            global_att_blocks=[5, 7, 9], window_pos_embed_bkg_spatial_size=[7, 7],
        ),
        neck=FpnNeck(
            position_encoding=PositionEmbeddingSine(num_pos_feats=256, normalize=True),
            d_model=d_model, backbone_channel_list=[768, 384, 192, 96],
            fpn_top_down_levels=[2, 3], fpn_interp_model="nearest",
        ),
    )
    return encoder

class RectalCancerStagingModel(nn.Module):
    def __init__(self, num_classes=4, d_model=256, dropout=0.3):
        """
        The end-to-end model for rectal cancer staging, incorporating the Memory Block.
        """
        super().__init__()
        self.d_model = d_model

        # --- 1. Feature Extractors ---
        self.t2w_encoder = get_medsam_encoder(d_model=self.d_model)
        self.dwi_encoder = get_medsam_encoder(d_model=self.d_model)

        # --- 2. Memory Blocks ---
        self.t2w_memory_block = MemoryBlock(d_model=d_model)
        self.dwi_memory_block = MemoryBlock(d_model=d_model)

        # --- 3. Mamba Fusion Module ---
        self.mamba_fusion = MambaFusion(d_model=d_model)

        # --- 4. Classification Head ---
        # The input to the classifier will be from the MambaFusion module.
        # MambaFusion concatenates the sequences, so the length is doubled.
        # We apply GAP over this doubled sequence length.
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), # Global Average Pooling
            nn.Flatten(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def _process_3d_input(self, encoder, volume_3d, slice_batch_size=4):
        """
        Helper to process a 3D volume slice by slice with a 2D encoder in batches.
        """
        b, c, d, h, w = volume_3d.shape
        
        all_features = []
        for i in range(0, d, slice_batch_size):
            batch_slices = volume_3d[:, :, i:i+slice_batch_size, :, :]
            current_d = batch_slices.shape[2]
            
            # Reshape to treat slices as batch items: (B*current_d, C, H, W)
            volume_2d_slices = batch_slices.permute(0, 2, 1, 3, 4).reshape(b * current_d, c, h, w)
            
            # The encoder expects 3 channels, so we repeat the single channel
            if c == 1:
                volume_2d_slices = volume_2d_slices.repeat(1, 3, 1, 1)

            # Get features from the 2D encoder
            feature_maps = encoder(volume_2d_slices)["backbone_fpn"][0]
            
            # Reshape back to (B, current_d, C, H, W) and then flatten for sequence models
            _, feat_c, feat_h, feat_w = feature_maps.shape
            # Reshape to (B, current_d, C, H*W)
            features_3d = feature_maps.reshape(b, current_d, feat_c, feat_h, feat_w)
            # Flatten spatial dims and combine with depth: (B, current_d*H*W, C)
            features_seq = features_3d.flatten(3).permute(0, 1, 3, 2).reshape(b, current_d * feat_h * feat_w, feat_c)
            all_features.append(features_seq)
            
        return torch.cat(all_features, dim=1)

    def forward(self, t2w_volume, dwi_volume, slice_batch_size=4):
        """
        Args:
            t2w_volume (torch.Tensor): Preprocessed T2W volume (B, C, D, H, W)
            dwi_volume (torch.Tensor): Preprocessed DWI volume (B, C, D, H, W)
            slice_batch_size (int): Batch size for processing 3D slices.
        """
        # --- 1. Feature Extraction ---
        # Process each 3D volume to get a feature sequence
        f_t2w_seq = self._process_3d_input(self.t2w_encoder, t2w_volume, slice_batch_size=slice_batch_size)
        f_dwi_seq = self._process_3d_input(self.dwi_encoder, dwi_volume, slice_batch_size=slice_batch_size)

        # --- 2. Memory Block Enhancement ---
        f_t2w_enhanced = self.t2w_memory_block(f_t2w_seq)
        f_dwi_enhanced = self.dwi_memory_block(f_dwi_seq)

        # --- 3. Mamba Fusion ---
        f_fused = self.mamba_fusion(f_t2w_enhanced, f_dwi_enhanced)

        # --- 4. Classification ---
        # Permute from (B, L, C) to (B, C, L) for pooling
        logits = self.classification_head(f_fused.permute(0, 2, 1))

        return logits

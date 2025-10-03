"""
This file implements the Bidirectional Gated & Alignment-aware Mamba Fusion (BGAMF) framework.

This is an OPTIMIZED version based on expert review, incorporating dropout for regularization
and an improved BiMamba fusion strategy (concatenation + projection) for richer feature interaction.

Core components of BGAMF:
1.  Alignment-aware Feature Mapping: A cross-attention module to spatially and semantically align
    features from different modalities (e.g., T2W and DWI MRI) before fusion.
2.  Bidirectional Mamba Modeling: A Bi-directional Mamba block that captures both forward and
    backward dependencies in the feature sequences, providing a global contextual understanding.
3.  Gated Fusion Mechanism: A dynamic gating unit that adaptively controls the contribution of
    each modality at different spatial locations, allowing the model to prioritize more
    informative features.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union

# ################################################################## #
# #                      1. BASIC BUILDING BLOCKS                    # #
# ################################################################## #

@dataclass
class MambaArgs:
    """
    Configuration arguments for a MambaBlock.
    """
    d_model: int
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4
    dt_rank: Union[int, str] = 'auto'
    bias: bool = False
    conv_bias: bool = True

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class MambaBlock(nn.Module):
    """
    A single, unidirectional Mamba block. This is the core SSM-based component.
    This is a naive implementation for functional correctness, not performance.
    """
    def __init__(self, args: MambaArgs):
        super().__init__()
        self.args = args
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner, out_channels=args.d_inner,
            bias=args.conv_bias, kernel_size=args.d_conv,
            groups=args.d_inner, padding=args.d_conv - 1,
        )
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        output = self.out_proj(y)
        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + u * D
        return y

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

# ################################################################## #
# #           2. OPTIMIZED BGAMF FRAMEWORK COMPONENTS              # #
# ################################################################## #

class AlignmentModule(nn.Module):
    """
    Alignment-aware Feature Mapping using Cross-Attention.
    This module aligns features from two modalities to correct for spatial/semantic misalignments.
    """
    def __init__(self, d_model: int, n_head: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.attn_t2w = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.attn_dwi = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, f_t2w, f_dwi):
        """
        Args:
            f_t2w: Features from T2W modality, shape (B, L, C)
            f_dwi: Features from DWI modality, shape (B, L, C)
        Returns:
            f_t2w_aligned: Aligned T2W features
            f_dwi_aligned: Aligned DWI features
        """
        # Align T2W features using DWI as context, with a residual connection
        f_t2w_aligned, _ = self.attn_t2w(query=f_t2w, key=f_dwi, value=f_dwi)
        f_t2w_aligned = self.norm1(f_t2w_aligned + f_t2w)

        # Align DWI features using T2W as context, with a residual connection
        f_dwi_aligned, _ = self.attn_dwi(query=f_dwi, key=f_t2w, value=f_t2w)
        f_dwi_aligned = self.norm2(f_dwi_aligned + f_dwi)

        return f_t2w_aligned, f_dwi_aligned

class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba Block (Optimized).
    Processes the sequence in both forward and backward directions, then fuses the outputs
    by concatenation and a linear projection. Includes dropout and a residual connection.
    """
    def __init__(self, args: MambaArgs, dropout: float = 0.1):
        super().__init__()
        self.forward_mamba = MambaBlock(args)
        self.backward_mamba = MambaBlock(args)
        self.out_proj = nn.Linear(args.d_model * 2, args.d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        # Forward pass
        fwd_out = self.forward_mamba(x)

        # Backward pass
        bwd_out = self.backward_mamba(torch.flip(x, dims=[1]))
        bwd_out = torch.flip(bwd_out, dims=[1])

        # Concatenate, project, apply dropout, and add residual connection
        merged = torch.cat([fwd_out, bwd_out], dim=-1)
        output = self.dropout(self.out_proj(merged))
        
        return self.norm(output + x)

# ################################################################## #
# #               3. THE COMPLETE BGAMF FUSION MODULE                # #
# ################################################################## #

class BGAMF(nn.Module):
    """
    Bidirectional Gated & Alignment-aware Mamba Fusion (BGAMF) Module.
    This is the main fusion framework that integrates all components.
    """
    def __init__(self, d_model: int, n_layer: int = 1, n_head_align: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        args = MambaArgs(d_model=d_model)

        # 1. Alignment-aware Feature Mapping
        self.alignment_module = AlignmentModule(d_model, n_head=n_head_align, dropout=dropout)

        # 2. Bidirectional Mamba Modeling (within each modality)
        self.bimamba_t2w = BiMambaBlock(args, dropout=dropout)
        self.bimamba_dwi = BiMambaBlock(args, dropout=dropout)

        # 3. Gated Fusion Mechanism
        self.gating_unit = nn.Linear(d_model * 2, d_model)
        self.gating_dropout = nn.Dropout(dropout)
        
        # Post-fusion processing
        self.fusion_norm = RMSNorm(d_model)
        self.post_fusion_bimamba = BiMambaBlock(args, dropout=dropout)

    def forward(self, f_t2w, f_dwi, return_gate: bool = False):
        """
        Args:
            f_t2w: Features from T2W modality, shape (B, L, C)
            f_dwi: Features from DWI modality, shape (B, L, C)
            return_gate: If True, returns the fusion gate for visualization.
        Returns:
            fused_output: The final fused and processed feature sequence.
            gate (optional): The computed fusion gate.
        """
        # --- Step 1: Alignment-aware Feature Mapping ---
        f_t2w_aligned, f_dwi_aligned = self.alignment_module(f_t2w, f_dwi)

        # --- Step 2: Bidirectional Mamba Modeling ---
        # Process each modality stream with a BiMamba block
        h_t2w = self.bimamba_t2w(f_t2w_aligned)
        h_dwi = self.bimamba_dwi(f_dwi_aligned)

        # --- Step 3: Gated Fusion Mechanism ---
        # Concatenate the processed features for gate computation
        h_combined = torch.cat([h_t2w, h_dwi], dim=-1)
        
        # Compute the gate: σ(W_g * [h_t2w; h_dwi])
        gate = torch.sigmoid(self.gating_unit(h_combined))
        
        # Apply the gate: gate * h_t2w + (1 - gate) * h_dwi
        f_fusion = self.gating_dropout(gate * h_t2w + (1 - gate) * h_dwi)
        
        # Add a residual connection from the two streams before final processing
        f_fusion_res = self.fusion_norm(f_fusion + h_t2w + h_dwi)

        # --- Step 4: Post-Fusion Processing ---
        # Process the fused sequence with another BiMamba block
        fused_output = self.post_fusion_bimamba(f_fusion_res)

        if return_gate:
            return fused_output, gate
        
        return fused_output

# For compatibility, we can keep the old name but point it to the new module
MambaFusion = BGAMF
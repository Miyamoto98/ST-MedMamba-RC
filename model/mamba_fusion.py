"""
Mamba-minimal implementation, taken from https://github.com/johnma2006/mamba-minimal
This file is modified to implement the Mamba Fusion module for multi-modal medical imaging.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union

# This is a simplified ModelArgs for Mamba, adapted for our use case.
@dataclass
class MambaArgs:
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

class MambaFusion(nn.Module):
    def __init__(self, d_model: int, n_layer: int = 1):
        super().__init__()
        args = MambaArgs(d_model=d_model)
        self.fusion_block = MambaBlock(args)
        self.norm = RMSNorm(d_model)

    def forward(self, f_t2w, f_dwi):
        """
        f_t2w: Enhanced features from T2W modality, shape (B, L, C)
        f_dwi: Enhanced features from DWI modality, shape (B, L, C)
        """
        # Concatenate features along the sequence dimension for joint processing
        combined_features = torch.cat([f_t2w, f_dwi], dim=1)
        
        # Process the combined sequence with Mamba
        fused_output = self.fusion_block(combined_features)
        
        # Add residual connection and normalize
        fused_output = self.norm(fused_output + combined_features)
        
        return fused_output
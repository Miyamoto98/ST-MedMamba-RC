
import torch
import torch.nn as nn

class MemoryBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        """
        A Memory Block to enhance features by capturing slice-to-slice relationships.
        Uses self-attention to refine features and cross-attention to query a memory bank.

        Args:
            d_model (int): The feature dimension.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.d_model = d_model

        # Self-attention to refine the features of each modality independently
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # A learnable "memory" bank. It's a set of learnable vectors.
        # The model will learn to store prototypical information about tumor variations here.
        num_memory_vectors = 64 # Hyperparameter
        self.memory_bank = nn.Parameter(torch.randn(1, num_memory_vectors, d_model))

        # Cross-attention to query the memory bank
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # Layer normalization and feed-forward network
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, features):
        """
        Args:
            features (torch.Tensor): Input features from the encoder, shape (B, L, C).
                                     B=batch, L=sequence length, C=d_model.

        Returns:
            torch.Tensor: Enhanced features, shape (B, L, C).
        """
        # 1. Self-Attention to refine features
        attn_output, _ = self.self_attn(features, features, features)
        features = self.norm1(features + attn_output)

        # 2. Cross-Attention to query the memory bank
        # The features act as the query, and the memory bank is the key/value.
        batch_size = features.shape[0]
        memory = self.memory_bank.repeat(batch_size, 1, 1)
        
        mem_output, _ = self.cross_attn(query=features, key=memory, value=memory)
        features = self.norm2(features + mem_output)

        # 3. Feed-Forward Network
        ffn_output = self.ffn(features)
        enhanced_features = features + ffn_output

        return enhanced_features

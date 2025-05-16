import torch
import torch.nn as nn

# ProbSparse Self-Attention
class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return attn_output

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.3):
        super().__init__()
        self.attn = ProbSparseSelfAttention(d_model, n_heads)
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.linear(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

# ✅ T3 (Simplified)
class Informer(nn.Module):
    def __init__(self, input_size, d_model=128, n_heads=4, e_layers=2, output_size=1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)

        self.encoder = nn.Sequential(*[
            EncoderLayer(d_model, n_heads)
            for _ in range(e_layers)
        ])

        self.output = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Sigmoid()  # 정규화된 경우
        )

    def forward(self, x):
        # x: [B, T, F]
        x = self.input_proj(x)     # [B, T, d_model]
        x = self.encoder(x)        # [B, T, d_model]
        x = x[:, -1, :]            # 마지막 시점의 embedding 사용
        return self.output(x)      # [B, output_size]

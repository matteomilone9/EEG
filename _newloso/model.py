"""
model.py — TCFormer (EEG-Only)
================================
Implementazione fedele di TCFormer come descritto in:
  Altaheri et al., "Temporal Convolutional Transformer for EEG-Based
  Motor Imagery Decoding", Scientific Reports, 2025.

Architettura:
  Input [B, C, T]
  └─ MK-CNN Block
     ├─ 3x Temporal Conv (K=20/32/64, F1=32 filtri ciascuno)
     ├─ BN + ELU
     ├─ Concatenazione lungo il canale CNN → [B, F1*3, C_eeg, T]
     ├─ DepthWise Spatial Conv (D=2)
     ├─ BN + ELU + AvgPool (P1=8) + Dropout
     ├─ PointWise 1x1 Conv → d_model = d_group * n_groups
     ├─ Temporal Conv (KC2=16) + BN + ELU + AvgPool (P2=7) + Dropout
     └─ Grouped SE Attention (1 peso scalare per gruppo)
  └─ Transformer Encoder (N layer)
     ├─ Pre-Norm LayerNorm
     ├─ Grouped-Query Attention (H query heads, G=H/2 kv groups)
     ├─ RoPE (Rotary Positional Embedding) su Q e K
     ├─ DropPath (schedule quadratico, max=0.25)
     ├─ FFN: Linear → GELU → Linear (expansion r=2)
     └─ Residual + Dropout
  └─ Fuse: concat(MK-CNN features, Transformer output) → [B, Tc, dF]
  └─ TCN Head (L=2 residual block, K_T=4, dilation 1 e 2)
     ├─ Conv1d grouped (n_groups+1) + BN + ELU + Dropout
     └─ Output: ultimo time step → [B, dF]
  └─ Classifier: Conv1d grouped → media logit per gruppo → [B, n_classes]

Tabella 1 (iperparametri):
  F1=32, K_c=(20,32,64), D=2, P1=8, P2=7
  d_group=16, N=2, H=4, G=2 (G=H/2), p_e=0.4
  K_T=4, L=2, p_t=0.3, drop_path_max=0.25
  FFN expansion r=2, KC2=16
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ════════════════════════════════════════════════════════════
# DropPath (stochastic depth)
# Schedule quadratico: drop_i = (i/(N-1))^2 * drop_path_max
# ════════════════════════════════════════════════════════════

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * noise / keep

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


# ════════════════════════════════════════════════════════════
# Grouped SE Attention
#
# Paper (Sezione "Grouped squeeze-and-excitation SE attention"):
#   "global average pooling → grouped 1x1 conv (reduction) → ReLU
#    → grouped 1x1 conv → sigmoid → broadcast per gruppo"
#   Ogni gruppo ha UN solo peso scalare (G pesi totali).
#   L'output viene sommato al residuo.
# ════════════════════════════════════════════════════════════

class GroupedSEAttention(nn.Module):
    """
    Grouped SE Attention lungo la dimensione dei canali CNN.
    n_groups = numero di kernel temporali (3 per K_c={20,32,64}).
    channels  = d_model = d_group * n_groups
    reduction = fattore di riduzione interno (default 4).
    """

    def __init__(self, channels: int, n_groups: int, reduction: int = 4) -> None:
        super().__init__()
        assert channels % n_groups == 0, "channels deve essere divisibile per n_groups"
        mid = max(1, channels // (n_groups * reduction))
        # Due 1x1 conv groupate: channels → mid → n_groups (1 scalare/gruppo)
        self.squeeze = nn.Conv1d(channels, mid * n_groups, 1, groups=n_groups, bias=False)
        self.excite  = nn.Conv1d(mid * n_groups, n_groups,  1, groups=n_groups, bias=False)
        self.channels  = channels
        self.n_groups  = n_groups

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, channels, T]
        z = x.mean(dim=-1, keepdim=True)        # [B, channels, 1]
        z = F.relu(self.squeeze(z))              # [B, mid*n_groups, 1]
        z = torch.sigmoid(self.excite(z))        # [B, n_groups, 1]
        # broadcast: ogni gruppo pesa tutti i suoi C/G canali
        g = self.channels // self.n_groups
        # [B, n_groups, 1] → [B, channels, 1]
        w = z.repeat_interleave(g, dim=1)
        return x + x * w                         # residuo


# ════════════════════════════════════════════════════════════
# MK-CNN Block
# ════════════════════════════════════════════════════════════

class MKCNNBlock(nn.Module):
    """
    Multi-Kernel CNN Block.

    Step 1: 3 conv temporali parallele (K_c=20/32/64, F1=32), BN, ELU
            → concat → [B, F1*n_groups, C_eeg, T]
    Step 2: DepthWise spatial conv (C_eeg×1, D=2)
            → BN, ELU, AvgPool(P1=8), Dropout
            → [B, F1*D*n_groups, 1, T//P1]
    Step 3: PointWise 1×1 → d_model = d_group*n_groups
            → [B, d_model, T//P1]
    Step 4: Temporal conv (KC2=16), BN, ELU, AvgPool(P2=7), Dropout
            → [B, d_model, Tc]   dove Tc = T//(P1*P2)
    Step 5: Grouped SE Attention
    """

    def __init__(
        self,
        n_channels:   int,           # C_eeg
        F1:           int = 32,      # filtri per kernel temporale
        kernel_lengths: Sequence[int] = (20, 32, 64),
        D:            int = 2,       # depth multiplier
        pool1:        int = 8,       # P1
        pool2:        int = 7,       # P2
        d_group:      int = 16,      # dim per gruppo
        dropout:      float = 0.4,   # p_c
        KC2:          int = 16,      # secondo kernel temporale
    ) -> None:
        super().__init__()
        self.n_groups = len(kernel_lengths)
        self.d_model  = d_group * self.n_groups
        d_model       = self.d_model

        # --- Step 1: conv temporali parallele ---
        self.temp_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, F1, (1, k), padding=(0, k // 2), bias=False),
                nn.BatchNorm2d(F1),
                nn.ELU(),
            )
            for k in kernel_lengths
        ])

        # --- Step 2: DepthWise spatial conv ---
        in_ch = F1 * self.n_groups
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * D, (n_channels, 1), groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch * D),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)),
            nn.Dropout(dropout),
        )

        # --- Step 3: PointWise 1×1 reduction ---
        self.pw_conv = nn.Sequential(
            nn.Conv2d(in_ch * D, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
        )

        # --- Step 4: Secondo temporal conv + pooling ---
        self.temp_conv2 = nn.Sequential(
            nn.Conv1d(d_model, d_model, KC2, padding=KC2 // 2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ELU(),
            nn.AvgPool1d(pool2),
            nn.Dropout(dropout),
        )

        # --- Step 5: Grouped SE Attention ---
        self.se = GroupedSEAttention(d_model, self.n_groups)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C_eeg, T]
        B, C, T = x.shape
        x = x.unsqueeze(1)                     # [B, 1, C_eeg, T]

        # Step 1: 3 conv temporali parallele
        branches = [conv(x) for conv in self.temp_convs]
        x = torch.cat(branches, dim=1)         # [B, F1*n_groups, C_eeg, T']

        # Step 2: DepthWise spatial
        x = self.dw_conv(x)                    # [B, F1*D*n_groups, 1, T'//P1]

        # Step 3: PointWise 1x1
        x = self.pw_conv(x)                    # [B, d_model, 1, T'//P1]
        x = x.squeeze(2)                       # [B, d_model, T'//P1]

        # Step 4: Secondo temporal conv + pooling
        x = self.temp_conv2(x)                 # [B, d_model, Tc]

        # Step 5: Grouped SE
        x = self.se(x)                         # [B, d_model, Tc]

        return x                               # [B, d_model, Tc]


# ════════════════════════════════════════════════════════════
# RoPE (Rotary Positional Embedding)
# ════════════════════════════════════════════════════════════

def _build_rope_cache(seq_len: int, head_dim: int, device: torch.device) -> Tensor:
    half = head_dim // 2
    theta = 1.0 / (10000 ** (torch.arange(0, half, device=device).float() / half))
    pos   = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)             # [Tc, half]
    return torch.cat([freqs, freqs], dim=-1)    # [Tc, head_dim]


def _rotate_half(x: Tensor) -> Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    # x: [B, n_heads, Tc, head_dim]
    # freqs: [Tc, head_dim]
    cos = freqs.cos()[None, None]
    sin = freqs.sin()[None, None]
    return x * cos + _rotate_half(x) * sin


# ════════════════════════════════════════════════════════════
# Grouped-Query Attention (GQA)
#
# Paper (Sezione "Transformer encoder with grouped-query attention"):
#   G = H/2  →  G gruppi di kv, H query heads
#   Ogni gruppo condivide una K e una V.
#   RoPE applicato su Q e K.
# ════════════════════════════════════════════════════════════

class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model:   int,
        q_heads:   int,
        kv_heads:  int,    # G = H/2
        dropout:   float = 0.0,
    ) -> None:
        super().__init__()
        assert q_heads % kv_heads == 0, "q_heads deve essere multiplo di kv_heads"
        assert d_model  % q_heads == 0, "d_model deve essere divisibile per q_heads"

        self.q_heads  = q_heads
        self.kv_heads = kv_heads
        self.head_dim = d_model // q_heads
        self.scale    = self.head_dim ** -0.5

        self.W_q = nn.Linear(d_model, d_model,                    bias=False)
        self.W_k = nn.Linear(d_model, kv_heads * self.head_dim,   bias=False)
        self.W_v = nn.Linear(d_model, kv_heads * self.head_dim,   bias=False)
        self.W_o = nn.Linear(d_model, d_model,                    bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, Tc, d = x.shape

        Q = self.W_q(x).reshape(B, Tc, self.q_heads,  self.head_dim).transpose(1, 2)
        K = self.W_k(x).reshape(B, Tc, self.kv_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).reshape(B, Tc, self.kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        freqs = _build_rope_cache(Tc, self.head_dim, x.device)
        Q = apply_rope(Q, freqs)
        K = apply_rope(K, freqs)

        # Espandi K e V per i query head (repeat per gruppo)
        ratio = self.q_heads // self.kv_heads
        K = K.repeat_interleave(ratio, dim=1)  # [B, q_heads, Tc, head_dim]
        V = V.repeat_interleave(ratio, dim=1)

        # Scaled dot-product attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, Tc, d)
        return self.W_o(out)


# ════════════════════════════════════════════════════════════
# Transformer Layer (pre-norm, GQA + FFN)
# ════════════════════════════════════════════════════════════

class TransformerLayer(nn.Module):
    """
    Pre-norm Transformer layer:
      Zi_norm = LN(Zi)
      Oi = Zi + Dropout(GQA(Zi_norm))
      Ei = Oi + Dropout(FFN(LN(Oi)))

    FFN: Linear → GELU → Linear  (expansion r=2)
    DropPath: stochastic depth per layer.
    """

    def __init__(
        self,
        d_model:    int,
        q_heads:    int,
        kv_heads:   int,
        ffn_expand: int   = 2,
        dropout:    float = 0.4,
        drop_path:  float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = GroupedQueryAttention(d_model, q_heads, kv_heads, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expand, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expand, d_model, bias=False),
        )
        self.drop2     = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.drop1(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.drop2(self.ffn(self.norm2(x))))
        return x


# ════════════════════════════════════════════════════════════
# Transformer Encoder
# ════════════════════════════════════════════════════════════

class TransformerEncoder(nn.Module):
    """
    N layer Transformer con schedule DropPath quadratico:
      drop_i = (i / (N-1))^2 * drop_path_max

    Pointwise 1x1 conv in ingresso per mescolare le informazioni
    tra i gruppi (come descritto nel paper).

    Output: proiezione da d_model → d_group (per la fusione con MK-CNN).
    """

    def __init__(
        self,
        d_model:       int,
        d_group:       int,
        n_layers:      int   = 2,
        q_heads:       int   = 4,
        kv_heads:      int   = 2,
        ffn_expand:    int   = 2,
        dropout:       float = 0.4,
        drop_path_max: float = 0.25,
    ) -> None:
        super().__init__()

        # DropPath schedule quadratico
        def _dp(i: int) -> float:
            if n_layers <= 1:
                return 0.0
            return ((i / (n_layers - 1)) ** 2) * drop_path_max

        self.mix_in = nn.Conv1d(d_model, d_model, 1, bias=False)

        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                q_heads=q_heads,
                kv_heads=kv_heads,
                ffn_expand=ffn_expand,
                dropout=dropout,
                drop_path=_dp(i),
            )
            for i in range(n_layers)
        ])

        self.norm   = nn.LayerNorm(d_model)
        self.proj   = nn.Conv1d(d_model, d_group, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, d_model, Tc]
        x = self.mix_in(x)                 # [B, d_model, Tc]
        x = x.permute(0, 2, 1)            # [B, Tc, d_model]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1)            # [B, d_model, Tc]
        return self.proj(x)               # [B, d_group, Tc]


# ════════════════════════════════════════════════════════════
# TCN Residual Block
#
# Paper (Sezione "Temporal convolutional network"):
#   "groups = n_groups + 1"
#   2 conv dilate causali per block, BN, ELU
#   dilation = 2^block_index (1 nel primo, 2 nel secondo block)
# ════════════════════════════════════════════════════════════

class TCNResBlock(nn.Module):
    def __init__(
        self,
        d_in:     int,
        d_out:    int,
        kernel:   int   = 4,
        dilation: int   = 1,
        n_groups: int   = 4,    # n_groups + 1
        dropout:  float = 0.3,
    ) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(d_in,  d_out, kernel, dilation=dilation,
                                padding=pad, groups=n_groups, bias=False)
        self.bn1   = nn.BatchNorm1d(d_out)
        self.conv2 = nn.Conv1d(d_out, d_out, kernel, dilation=dilation,
                                padding=pad, groups=n_groups, bias=False)
        self.bn2   = nn.BatchNorm1d(d_out)
        self.drop  = nn.Dropout(dropout)
        self.resid = (nn.Conv1d(d_in, d_out, 1, bias=False)
                      if d_in != d_out else nn.Identity())

    def _causal_trim(self, x: Tensor, pad: int) -> Tensor:
        # Rimuove il padding causale dall'estremità destra
        return x[:, :, :-pad] if pad > 0 else x

    def forward(self, x: Tensor) -> Tensor:
        r  = self.resid(x)
        pad = self.conv1.padding[0]
        h  = self.drop(F.elu(self.bn1(self._causal_trim(self.conv1(x),  pad))))
        h  = self.drop(F.elu(self.bn2(self._causal_trim(self.conv2(h),  pad))))
        return h + r


# ════════════════════════════════════════════════════════════
# TCN Head
#
# Paper: L=2 residual block, dilation 1 e 2, K_T=4
#   "only the final output (last time step) is retained"
# ════════════════════════════════════════════════════════════

class TCNHead(nn.Module):
    def __init__(
        self,
        d_in:     int,          # d_F = (n_groups+1) * d_group
        n_groups: int,          # n_groups + 1
        kernel:   int   = 4,
        n_blocks: int   = 2,
        dropout:  float = 0.3,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            TCNResBlock(
                d_in     = d_in,
                d_out    = d_in,
                kernel   = kernel,
                dilation = 2 ** i,
                n_groups = n_groups,
                dropout  = dropout,
            )
            for i in range(n_blocks)
        ])

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, d_F, Tc]
        for block in self.blocks:
            x = block(x)
        return x[:, :, -1]    # ultimo time step → [B, d_F]


# ════════════════════════════════════════════════════════════
# Classifier
#
# Paper: "pointwise 1x1 conv within each group,
#         d_group → n_classes; average along group dim"
# ════════════════════════════════════════════════════════════

class GroupedClassifier(nn.Module):
    def __init__(
        self,
        d_group:   int,
        n_groups:  int,
        n_classes: int,
        max_norm:  float = 0.25,
    ) -> None:
        super().__init__()
        n_g_total  = n_groups + 1          # include il gruppo Transformer
        d_in       = d_group * n_g_total
        self.n_groups   = n_g_total
        self.d_group    = d_group
        self.n_classes  = n_classes
        # 1x1 conv grouped: [B, d_in, 1] → [B, n_classes * n_g_total, 1]
        self.conv  = nn.Conv1d(d_in, n_classes * n_g_total,
                               1, groups=n_g_total, bias=False)
        self.max_norm = max_norm

    def _apply_max_norm(self) -> None:
        w = self.conv.weight
        norms = w.view(w.shape[0], -1).norm(dim=1, keepdim=True).clamp(min=1e-8)
        scale = (norms / self.max_norm).clamp(min=1.0)
        self.conv.weight.data = w / scale.view(-1, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, d_F]  dove d_F = d_group * (n_groups+1)
        self._apply_max_norm()
        x = x.unsqueeze(-1)                # [B, d_F, 1]
        x = self.conv(x)                   # [B, n_classes*(n_groups+1), 1]
        x = x.squeeze(-1)                  # [B, n_classes*(n_groups+1)]
        x = x.reshape(x.shape[0], self.n_groups, self.n_classes)
        return x.mean(dim=1)               # [B, n_classes]


# ════════════════════════════════════════════════════════════
# TCFormer
# ════════════════════════════════════════════════════════════

class TCFormer(nn.Module):
    """
    TCFormer EEG-Only.

    Args:
        n_channels:          Numero di canali EEG (es. 22 per BCIC IV-2a)
        n_classes:           Numero di classi MI (es. 4)
        F1:                  Filtri per kernel temporale (default 32)
        kernel_lengths:      Kernel temporali paralleli (default (20,32,64))
        D:                   Depth multiplier DepthWise (default 2)
        pool1:               Primo AvgPool stride (default 8)
        pool2:               Secondo AvgPool stride (default 7)
        d_group:             Dimensione per gruppo (default 16)
        trans_depth:         Numero layer Transformer (default 2)
        q_heads:             Query heads (default 4)
        kv_heads:            KV heads = G = H/2 (default 2)
        ffn_expansion:       Expansion ratio FFN (default 2)
        trans_dropout:       Dropout Transformer (default 0.4)
        conv_dropout:        Dropout MK-CNN (default 0.4)
        drop_path_max:       Max DropPath (default 0.25)
        tcn_depth:           Blocchi TCN (default 2)
        tcn_kernel:          Kernel TCN (default 4)
        tcn_dropout:         Dropout TCN (default 0.3)
        classifier_max_norm: Max norm classificatore (default 0.25)
    """

    def __init__(
        self,
        n_channels:          int   = 22,
        n_classes:           int   = 4,
        F1:                  int   = 32,
        kernel_lengths:      Sequence[int] = (20, 32, 64),
        D:                   int   = 2,
        pool1:               int   = 8,
        pool2:               int   = 7,
        d_group:             int   = 16,
        trans_depth:         int   = 2,
        q_heads:             int   = 4,
        kv_heads:            int   = 2,
        ffn_expansion:       int   = 2,
        trans_dropout:       float = 0.4,
        conv_dropout:        float = 0.4,
        drop_path_max:       float = 0.25,
        tcn_depth:           int   = 2,
        tcn_kernel:          int   = 4,
        tcn_dropout:         float = 0.3,
        classifier_max_norm: float = 0.25,
    ) -> None:
        super().__init__()
        self.n_groups = len(kernel_lengths)
        d_model = d_group * self.n_groups

        self.mk_cnn = MKCNNBlock(
            n_channels    = n_channels,
            F1            = F1,
            kernel_lengths= kernel_lengths,
            D             = D,
            pool1         = pool1,
            pool2         = pool2,
            d_group       = d_group,
            dropout       = conv_dropout,
        )

        self.transformer = TransformerEncoder(
            d_model       = d_model,
            d_group       = d_group,
            n_layers      = trans_depth,
            q_heads       = q_heads,
            kv_heads      = kv_heads,
            ffn_expand    = ffn_expansion,
            dropout       = trans_dropout,
            drop_path_max = drop_path_max,
        )

        # d_F = (n_groups + 1) * d_group
        d_F = (self.n_groups + 1) * d_group

        self.tcn = TCNHead(
            d_in    = d_F,
            n_groups= self.n_groups + 1,
            kernel  = tcn_kernel,
            n_blocks= tcn_depth,
            dropout = tcn_dropout,
        )

        self.classifier = GroupedClassifier(
            d_group   = d_group,
            n_groups  = self.n_groups,
            n_classes = n_classes,
            max_norm  = classifier_max_norm,
        )

    def get_features(self, x: Tensor) -> Tensor:
        """Restituisce le feature prima del classificatore [B, d_F]."""
        cnn_feat  = self.mk_cnn(x)                         # [B, d_model, Tc]
        trans_out = self.transformer(cnn_feat)              # [B, d_group, Tc]
        fused     = torch.cat([cnn_feat, trans_out], dim=1) # [B, d_F, Tc]
        return self.tcn(fused)                              # [B, d_F]

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.get_features(x))       # [B, n_classes]


# ════════════════════════════════════════════════════════════
# Factory
# ════════════════════════════════════════════════════════════

def build_tcformer(cfg: Optional[dict] = None) -> TCFormer:
    """
    Costruisce TCFormer dai parametri di configurazione.
    I valori di default corrispondono alla Tabella 1 del paper
    (Altaheri et al., 2025) per BCIC IV-2a.
    """
    if cfg is None:
        cfg = {}
    return TCFormer(
        n_channels          = cfg.get("n_channels",          22),
        n_classes           = cfg.get("n_classes",            4),
        F1                  = cfg.get("F1",                  32),
        kernel_lengths      = cfg.get("temp_kernel_lengths",  (20, 32, 64)),
        D                   = cfg.get("D",                    2),
        pool1               = cfg.get("pool_length_1",        8),
        pool2               = cfg.get("pool_length_2",        7),
        d_group             = cfg.get("d_group",             16),
        trans_depth         = cfg.get("trans_depth",          2),
        q_heads             = cfg.get("q_heads",              4),
        kv_heads            = cfg.get("kv_heads",             2),
        ffn_expansion       = cfg.get("ffn_expansion",        2),
        trans_dropout       = cfg.get("trans_dropout",        0.4),
        conv_dropout        = cfg.get("dropout_conv",         0.4),
        drop_path_max       = cfg.get("drop_path_max",        0.25),
        tcn_depth           = cfg.get("tcn_depth",            2),
        tcn_kernel          = cfg.get("tcn_kernel",           4),
        tcn_dropout         = cfg.get("tcn_dropout",          0.3),
        classifier_max_norm = cfg.get("classifier_max_norm",  0.25),
    )

# model_kd.py — Modelli self-contained per pipeline Teacher-Student KD
# Tutti i building block di model.py sono inclusi qui: nessun import da model.py
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════
# Utilities (da model.py)
# ════════════════════════════════════════════════════════════

def glorot_zero(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, **kw):
        pad = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation, **kw)
        self.causal_pad = pad

    def forward(self, x):
        out = super().forward(x)
        return out[..., :-self.causal_pad] if self.causal_pad > 0 else out


class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, max_norm=1.0, **kw):
        self.max_norm = max_norm
        super().__init__(*args, **kw)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class ECABlock2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        k = int(abs(math.log2(channels) + 1))
        k = k if k % 2 == 1 else k + 1
        k = max(k, 3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        y = self.avgpool(x).squeeze(-1).transpose(-1, -2)
        y = self.sigmoid(self.conv(y))
        return x * y.transpose(-1, -2).unsqueeze(-1).expand_as(x)


class ChannelGroupAttention(nn.Module):
    def __init__(self, in_channels: int, num_groups: int):
        super().__init__()
        assert in_channels % num_groups == 0
        self.num_groups = num_groups
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_groups),
            nn.Softmax(dim=-1),
        )
        glorot_zero(self)

    def forward(self, x):
        B, C, T = x.shape
        w = self.attn(x).view(B, self.num_groups, 1, 1)
        g = C // self.num_groups
        return (x.view(B, self.num_groups, g, T) * w).view(B, C, T)


class MultiKernelConvBlock(nn.Module):
    def __init__(self, n_channels, temp_kernel_lengths=(20, 32, 64),
                 F1=16, D=2, pool_length_1=8, pool_length_2=7,
                 dropout=0.3, d_group=16, use_group_attn=True):
        super().__init__()
        from einops.layers.torch import Rearrange
        self.rearrange = Rearrange("b c seq -> b 1 c seq")
        self.temporal_convs = nn.ModuleList([
            nn.Sequential(
                nn.ConstantPad2d((k // 2 - 1, k // 2, 0, 0) if k % 2 == 0 else (k // 2, k // 2, 0, 0), 0),
                nn.Conv2d(1, F1, (1, k), bias=False),
                nn.BatchNorm2d(F1),
            )
            for k in temp_kernel_lengths
        ])
        n_groups = len(temp_kernel_lengths)
        self.d_model = d_group * n_groups
        F2 = F1 * n_groups * D
        self.channel_DW_conv = nn.Sequential(
            nn.Conv2d(F1 * n_groups, F2, (n_channels, 1), bias=False, groups=F1 * n_groups),
            nn.BatchNorm2d(F2),
            nn.ELU(),
        )
        self.pool1 = nn.AvgPool2d((1, pool_length_1))
        self.drop1 = nn.Dropout(dropout)
        self.eca = ECABlock2d(F2)
        self.use_cr2 = self.d_model != F2
        if self.use_cr2:
            self.channel_reduction2 = nn.Sequential(
                nn.Conv2d(F2, self.d_model, (1, 1), bias=False, groups=n_groups),
                nn.BatchNorm2d(self.d_model),
            )
        self.temporal_conv2 = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, (1, 16), padding="same", bias=False, groups=n_groups),
            nn.BatchNorm2d(self.d_model),
            nn.ELU(),
        )
        self.use_group_attn = use_group_attn and n_groups > 1
        if self.use_group_attn:
            self.group_attn = ChannelGroupAttention(self.d_model, n_groups)
        self.pool2 = nn.AvgPool2d((1, pool_length_2))
        self.drop2 = nn.Dropout(dropout)
        glorot_zero(self)

    def forward(self, x):
        x = self.rearrange(x)
        cx = [c(x) for c in self.temporal_convs]
        x = torch.cat(cx, dim=1)
        x = self.drop1(self.pool1(self.channel_DW_conv(x)))
        x = self.eca(x)
        if self.use_cr2:
            x = self.channel_reduction2(x)
        x = self.temporal_conv2(x)
        if self.use_group_attn:
            xs = x.squeeze(2)
            xs = self.group_attn(xs)
            x = xs.unsqueeze(2)
        return self.drop2(self.pool2(x)).squeeze(2)


def build_rope_cache(head_dim, seq_len, device):
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    idx = torch.arange(seq_len, device=device).float()
    emb = torch.cat([torch.outer(idx, theta)] * 2, dim=-1)
    return emb.cos(), emb.sin()


def rope_rotate(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def apply_rope(q, k, cos, sin):
    return q * cos + rope_rotate(q) * sin, k * cos + rope_rotate(k) * sin


class GQAttention(nn.Module):
    def __init__(self, d_model, q_heads, kv_heads, dropout=0.3):
        super().__init__()
        assert d_model % q_heads == 0 and q_heads % kv_heads == 0
        self.qh, self.kvh = q_heads, kv_heads
        self.hd = d_model // q_heads
        self.scale = self.hd ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * kv_heads * self.hd, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        glorot_zero(self)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.qh, self.hd).transpose(1, 2)
        kv = self.kv_proj(x).view(B, T, self.kvh, 2, self.hd)
        k = kv[..., 0, :].transpose(1, 2).repeat_interleave(self.qh // self.kvh, dim=1)
        v = kv[..., 1, :].transpose(1, 2).repeat_interleave(self.qh // self.kvh, dim=1)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        attn = self.drop((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.o_proj((attn @ v).transpose(1, 2).contiguous().view(B, T, C))


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0 or not self.training:
            return x
        kp = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x / kp * torch.rand(shape, dtype=x.dtype, device=x.device).floor_()


class TransformerBlock(nn.Module):
    def __init__(self, d_model, q_heads, kv_heads, dropout=0.4, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GQAttention(d_model, q_heads, kv_heads, dropout)
        self.dp = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, cos, sin):
        x = x + self.dp(self.attn(self.norm1(x), cos, sin))
        x = x + self.dp(self.mlp(self.norm2(x)))
        return x


class TCNBlock(nn.Module):
    def __init__(self, nf, ks=4, dil=1, ng=1, dropout=0.3):
        super().__init__()
        self.c1 = CausalConv1d(nf, nf, ks, dil, groups=ng)
        self.b1 = nn.BatchNorm1d(nf)
        self.e1 = nn.ELU()
        self.d1 = nn.Dropout(dropout)
        self.c2 = CausalConv1d(nf, nf, ks, dil, groups=ng)
        self.b2 = nn.BatchNorm1d(nf)
        self.e2 = nn.ELU()
        self.d2 = nn.Dropout(dropout)
        self.act = nn.ELU()
        nn.init.constant_(self.c1.bias, 0)
        nn.init.constant_(self.c2.bias, 0)

    def forward(self, x):
        h = self.d1(self.e1(self.b1(self.c1(x))))
        h = self.d2(self.e2(self.b2(self.c2(h))))
        return self.act(x + h)


class TCNHead(nn.Module):
    def __init__(self, d, n_groups, n_classes, depth=2, ks=4, dropout=0.3):
        super().__init__()
        self.ng = n_groups
        self.nc = n_classes
        self.tcn = nn.Sequential(*[TCNBlock(d, ks, 2 ** i, n_groups, dropout) for i in range(depth)])
        self.cls = Conv1dWithConstraint(d, n_classes * n_groups, 1, groups=n_groups, max_norm=0.25)

    def forward(self, x):
        x = self.tcn(x)[..., -1]        # [B, d, T] → [B, d]
        x = self.cls(x.unsqueeze(-1))   # [B, d] → [B, d, 1] → Conv1d → [B, n_classes*n_groups, 1]
        return x.view(x.size(0), self.ng, self.nc).mean(1)


class TCFormerModule(nn.Module):
    def __init__(self, n_channels, n_classes,
                 F1=16, temp_kernel_lengths=(20, 32, 64), D=2,
                 pool_length_1=8, pool_length_2=7, dropout_conv=0.3,
                 d_group=16, use_group_attn=True,
                 q_heads=4, kv_heads=2,
                 trans_depth=2, trans_dropout=0.4, drop_path_max=0.1,
                 tcn_depth=2, kernel_length_tcn=4, dropout_tcn=0.3):
        super().__init__()
        from einops.layers.torch import Rearrange
        n_groups = len(temp_kernel_lengths)
        self.d_model = d_group * n_groups
        self.d_group = d_group
        self.n_groups = n_groups
        self.conv_block = MultiKernelConvBlock(
            n_channels, temp_kernel_lengths, F1, D,
            pool_length_1, pool_length_2, dropout_conv, d_group, use_group_attn,
        )
        self.mix = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, 1, bias=False),
            nn.BatchNorm1d(self.d_model),
            nn.SiLU(),
        )
        self.to_seq = Rearrange("b c t -> b t c")
        dpr = torch.linspace(0, 1, trans_depth) ** 2 * drop_path_max
        self.transformer = nn.ModuleList([
            TransformerBlock(self.d_model, q_heads, kv_heads, trans_dropout, dpr[i].item())
            for i in range(trans_depth)
        ])
        self.reduce = nn.Sequential(
            Rearrange("b t c -> b c t"),
            nn.Conv1d(self.d_model, d_group, 1, bias=False),
            nn.BatchNorm1d(d_group),
            nn.SiLU(),
        )
        d_tcf = d_group * (n_groups + 1)
        self.tcn_head = TCNHead(d_tcf, n_groups + 1, n_classes, tcn_depth, kernel_length_tcn, dropout_tcn)
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        glorot_zero(self)

    def _rope_cache(self, T, dev):
        hd = self.transformer[0].attn.hd
        if self.cos is None or self.cos.shape[0] < T:
            self.cos, self.sin = build_rope_cache(hd, T, dev)
        return self.cos, self.sin

    def get_features(self, x):
        conv_f = self.conv_block(x)
        B, C, T = conv_f.shape
        tok = self.to_seq(self.mix(conv_f))
        cos, sin = self._rope_cache(T, x.device)
        for blk in self.transformer:
            tok = blk(tok, cos, sin)
        return torch.cat([conv_f, self.reduce(tok)], dim=1)

    def forward(self, x):
        return self.tcn_head(self.get_features(x))


# ════════════════════════════════════════════════════════════
# KD-specific classes
# ════════════════════════════════════════════════════════════

def _get_n_groups(cfg: dict, prefix: str) -> int:
    return len(cfg[f"{prefix}_temp_kernel_lengths"])


def _get_d_model(cfg: dict, prefix: str) -> int:
    return cfg[f"{prefix}_d_group"] * _get_n_groups(cfg, prefix)


# ── Student ──────────────────────────────────────────────────

class EEGStudentWrapper(nn.Module):
    """Wrapper dello student EEG-only basato su TCFormerModule."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.student = TCFormerModule(
            n_channels=cfg["n_channels"],
            n_classes=cfg["n_classes"],
            F1=cfg["student_F1"],
            temp_kernel_lengths=cfg["student_temp_kernel_lengths"],
            D=cfg["student_D"],
            pool_length_1=cfg["student_pool_length_1"],
            pool_length_2=cfg["student_pool_length_2"],
            dropout_conv=cfg["student_dropout_conv"],
            d_group=cfg["student_d_group"],
            use_group_attn=cfg["student_use_group_attn"],
            q_heads=cfg["student_q_heads"],
            kv_heads=cfg["student_kv_heads"],
            trans_depth=cfg["student_trans_depth"],
            trans_dropout=cfg["student_trans_dropout"],
            drop_path_max=cfg["student_drop_path_max"],
            tcn_depth=cfg["student_tcn_depth"],
            kernel_length_tcn=cfg["student_kernel_length_tcn"],
            dropout_tcn=cfg["student_dropout_tcn"],
        )

    def forward(self, eeg):
        return self.student(eeg)

    def get_features(self, eeg):
        return self.student.get_features(eeg)


def build_student_model(cfg: dict) -> nn.Module:
    return EEGStudentWrapper(cfg)


# ── GAF token encoder ────────────────────────────────────────

class GAFTokenEncoder(nn.Module):
    """Encoder leggero che trasforma GAF in token [B, 1, d_model]."""

    def __init__(self, n_channels: int, token_dim: int = 64,
                 base_channels: int = 16, dropout: float = 0.3):
        super().__init__()
        self.token_dim = token_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, token_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(token_dim),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
        )
        glorot_zero(self)

    def forward(self, gaf):
        if gaf.dim() == 4:
            B, C, H, W = gaf.shape
            x = gaf.view(B * C, 1, H, W)
        elif gaf.dim() == 5:
            B, C, N_CH, H, W = gaf.shape
            x = gaf.view(B * C * N_CH, 1, H, W)
        else:
            raise ValueError(f"GAF shape inatteso: {gaf.shape}")
        x = self.cnn(x).flatten(1)
        x = self.proj(x)
        x = x.view(B, -1, self.token_dim)
        return x.mean(dim=1, keepdim=True)   # [B, 1, token_dim]


# ── GAF Projection Head ───────────────────────────────────────

class GafProjectionHead(nn.Module):
    """
    Projection head temporaneo: mappa embedding EEG → spazio GAF.
    Usato solo durante il KD-Align training, poi scartato.
    """

    def __init__(self, d_in: int, d_out: int = 64, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
        )
        glorot_zero(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(-1)   # [B, d_in]
        return self.proj(x)


def build_gaf_proj_head(cfg: dict) -> GafProjectionHead:
    n_groups = _get_n_groups(cfg, "student")
    d_tcf = cfg["student_d_group"] * (n_groups + 1)
    d_out = cfg["teacher_gaf_token_dim"]
    dropout = cfg["student_dropout_conv"]
    return GafProjectionHead(d_in=d_tcf, d_out=d_out, dropout=dropout)


# ── Cross-modal attention block ──────────────────────────────

class CrossModalAttentionBlock(nn.Module):
    """Cross-attention pre-norm: EEG tokens = query, GAF tokens = key/value."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.3,
                 ff_mult: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.dp = DropPath(drop_path)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        glorot_zero(self)

    def forward(self, x_eeg, x_gaf):
        q = self.norm_q(x_eeg)
        kv = self.norm_kv(x_gaf)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = x_eeg + self.dp(attn_out)
        x = x + self.dp(self.ff(self.norm_ff(x)))
        return x


# ── Teacher cross-modal ──────────────────────────────────────

class TeacherCrossModal(nn.Module):
    """
    Teacher: backbone EEG + GAF encoder + cross-attention.
    use_gaf=False → equivalente EEG-only.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.use_gaf = cfg.get("use_gaf", True)
        self.n_classes = cfg["n_classes"]

        self.eeg_backbone = TCFormerModule(
            n_channels=cfg["n_channels"],
            n_classes=cfg["n_classes"],
            F1=cfg["teacher_F1"],
            temp_kernel_lengths=cfg["teacher_temp_kernel_lengths"],
            D=cfg["teacher_D"],
            pool_length_1=cfg["teacher_pool_length_1"],
            pool_length_2=cfg["teacher_pool_length_2"],
            dropout_conv=cfg["teacher_dropout_conv"],
            d_group=cfg["teacher_d_group"],
            use_group_attn=cfg["teacher_use_group_attn"],
            q_heads=cfg["teacher_q_heads"],
            kv_heads=cfg["teacher_kv_heads"],
            trans_depth=cfg["teacher_trans_depth"],
            trans_dropout=cfg["teacher_trans_dropout"],
            drop_path_max=cfg["teacher_drop_path_max"],
            tcn_depth=cfg["teacher_tcn_depth"],
            kernel_length_tcn=cfg["teacher_kernel_length_tcn"],
            dropout_tcn=cfg["teacher_dropout_tcn"],
        )

        self.eeg_feat_dim = cfg["teacher_d_group"] * (_get_n_groups(cfg, "teacher") + 1)

        if self.use_gaf:
            self.cross_dim = cfg["teacher_gaf_token_dim"]

            self.eeg_proj = nn.Sequential(
                nn.Conv1d(self.eeg_feat_dim, self.cross_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(self.cross_dim),
                nn.SiLU(),
            )

            self.gaf_encoder = GAFTokenEncoder(
                n_channels=cfg["n_channels"],
                token_dim=cfg["teacher_gaf_token_dim"],
                base_channels=cfg["teacher_gaf_base_channels"],
                dropout=cfg["teacher_gaf_dropout"],
            )

            dpr = torch.linspace(0, cfg["teacher_drop_path_max"],
                                 cfg["teacher_cross_attn_depth"]).tolist()
            self.cross_blocks = nn.ModuleList([
                CrossModalAttentionBlock(
                    d_model=cfg["teacher_gaf_token_dim"],
                    n_heads=cfg["teacher_cross_attn_heads"],
                    dropout=cfg["teacher_cross_attn_dropout"],
                    ff_mult=cfg["teacher_ff_mult"],
                    drop_path=dpr[i],
                )
                for i in range(cfg["teacher_cross_attn_depth"])
            ])

            self.cls_head = nn.Sequential(
                nn.LayerNorm(cfg["teacher_gaf_token_dim"]),
                nn.Linear(cfg["teacher_gaf_token_dim"], cfg["teacher_cls_hidden"]),
                nn.GELU(),
                nn.Dropout(cfg["teacher_cross_attn_dropout"]),
                nn.Linear(cfg["teacher_cls_hidden"], cfg["n_classes"]),
            )

            glorot_zero(self)

    def _forward_eeg_only(self, eeg):
        return self.eeg_backbone(eeg)

    def get_eeg_tokens(self, eeg):
        feat = self.eeg_backbone.get_features(eeg)   # [B, C, T]
        tok = self.eeg_proj(feat).transpose(1, 2)    # [B, T, D]
        return tok

    def get_gaf_tokens(self, gaf):
        return self.gaf_encoder(gaf)   # [B, 1, D]

    def get_teacher_features(self, eeg, gaf=None):
        if not self.use_gaf or gaf is None:
            return self.eeg_backbone.get_features(eeg)
        gaf_is_real = gaf.shape[-1] > 1 and gaf.shape[-2] > 1
        if not gaf_is_real:
            return self.eeg_backbone.get_features(eeg)
        x_eeg = self.get_eeg_tokens(eeg)
        x_gaf = self.get_gaf_tokens(gaf)
        for blk in self.cross_blocks:
            x_eeg = blk(x_eeg, x_gaf)
        return x_eeg.mean(dim=1)   # [B, D]

    def forward(self, eeg, gaf=None):
        if not self.use_gaf or gaf is None:
            return self._forward_eeg_only(eeg), None
        gaf_is_real = gaf.shape[-1] > 1 and gaf.shape[-2] > 1
        if not gaf_is_real:
            return self._forward_eeg_only(eeg), None
        pooled = self.get_teacher_features(eeg, gaf)   # [B, D]
        return self.cls_head(pooled), None


def build_teacher_model(cfg: dict) -> nn.Module:
    return TeacherCrossModal(cfg)
# model.py — Architettura TCFormerWithAux
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Utilità ──────────────────────────────────────────────────

def glorot_zero(module: nn.Module):
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ── Convoluzione causale e con vincolo di norma ───────────────

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, **kw):
        pad = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation, **kw)
        self._causal_pad = pad

    def forward(self, x):
        out = super().forward(x)
        return out[..., :-self._causal_pad] if self._causal_pad > 0 else out


class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, max_norm=1.0, **kw):
        self.max_norm = max_norm
        super().__init__(*args, **kw)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


# ── ECA + Channel Group Attention ────────────────────────────

class ECABlock2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        k = int(abs(math.log2(channels) + 1))
        k = k if k % 2 == 1 else k + 1
        k = max(k, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv     = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid  = nn.Sigmoid()
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.sigmoid(self.conv(y))
        return x * y.transpose(-1, -2).unsqueeze(-1).expand_as(x)


class ChannelGroupAttention(nn.Module):
    def __init__(self, in_channels: int, num_groups: int):
        super().__init__()
        assert in_channels % num_groups == 0
        self.num_groups = num_groups
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(in_channels, num_groups), nn.Softmax(dim=-1))
        glorot_zero(self)

    def forward(self, x):
        B, C, T = x.shape
        w = self.attn(x).view(B, self.num_groups, 1, 1)
        g = C // self.num_groups
        return (x.view(B, self.num_groups, g, T) * w).view(B, C, T)


# ── Multi-Kernel Conv Block ───────────────────────────────────

class MultiKernelConvBlock(nn.Module):
    def __init__(self, n_channels, temp_kernel_lengths=(20, 32, 64),
                 F1=16, D=2, pool_length_1=8, pool_length_2=7,
                 dropout=0.3, d_group=16, use_group_attn=True):
        super().__init__()
        from einops.layers.torch import Rearrange
        self.rearrange = Rearrange("b c seq -> b 1 c seq")
        self.temporal_convs = nn.ModuleList([
            nn.Sequential(
                nn.ConstantPad2d((k//2-1, k//2, 0, 0) if k % 2 == 0 else (k//2, k//2, 0, 0), 0),
                nn.Conv2d(1, F1, (1, k), bias=False),
                nn.BatchNorm2d(F1),
            ) for k in temp_kernel_lengths])
        n_groups      = len(temp_kernel_lengths)
        self.d_model  = d_group * n_groups
        F2            = F1 * n_groups * D
        self.channel_DW_conv = nn.Sequential(
            nn.Conv2d(F1 * n_groups, F2, (n_channels, 1), bias=False, groups=F1 * n_groups),
            nn.BatchNorm2d(F2), nn.ELU())
        self.pool1 = nn.AvgPool2d((1, pool_length_1))
        self.drop1 = nn.Dropout(dropout)
        self.eca   = ECABlock2d(F2)
        self.use_cr2 = (self.d_model != F2)
        if self.use_cr2:
            self.channel_reduction_2 = nn.Sequential(
                nn.Conv2d(F2, self.d_model, (1, 1), bias=False, groups=n_groups),
                nn.BatchNorm2d(self.d_model))
        self.temporal_conv_2 = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, (1, 16), padding='same',
                      bias=False, groups=n_groups),
            nn.BatchNorm2d(self.d_model), nn.ELU())
        self.use_group_attn = use_group_attn and n_groups > 1
        if self.use_group_attn:
            self.group_attn = ChannelGroupAttention(self.d_model, n_groups)
        self.pool2 = nn.AvgPool2d((1, pool_length_2))
        self.drop2 = nn.Dropout(dropout)
        glorot_zero(self)

    def forward(self, x):
        x = self.rearrange(x)
        x = torch.cat([c(x) for c in self.temporal_convs], dim=1)
        x = self.drop1(self.pool1(self.channel_DW_conv(x)))
        x = self.eca(x)
        if self.use_cr2:
            x = self.channel_reduction_2(x)
        x = self.temporal_conv_2(x)
        if self.use_group_attn:
            xs = x.squeeze(2)
            xs = xs + self.group_attn(xs)
            x  = xs.unsqueeze(2)
        return self.drop2(self.pool2(x)).squeeze(2)


# ── RoPE ─────────────────────────────────────────────────────

def _build_rope_cache(head_dim, seq_len, device):
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    idx   = torch.arange(seq_len, device=device).float()
    emb   = torch.cat((torch.outer(idx, theta),) * 2, dim=-1)
    return emb.cos(), emb.sin()


def _rope_rotate(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rope(q, k, cos, sin):
    return (q * cos) + (_rope_rotate(q) * sin), (k * cos) + (_rope_rotate(k) * sin)


# ── GQA + DropPath + TransformerBlock ────────────────────────

class GQAttention(nn.Module):
    def __init__(self, d_model, q_heads, kv_heads, dropout=0.3):
        super().__init__()
        assert d_model % q_heads == 0 and q_heads % kv_heads == 0
        self.qh, self.kvh = q_heads, kv_heads
        self.hd    = d_model // q_heads
        self.scale = self.hd ** -0.5
        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * kv_heads * self.hd, bias=False)
        self.o_proj  = nn.Linear(d_model, d_model, bias=False)
        self.drop    = nn.Dropout(dropout)
        glorot_zero(self)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q  = self.q_proj(x).view(B, T, self.qh, self.hd).transpose(1, 2)
        kv = self.kv_proj(x).view(B, T, self.kvh, 2, self.hd)
        k  = kv[..., 0, :].transpose(1, 2).repeat_interleave(self.qh // self.kvh, dim=1)
        v  = kv[..., 1, :].transpose(1, 2).repeat_interleave(self.qh // self.kvh, dim=1)
        q, k = _apply_rope(q, k, cos[:T], sin[:T])
        attn = self.drop((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.o_proj((attn @ v).transpose(1, 2).contiguous().view(B, T, C))


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0 or not self.training: return x
        kp    = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x / kp * (kp + torch.rand(shape, dtype=x.dtype, device=x.device)).floor_()


class TransformerBlock(nn.Module):
    def __init__(self, d_model, q_heads, kv_heads, dropout=0.4, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = GQAttention(d_model, q_heads, kv_heads, dropout)
        self.dp    = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x, cos, sin):
        x = x + self.dp(self.attn(self.norm1(x), cos, sin))
        x = x + self.dp(self.mlp(self.norm2(x)))
        return x


# ── TCN ──────────────────────────────────────────────────────

class TCNBlock(nn.Module):
    def __init__(self, nf, ks=4, dil=1, ng=1, dropout=0.3):
        super().__init__()
        self.c1  = CausalConv1d(nf, nf, ks, dil, groups=ng)
        self.b1  = nn.BatchNorm1d(nf)
        self.e1  = nn.ELU()
        self.d1  = nn.Dropout(dropout)
        self.c2  = CausalConv1d(nf, nf, ks, dil, groups=ng)
        self.b2  = nn.BatchNorm1d(nf)
        self.e2  = nn.ELU()
        self.d2  = nn.Dropout(dropout)
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
        self.ng  = n_groups
        self.nc  = n_classes
        self.tcn = nn.Sequential(*[TCNBlock(d, ks, 2**i, n_groups, dropout) for i in range(depth)])
        self.cls = Conv1dWithConstraint(d, n_classes * n_groups, 1, groups=n_groups, max_norm=0.25)

    def forward(self, x):
        x = self.tcn(x)[..., -1:]
        x = self.cls(x).squeeze(-1)
        return x.view(x.size(0), self.ng, self.nc).mean(1)


# ── TCFormerModule ───────────────────────────────────────────

class TCFormerModule(nn.Module):
    def __init__(self, n_channels, n_classes, F1=16, temp_kernel_lengths=(20, 32, 64),
                 D=2, pool_length_1=8, pool_length_2=7, dropout_conv=0.3, d_group=16,
                 use_group_attn=True, q_heads=4, kv_heads=2, trans_depth=2,
                 trans_dropout=0.4, drop_path_max=0.1, tcn_depth=2,
                 kernel_length_tcn=4, dropout_tcn=0.3):
        super().__init__()
        from einops.layers.torch import Rearrange
        n_groups      = len(temp_kernel_lengths)
        self.d_model  = d_group * n_groups
        self.d_group  = d_group
        self.n_groups = n_groups
        self.conv_block = MultiKernelConvBlock(
            n_channels, temp_kernel_lengths, F1, D,
            pool_length_1, pool_length_2, dropout_conv, d_group, use_group_attn)
        self.mix = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, 1, bias=False),
            nn.BatchNorm1d(self.d_model), nn.SiLU())
        self.to_seq = Rearrange("b c t -> b t c")
        dpr = (torch.linspace(0, 1, trans_depth) ** 2 * drop_path_max).tolist()
        self.transformer = nn.ModuleList([
            TransformerBlock(self.d_model, q_heads, kv_heads, trans_dropout, dpr[i])
            for i in range(trans_depth)])
        self.reduce = nn.Sequential(
            Rearrange("b t c -> b c t"),
            nn.Conv1d(self.d_model, d_group, 1, bias=False),
            nn.BatchNorm1d(d_group), nn.SiLU())
        d_tcf = d_group * (n_groups + 1)
        self.tcn_head = TCNHead(d_tcf, n_groups + 1, n_classes,
                                tcn_depth, kernel_length_tcn, dropout_tcn)
        self.register_buffer("_cos", None, persistent=False)
        self.register_buffer("_sin", None, persistent=False)
        glorot_zero(self)

    def _rope_cache(self, T, dev):
        hd = self.transformer[0].attn.hd
        if self._cos is None or self._cos.shape[0] < T:
            self._cos, self._sin = _build_rope_cache(hd, T, dev)
        return self._cos, self._sin

    def get_features(self, x):
        conv_f = self.conv_block(x)
        B, C, T = conv_f.shape
        tok = self.to_seq(self.mix(conv_f))
        cos, sin = self._rope_cache(T, x.device)
        for blk in self.transformer:
            tok = blk(tok, cos, sin)
        return torch.cat((conv_f, self.reduce(tok)), dim=1)

    def forward(self, x):
        return self.tcn_head(self.get_features(x))


# ── GAF Encoder + Fusion Heads ───────────────────────────────

class GAFMiniEncoder(nn.Module):
    def __init__(self, n_channels=22, out_dim=64, dropout=0.4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, out_dim, 3, padding=1), nn.BatchNorm2d(out_dim),
            nn.ELU(), nn.AdaptiveAvgPool2d((2, 2)))
        self.pool_ch = nn.Sequential(
            nn.Linear(out_dim * 4, out_dim), nn.ELU(), nn.Dropout(dropout))
        glorot_zero(self)

    def forward(self, x):
        B, C, H, W = x.shape
        f = self.cnn(x.view(B * C, 1, H, W)).view(B, C, -1)
        return self.pool_ch(f.mean(1))


class AuxHead(nn.Module):
    """Fusione additiva v14/v15 — default."""
    def __init__(self, d_tcf, gaf_dim, n_classes, hidden=64, dropout=0.4):
        super().__init__()
        self.tcf_proj = nn.Sequential(nn.Linear(d_tcf, hidden), nn.LayerNorm(hidden))
        self.gaf_proj = nn.Sequential(nn.Linear(gaf_dim, hidden), nn.LayerNorm(hidden))
        self.cls      = nn.Sequential(nn.ELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))
        glorot_zero(self)

    def forward(self, tcf_feat, gaf_emb):
        return self.cls(self.tcf_proj(tcf_feat.mean(-1)) + self.gaf_proj(gaf_emb))


class GAFCrossAttnHead(nn.Module):
    """Fusione cross-attention v15 — attivabile con cross_attention=True."""
    def __init__(self, d_tcf, gaf_dim, n_classes, dk=32, hidden=64,
                 dropout=0.4, attn_dropout=0.3):
        super().__init__()
        self.scale     = dk ** -0.5
        self.q_proj    = nn.Linear(d_tcf,   dk, bias=False)
        self.kv_proj   = nn.Linear(gaf_dim, dk, bias=False)
        self.out_proj  = nn.Linear(dk, d_tcf,   bias=False)
        self.norm      = nn.LayerNorm(d_tcf)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.cls = nn.Sequential(
            nn.Linear(d_tcf, hidden), nn.LayerNorm(hidden),
            nn.ELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))
        glorot_zero(self)

    def forward(self, tcf_feat, gaf_emb):
        B, d_tcf, T = tcf_feat.shape
        x       = tcf_feat.transpose(1, 2)
        Q       = self.q_proj(x)
        KV      = self.kv_proj(gaf_emb).unsqueeze(1)
        scores  = (Q @ KV.transpose(-2, -1)) * self.scale
        weights = self.attn_drop(scores.softmax(dim=1))
        attn_out = weights * KV
        x = self.norm(x + self.out_proj(attn_out))
        return self.cls(x.mean(1))


# ── Modello completo ─────────────────────────────────────────

class TCFormerWithAux(nn.Module):
    def __init__(self, n_channels, n_classes, cfg):
        super().__init__()
        self.use_gaf  = cfg['use_gaf']
        self.tcformer = TCFormerModule(
            n_channels=n_channels, n_classes=n_classes,
            F1=cfg['F1'], temp_kernel_lengths=cfg['temp_kernel_lengths'],
            D=cfg['D'], pool_length_1=cfg['pool_length_1'],
            pool_length_2=cfg['pool_length_2'], dropout_conv=cfg['dropout_conv'],
            d_group=cfg['d_group'], use_group_attn=cfg['use_group_attn'],
            q_heads=cfg['q_heads'], kv_heads=cfg['kv_heads'],
            trans_depth=cfg['trans_depth'], trans_dropout=cfg['trans_dropout'],
            drop_path_max=cfg['drop_path_max'], tcn_depth=cfg['tcn_depth'],
            kernel_length_tcn=cfg['kernel_length_tcn'], dropout_tcn=cfg['dropout_tcn'])
        if self.use_gaf:
            n_groups = len(cfg['temp_kernel_lengths'])
            d_tcf    = cfg['d_group'] * (n_groups + 1)
            gaf_dim  = cfg['gaf_aux_hidden']
            self.gaf_enc = GAFMiniEncoder(n_channels, gaf_dim, cfg['gaf_aux_dropout'])
            if cfg.get('cross_attention', False):
                self.aux_head = GAFCrossAttnHead(
                    d_tcf=d_tcf, gaf_dim=gaf_dim, n_classes=n_classes,
                    dk=cfg.get('cross_attn_dk', 32), hidden=cfg['gaf_aux_hidden'],
                    dropout=cfg['gaf_aux_dropout'],
                    attn_dropout=cfg.get('cross_attn_dropout', 0.3))
            else:
                self.aux_head = AuxHead(d_tcf, gaf_dim, n_classes,
                                        cfg['gaf_aux_hidden'], cfg['gaf_aux_dropout'])

    def forward(self, eeg, gaf=None):
        if self.use_gaf and gaf is not None:
            # Skip GAF encoder se dummy (Fase 1 LOSO: H==1, W==1)
            gaf_is_real = gaf.shape[-1] > 1 and gaf.shape[-2] > 1
            feat        = self.tcformer.get_features(eeg)
            logits_main = self.tcformer.tcn_head(feat)
            if gaf_is_real:
                logits_aux = self.aux_head(feat.detach(), self.gaf_enc(gaf))
                return logits_main, logits_aux
            else:
                return logits_main, None
        return self.tcformer(eeg)


def build_model(cfg) -> TCFormerWithAux:
    return TCFormerWithAux(cfg['n_channels'], cfg['n_classes'], cfg)
# model.py
# ─────────────────────────────────────────────────────────────────────────────
# QR-ViT-Det: Vision Transformer + DETR-style detection head
#
# Architecture overview
# ─────────────────────
#  Input (B, 1, 180, 320)
#       │
#  CNN Stem   4× conv blocks, stride=10 total
#       │ (B, embed_dim, 18, 32)
#  Flatten + Positional Encoding
#       │ (B, 576, embed_dim)
#  Transformer Encoder  (enc_layers × MHA + FFN)
#       │ (B, 3600, embed_dim)
#  Transformer Decoder  (dec_layers × cross-attn + self-attn)
#       ↑
#  Object Queries  (num_queries, embed_dim)  — learned
#       │
#  Detection FFN  →  sigmoid → (B, num_queries, 4)  boxes
#                →  linear  → (B, num_queries, 2)   logits  [qr, no-obj]
# ─────────────────────────────────────────────────────────────────────────────

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG


# ──────────────────────────────────────────────────────────────────────────────
# CNN Stem: learns patch-like features, outputs 1/10 spatial resolution
# ──────────────────────────────────────────────────────────────────────────────


class CNNStem(nn.Module):
    """
    Fixed CNN Stem: No skipped pixels!
    Reduces (1, 180, 320) -> (embed_dim, 18, 32) exactly.
    """
    def __init__(self, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            # Block 1: Stride 2 (halves the image safely)
            # Output: (32, 90, 160)
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            
            # Block 2: Kernel 5 / Stride 5 (Tiles perfectly, zero skipped pixels)
            # Output: (64, 18, 32)
            nn.Conv2d(32, 64, kernel_size=5, stride=5, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            # Block 3: Feature mixing (maintains 18x32 spatial size)
            # Output: (embed_dim, 18, 32)
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

# ──────────────────────────────────────────────────────────────────────────────
# 2-D Sinusoidal Positional Encoding
# ──────────────────────────────────────────────────────────────────────────────


class PositionalEncoding2D(nn.Module):
    """
    Fixed 2-D sinusoidal positional encoding for a (H_feat × W_feat) grid.
    Adds (1, H*W, embed_dim) to the sequence.
    """

    def __init__(self, embed_dim: int, h_feat: int, w_feat: int):
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2-D sin/cos PE"
        d = embed_dim // 2  # half for x, half for y
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))

        pe = torch.zeros(h_feat, w_feat, embed_dim)

        y_pos = torch.arange(h_feat).unsqueeze(1).float()  # (H, 1)
        x_pos = torch.arange(w_feat).unsqueeze(1).float()  # (W, 1)

        # Y encoding → first half of channels
        pe[:, :, 0:d:2] = (torch.sin(y_pos * div_term)).unsqueeze(1).expand(h_feat, w_feat, -1)
        pe[:, :, 1:d:2] = (torch.cos(y_pos * div_term)).unsqueeze(1).expand(h_feat, w_feat, -1)

        # X encoding → second half of channels
        pe[:, :, d::2] = (torch.sin(x_pos * div_term)).unsqueeze(0).expand(h_feat, w_feat, -1)
        pe[:, :, d + 1 :: 2] = (torch.cos(x_pos * div_term)).unsqueeze(0).expand(h_feat, w_feat, -1)

        pe = pe.view(1, h_feat * w_feat, embed_dim)  # (1, H*W, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, embed_dim)"""
        return x + self.pe


# ──────────────────────────────────────────────────────────────────────────────
# Transformer building blocks
# ──────────────────────────────────────────────────────────────────────────────
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    # Change the forward signature
    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        # Add position to Query and Key (Standard DETR practice)
        q = k = h + pos
        attn_out, _ = self.self_attn(q, k, h)
        x = x + self.drop(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    # ADD 'pos' to the signature
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, query_pos: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # 1. Self-attention over queries
        h = self.norm1(tgt)
        q = k = h + query_pos
        sa, _ = self.self_attn(q, k, h)
        tgt = tgt + self.drop(sa)

        # 2. Cross-attention: queries attend to encoder memory
        h = self.norm2(tgt)
        q = h + query_pos
        k = memory + pos  
        
        ca, _ = self.cross_attn(q, k, memory) # Pass q, k, and memory(values)
        tgt = tgt + self.drop(ca)

        tgt = tgt + self.ffn(self.norm3(tgt))
        return tgt


# ──────────────────────────────────────────────────────────────────────────────
# Detection FFN (shared, called on each query embedding)
# ──────────────────────────────────────────────────────────────────────────────


class MLP(nn.Module):
    """Simple 3-layer MLP used as the detection head."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Full QR-ViT-Det Model
# ──────────────────────────────────────────────────────────────────────────────


class QRViTDet(nn.Module):
    """
    End-to-end QR code detector.

    Inputs
    ------
    x : (B, 1, H, W)  float32 in [0, 1]

    Outputs (dict)
    ------
    "pred_boxes"   : (B, num_queries, 4)  sigmoid-activated, values in [0,1]
    "pred_logits"  : (B, num_queries, 2)  raw logits  [qr_code, no-object]
    """

    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg

        h_feat = cfg.img_h // cfg.patch_stride
        w_feat = cfg.img_w // cfg.patch_stride

        # ── Backbone ────────────────────────────────────────────────────────
        self.backbone = CNNStem(cfg.in_channels, cfg.embed_dim)

        # ── Positional Encoding ─────────────────────────────────────────────
        self.pos_enc = PositionalEncoding2D(cfg.embed_dim, h_feat, w_feat)

        # ── Encoder ─────────────────────────────────────────────────────────
        self.encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout)
                for _ in range(cfg.enc_layers)
            ]
        )
        self.enc_norm = nn.LayerNorm(cfg.embed_dim)

        # ── Object Queries ──────────────────────────────────────────────────
        self.query_embed = nn.Embedding(cfg.num_queries, cfg.embed_dim)

        # ── Decoder ─────────────────────────────────────────────────────────
        self.decoder = nn.ModuleList(
            [
                TransformerDecoderLayer(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout)
                for _ in range(cfg.dec_layers)
            ]
        )
        self.dec_norm = nn.LayerNorm(cfg.embed_dim)

        # ── Detection heads ─────────────────────────────────────────────────
        self.bbox_head = MLP(cfg.embed_dim, cfg.embed_dim, 4, num_layers=3)
        self.class_head = nn.Linear(cfg.embed_dim, 2)  # [qr_code, no-object]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                
        # Give queries a random starting point to prevent collapse
        nn.init.normal_(self.query_embed.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> dict:
        B = x.size(0)

        # ── Backbone ─────────────────────────────────────────────────────────
        feat = self.backbone(x)  # (B, E, H/10, W/10)
        E = feat.size(1)
        feat_seq = feat.flatten(2).permute(0, 2, 1)  # (B, S, E)  S=576

        # Get the 2D position ONCE
        pos = self.pos_enc.pe # This is the (1, 576, 128) tensor

        # ── Encoder ──────────────────────────────────────────────────────────
        memory = feat_seq
        
        for layer in self.encoder:
            memory = layer(memory, pos=pos) # Inject at every layer
        memory = self.enc_norm(memory)

        # ── Decoder ──────────────────────────────────────────────────────────
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, E)
        tgt = torch.zeros_like(queries)  # DETR strictly starts with zeros!
        
        for layer in self.decoder:
            # Pass BOTH query_pos and the image pos
            tgt = layer(tgt, memory, query_pos=queries, pos=pos) 
            
        tgt = self.dec_norm(tgt)  # (B, Q, E)

        # ── Heads ─────────────────────────────────────────────────────────────
        pred_cxcywh = self.bbox_head(tgt).sigmoid()  # (B, Q, 4) in [0,1]
        
        cx, cy, w, h = pred_cxcywh.unbind(-1)
        
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        
        pred_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        pred_logits = self.class_head(tgt)  # (B, Q, 2)

        return {"pred_boxes": pred_boxes, "pred_logits": pred_logits}
    
    @property
    def backbone_params(self):
        return list(self.backbone.parameters())

    @property
    def non_backbone_params(self):
        backbone_ids = {id(p) for p in self.backbone.parameters()}
        return [p for p in self.parameters() if id(p) not in backbone_ids]

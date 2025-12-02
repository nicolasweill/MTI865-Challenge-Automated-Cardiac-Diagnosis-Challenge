import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

# ---------------------------
# Utils
# ---------------------------
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class MLP(nn.Module):
    def __init__(self, in_dim, mlp_dim, drop=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_dim, in_dim),
            nn.Dropout(drop),
        )
    def forward(self,x): return self.net(x)

# ---------------------------
# Patch embedding from feature map (Conv stem -> patch tokens)
# ---------------------------
class PatchEmbedFromFeature(nn.Module):
    """
    Convert a CNN feature map (B, C, H, W) into patch tokens for Transformer:
    - splits HxW into non-overlapping patches of size (patch_size x patch_size)
    - flattens each patch and projects to embed_dim
    """
    def __init__(self, in_ch, embed_dim, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        # resulting tokens = (H/patch)*(W/patch)
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)                 # [B, embed, Hp, Wp]
        B, E, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1,2)  # [B, N_tokens, E]
        return x, (Hp, Wp)

# ---------------------------
# Simple Transformer Encoder block (multihead + MLP)
# ---------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, qkv_bias=True, drop=0., attn_drop=0., drop_path_prob=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path_prob = drop_path_prob
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_dim, drop)

    def forward(self, x):
        # x: [B, N, D]
        x_att = self.norm1(x)
        attn_out, _ = self.attn(x_att, x_att, x_att, need_weights=False)
        x = x + drop_path(attn_out, self.drop_path_prob, self.training)
        x_mlp = self.norm2(x)
        x = x + drop_path(self.mlp(x_mlp), self.drop_path_prob, self.training)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, depth, dim, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(dim, **kwargs) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        for l in self.layers: x = l(x)
        return self.norm(x)

# ---------------------------
# Decoder block
# ---------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, groups=8, dropout=0.0):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups,out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups,out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x, skip):
        x = self.upsample(x)
        if skip is not None:
            # center crop if mismatch
            if skip.shape[2:] != x.shape[2:]:
                th, tw = x.shape[2:]
                sh, sw = skip.shape[2:]
                top = (sh - th) // 2
                left = (sw - tw) // 2
                skip = skip[:, :, top:top+th, left:left+tw]
            x = torch.cat([x, skip], dim=1)
        else:
            # no skip available
            pass
        return self.conv(x)

# ---------------------------
# Hybrid Transformer U-Net (2D)
# ---------------------------
class HybridTransformerUNet2D(nn.Module):
    """
    - Conv encoder (hierarchical) to produce multi-scale skips
    - At deepest level, create patch tokens from conv features and run Transformer
    - Project transformer tokens back to spatial map and decode with U-Net like decoder
    Returns: logits (B, num_classes, H, W) OR (logits, aux1, aux2, aux3) if deep_supervision True
    """
    def __init__(self,
                 in_channels=1,
                 num_classes=4,
                 base_filters=32,
                 embed_dim=512,
                 trans_depth=8,
                 trans_heads=8,
                 patch_size=2,
                 drop_path=0.0,
                 deep_supervision=True):
        super().__init__()

        # store hyperparams as attributes for later use
        self.f = base_filters
        self.num_classes = num_classes

        f = self.f

        # encoder: conv stem -> enc1..enc4
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, f, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8,f), num_channels=f),
            nn.ReLU(inplace=True)
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(f, f, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8,f), num_channels=f),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(f, f*2, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8,f*2), num_channels=f*2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(f*2, f*4, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8,f*4), num_channels=f*4),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)

        # deep features before transformer
        self.enc4 = nn.Sequential(
            nn.Conv2d(f*4, f*8, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8,f*8), num_channels=f*8),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(2)

        # project conv features to patch tokens
        # choose embed_dim and patch_size so that spatial dims reduce reasonably
        self.patch_emb = PatchEmbedFromFeature(in_ch=f*8, embed_dim=embed_dim, patch_size=patch_size)

        # transformer encoder
        self.transformer = TransformerEncoder(depth=trans_depth, dim=embed_dim,
                                              num_heads=trans_heads, mlp_ratio=4.0, drop=0.0,
                                              attn_drop=0.0, drop_path_prob=drop_path)

        # project tokens back to a spatial feature map
        self.token_to_map = nn.Linear(embed_dim, f*8)  # maps token dim -> decoder feature channels

        # decoder projection conv to combine reprojected tokens
        self.decoder_conv_proj = nn.Conv2d(f*8, f*8, 3, padding=1)

        # decoder blocks (mirrors encoder)
        self.dec4 = DecoderBlock(in_ch=f*8, skip_ch=f*8, out_ch=f*4)
        self.dec3 = DecoderBlock(in_ch=f*4, skip_ch=f*4, out_ch=f*2)
        self.dec2 = DecoderBlock(in_ch=f*2, skip_ch=f*2, out_ch=f)
        self.dec1 = DecoderBlock(in_ch=f, skip_ch=f, out_ch=f)

        # final conv
        self.final = nn.Sequential(
            nn.Conv2d(f, f, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8,f), num_channels=f),
            nn.ReLU(inplace=True),
            nn.Conv2d(f, num_classes, kernel_size=1)
        )

        # deep supervision layers: defined once here with correct in_channels
        self.deep_supervision = deep_supervision
        if deep_supervision:
            # d1 -> out_ch = f
            # d2 -> out_ch = f
            # d3 -> out_ch = f*2
            self.ds1 = nn.Conv2d(f, self.num_classes, 1)
            self.ds2 = nn.Conv2d(f, self.num_classes, 1)
            self.ds3 = nn.Conv2d(f*2, self.num_classes, 1)

        # init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0.0)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        s = self.stem(x)
        e1 = self.enc1(s)           # [B, f, H, W]
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)          # [B, 2f, H/2, W/2]
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)          # [B, 4f, H/4, W/4]
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)          # [B, 8f, H/8, W/8]
        p4 = self.pool4(e4)         # [B, 8f, H/16, W/16]

        # Patch embedding + transformer
        tokens, (Hp, Wp) = self.patch_emb(p4)     # tokens: [B, N, E]
        tokens = self.transformer(tokens)         # [B, N, E]

        # map tokens back to spatial map
        # tokens -> [B, N, D] -> reshape [B, D, Hp, Wp]
        proj = self.token_to_map(tokens)          # [B, N, D2]
        Bp, N, D2 = proj.shape
        proj = proj.transpose(1,2).reshape(Bp, D2, Hp, Wp)  # [B, D2, Hp, Wp]
        proj = self.decoder_conv_proj(proj)       # [B, D2, Hp, Wp]

        # Upsample proj to match p4 spatial if needed
        p4_up = F.interpolate(proj, size=p4.shape[2:], mode='bilinear', align_corners=False)

        # Decoder: combine with skips
        d4 = self.dec4(p4_up, e4)   # -> [B, 4f, H/8, W/8]
        d3 = self.dec3(d4, e3)      # -> [B, 2f, H/4, W/4]
        d2 = self.dec2(d3, e2)      # -> [B, f, H/2, W/2]
        d1 = self.dec1(d2, e1)      # -> [B, f, H, W]

        out = self.final(d1)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        if self.deep_supervision:
            aux1 = self.ds1(d1)  # expects f channels
            aux2 = self.ds2(d2)  # expects f channels
            aux3 = self.ds3(d3)  # expects 2f channels

            aux1 = F.interpolate(aux1, size=(H, W), mode='bilinear', align_corners=False)
            aux2 = F.interpolate(aux2, size=(H, W), mode='bilinear', align_corners=False)
            aux3 = F.interpolate(aux3, size=(H, W), mode='bilinear', align_corners=False)

            return out, aux1, aux2, aux3

        return out

# ---------------------------
# Quick sanity check
# ---------------------------
if __name__ == "__main__":
    model = HybridTransformerUNet2D(in_channels=1, num_classes=4, base_filters=32,
                                    embed_dim=256, trans_depth=6, trans_heads=8, patch_size=2)
    x = torch.randn(2,1,224,224)
    y = model(x)
    if isinstance(y, tuple):
        print("Output shapes:", [t.shape for t in y])
    else:
        print("Output shape:", y.shape)

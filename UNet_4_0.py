import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Implémente le Stochastic Depth
    Pendant l'entraînement, on "éteint" aléatoirement certaines branches entières du réseau.
    Cela force le réseau à être robuste 
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    #gestion des dimensions pour que le masque soit broadcastable
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class MLP(nn.Module):
    """
    Multi-Layer Perceptron utilisé dans les blocs Transformer
    Structure : Linear -> GELU -> Dropout -> Linear -> Dropout
    """
    def __init__(self, in_dim, mlp_dim, drop=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.GELU(),# Activation plus douce que ReLU
            nn.Dropout(drop),
            nn.Linear(mlp_dim, in_dim),
            nn.Dropout(drop),
        )
    def forward(self,x): return self.net(x)


class PatchEmbedFromFeature(nn.Module):
    """
    Transforme une Feature Map CNN (2D) en une séquence de tokens (1D) pour le Transformer.
    """
    def __init__(self, in_ch, embed_dim, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        # On utilise une convolution avec stride=patch_size pour découper et projeter en même temps
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                 
        B, E, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1,2) 
        return x, (Hp, Wp)


class TransformerEncoderLayer(nn.Module):
    """
    LayerNorm -> Attention -> LayerNorm -> MLP
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, qkv_bias=True, drop=0., attn_drop=0., drop_path_prob=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path_prob = drop_path_prob
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_dim, drop)

    def forward(self, x):
        x_att = self.norm1(x)
        attn_out, _ = self.attn(x_att, x_att, x_att, need_weights=False)
        x = x + drop_path(attn_out, self.drop_path_prob, self.training)
        x_mlp = self.norm2(x)
        x = x + drop_path(self.mlp(x_mlp), self.drop_path_prob, self.training)
        return x

class TransformerEncoder(nn.Module):
    """
    Empilement de plusieurs couches TransformerEncoderLayer
    """
    def __init__(self, depth, dim, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(dim, **kwargs) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)# Normalisation finale

    def forward(self, x):
        for l in self.layers: x = l(x)
        return self.norm(x)



class DecoderBlock(nn.Module):
    """
    Bloc de décodage U-Net classique
    Upsample -> Concaténation (Skip) -> Double Convolution
    """
    def __init__(self, in_ch, skip_ch, out_ch, groups=8, dropout=0.0):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Convolution pour fusionner l'entrée upsample et le skip connection
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
            # center crop si mismatch d'un pixel
            if skip.shape[2:] != x.shape[2:]:
                th, tw = x.shape[2:]
                sh, sw = skip.shape[2:]
                top = (sh - th) // 2
                left = (sw - tw) // 2
                skip = skip[:, :, top:top+th, left:left+tw]
            x = torch.cat([x, skip], dim=1)
        else:
            pass
        return self.conv(x)


class HybridTransformerUNet2D(nn.Module):
    """
    Architecture TransUNet : 
    CNN Encoder -> Transformer Bottleneck -> CNN Decoder
    
    Args:
        in_channels (int): 1 image
        num_classes (int): 4 (BG, RV, Myo, LV)
        embed_dim (int): Dimension du Transformer 
        trans_depth (int): Nombre de couches Transformer 
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

        self.f = base_filters
        self.num_classes = num_classes

        f = self.f

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

        self.enc4 = nn.Sequential(
            nn.Conv2d(f*4, f*8, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8,f*8), num_channels=f*8),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(2)

        self.patch_emb = PatchEmbedFromFeature(in_ch=f*8, embed_dim=embed_dim, patch_size=patch_size)

        self.transformer = TransformerEncoder(depth=trans_depth, dim=embed_dim,
                                              num_heads=trans_heads, mlp_ratio=4.0, drop=0.0,
                                              attn_drop=0.0, drop_path_prob=drop_path)

        self.token_to_map = nn.Linear(embed_dim, f*8)  
        self.decoder_conv_proj = nn.Conv2d(f*8, f*8, 3, padding=1)

        self.dec4 = DecoderBlock(in_ch=f*8, skip_ch=f*8, out_ch=f*4)
        self.dec3 = DecoderBlock(in_ch=f*4, skip_ch=f*4, out_ch=f*2)
        self.dec2 = DecoderBlock(in_ch=f*2, skip_ch=f*2, out_ch=f)
        self.dec1 = DecoderBlock(in_ch=f, skip_ch=f, out_ch=f)

        self.final = nn.Sequential(
            nn.Conv2d(f, f, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8,f), num_channels=f),
            nn.ReLU(inplace=True),
            nn.Conv2d(f, num_classes, kernel_size=1)
        )

        self.deep_supervision = deep_supervision
        if deep_supervision:

            self.ds1 = nn.Conv2d(f, self.num_classes, 1)
            self.ds2 = nn.Conv2d(f, self.num_classes, 1)
            self.ds3 = nn.Conv2d(f*2, self.num_classes, 1)

        
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
        B, C, H, W = x.shape
        s = self.stem(x)
        e1 = self.enc1(s)          
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)         
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)         
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)          
        p4 = self.pool4(e4)        

        tokens, (Hp, Wp) = self.patch_emb(p4)     
        tokens = self.transformer(tokens)         

        proj = self.token_to_map(tokens)          
        Bp, N, D2 = proj.shape
        proj = proj.transpose(1,2).reshape(Bp, D2, Hp, Wp)  
        proj = self.decoder_conv_proj(proj)      

        p4_up = F.interpolate(proj, size=p4.shape[2:], mode='bilinear', align_corners=False)

        d4 = self.dec4(p4_up, e4)   
        d3 = self.dec3(d4, e3)     
        d2 = self.dec2(d3, e2)      
        d1 = self.dec1(d2, e1)     

        out = self.final(d1)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        if self.deep_supervision:
            aux1 = self.ds1(d1)  
            aux2 = self.ds2(d2) 
            aux3 = self.ds3(d3)  

            aux1 = F.interpolate(aux1, size=(H, W), mode='bilinear', align_corners=False)
            aux2 = F.interpolate(aux2, size=(H, W), mode='bilinear', align_corners=False)
            aux3 = F.interpolate(aux3, size=(H, W), mode='bilinear', align_corners=False)

            return out, aux1, aux2, aux3

        return out


if __name__ == "__main__":
    model = HybridTransformerUNet2D(in_channels=1, num_classes=4, base_filters=32,
                                    embed_dim=256, trans_depth=6, trans_heads=8, patch_size=2)
    x = torch.randn(2,1,224,224)
    y = model(x)
    if isinstance(y, tuple):
        print("Output shapes:", [t.shape for t in y])
    else:
        print("Output shape:", y.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNReLU(nn.Sequential):
    """
    Bloc de convolution de base.
    Utilise GroupNorm au lieu de BatchNorm.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, groups=8):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            # On divise les canaux en groupes pour la normalisation
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True)
        )


class ResidualBlock(nn.Module):
    """
    Bloc Résiduel avec SE et Dropout
    Permet au gradient de circuler plus facilement 
    ce qui permet de faire des réseaux plus profonds.
    """
    def __init__(self, in_ch, out_ch, stride=1, groups=8, dropout=0.0, use_se=True):
        super().__init__()
        self.conv1 = ConvGNReLU(in_ch, out_ch, groups=groups)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_ch != out_ch or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            )
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_ch)
        self.dropout = nn.Dropout2d(p=dropout) if dropout and dropout > 0.0 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_se:
            out = self.se(out)
        out = self.dropout(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


class SEBlock(nn.Module):
    """
    Le réseau apprend un poids pour chaque canal.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class AttentionGate(nn.Module):
    """
    Attention Gate standard modifiée pour accepter un MASQUE GUIDE.
    """
    def __init__(self, in_channels_x, in_channels_g, inter_channels):
        super().__init__()
        #transformation linéaire des entrées
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels_g, inter_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels_x, inter_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=inter_channels)
        )
        #calcul du coeff d'attention 
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()# carte de probabilité [0, 1]
        )

    def forward(self, x, g, mask=None):
        # x : Features venant de l'encodeur 
        # g : Features venant du décodeur 
        # mask : masque guide probabiliste
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.psi(g1 + x1)

        if mask is not None:
            # redimensionner le masque à la taille actuelle
            if mask.shape[2:] != psi.shape[2:]:
                mask = F.interpolate(mask, size=psi.shape[2:], mode='nearest')
            psi = psi * mask
        
        # On pondère les features de l'encodeur par cette carte d'attention
        return x * psi


class ASPP(nn.Module):
    """
    Permet au réseau de voir le contexte à plusieurs échelles sans réduire la résolution.
    """
    def __init__(self, in_channels, out_channels, dilations=(1, 6, 12, 18)):
        super().__init__()
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(
                nn.Sequential(
                    #elargir le champs visuel
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d, dilation=d, bias=False),
                    nn.GroupNorm(num_groups=1 if out_channels < 8 else 8, num_channels=out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        # Pooling global pour le contexte de l'image entière
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )
        #fusion de tout
        self.project = nn.Sequential(
            nn.Conv2d(len(dilations) * out_channels + out_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1 if out_channels < 8 else 8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        res = []
        for b in self.blocks:
            res.append(b(x))
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(gp)
        res = torch.cat(res, dim=1)
        return self.project(res)

class DecoderBlock(nn.Module):
    """
    Bloc de décodage unifié.
    """
    def __init__(self, in_ch, out_ch, use_attention=True, dropout=0.0):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.use_attention = use_attention
        if use_attention:
            self.att = AttentionGate(in_channels_x=out_ch,
                                     in_channels_g=in_ch,
                                     inter_channels=out_ch // 2 if out_ch >= 2 else 1)
        self.conv = ResidualBlock(in_ch + out_ch, out_ch, use_se=True, dropout=dropout)

    def forward(self, x, skip, mask=None):
        x = self.upsample(x)
        # Gestion des différences de taille
        if skip.shape[2:] != x.shape[2:]:
            skip = center_crop(skip, x.shape[2:])

        if self.use_attention:
            #on passe le masque à l'Attention Gate
            skip = self.att(skip, x, mask=mask)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x



def center_crop(tensor, target_spatial):
    _, _, h, w = tensor.shape
    th, tw = target_spatial
    if h == th and w == tw:
        return tensor
    top = (h - th) // 2
    left = (w - tw) // 2
    return tensor[:, :, top:top + th, left:left + tw]



class EnhancedUNetAttGate(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=4,
                 base_filters=32,
                 deep_supervision=True,
                 dropout=0.1,
                 use_attention=True):
        super().__init__()
        f = base_filters
        self.deep_supervision = deep_supervision

        # Encoder 
        self.enc1 = nn.Sequential(
            ResidualBlock(in_channels, f, use_se=True, dropout=dropout),
            ResidualBlock(f, f, use_se=True, dropout=dropout)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            ResidualBlock(f, f * 2, use_se=True, dropout=dropout),
            ResidualBlock(f * 2, f * 2, use_se=True, dropout=dropout)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            ResidualBlock(f * 2, f * 4, use_se=True, dropout=dropout),
            ResidualBlock(f * 4, f * 4, use_se=True, dropout=dropout)
        )
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            ResidualBlock(f * 4, f * 8, use_se=True, dropout=dropout),
            ResidualBlock(f * 8, f * 8, use_se=True, dropout=dropout)
        )
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck: ASPP
        self.bottleneck = nn.Sequential(
            ResidualBlock(f * 8, f * 16, use_se=True, dropout=dropout),
            ASPP(f * 16, f * 16 // 2)
        )

        # Decoder 
        self.dec4 = DecoderBlock(in_ch=f * 16 // 2, out_ch=f * 8, use_attention=use_attention, dropout=dropout)
        self.dec3 = DecoderBlock(in_ch=f * 8, out_ch=f * 4, use_attention=use_attention, dropout=dropout)
        self.dec2 = DecoderBlock(in_ch=f * 4, out_ch=f * 2, use_attention=use_attention, dropout=dropout)
        self.dec1 = DecoderBlock(in_ch=f * 2, out_ch=f, use_attention=use_attention, dropout=dropout)

        # Petites convolutions 1x1 qui prédisent la segmentation à basse résolution
        if self.deep_supervision:
            self.ds3 = nn.Conv2d(f * 4, num_classes, kernel_size=1)
            self.ds2 = nn.Conv2d(f * 2, num_classes, kernel_size=1)
            self.ds1 = nn.Conv2d(f, num_classes, kernel_size=1)

        self.final = nn.Sequential(
            nn.Conv2d(f, f, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8 if f >= 8 else 1, num_channels=f),
            nn.ReLU(inplace=True),
            nn.Conv2d(f, num_classes, kernel_size=1)
        )

        self.initialize()

    def forward(self, x, masks=None):
        """
        Args:
            x: Image 
            masks: Guides probabilistes 
        """
        # gestion des masques pour l'attention
        m1, m2, m3, m4 = torch.chunk(masks, 4, dim=1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder avec Deep Guidance
        d4 = self.dec4(b,  e4, mask=m4)
        d3 = self.dec3(d4, e3, mask=m3)
        d2 = self.dec2(d3, e2, mask=m2)
        d1 = self.dec1(d2, e1, mask=m1)

        out = self.final(d1)

        if self.deep_supervision:
            # On retourne toutes les sorties pour calculer la loss sur chaque échelle
            ds3 = F.interpolate(self.ds3(d3), size=x.shape[2:], mode='bilinear', align_corners=False)
            ds2 = F.interpolate(self.ds2(d2), size=x.shape[2:], mode='bilinear', align_corners=False)
            ds1 = F.interpolate(self.ds1(d1), size=x.shape[2:], mode='bilinear', align_corners=False)
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
            return out, ds1, ds2, ds3

        return F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)


    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)




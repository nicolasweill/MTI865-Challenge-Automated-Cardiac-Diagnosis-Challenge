"""
UNet++ (Nested U-Net) with Attention Gates, SE blocks, ASPP bottleneck and Deep Supervision
Optimized defaults for ACDC cardiac MRI segmentation (in_channels=1, num_classes=4)

Features:
- conv blocks with GroupNorm and optional SE (Squeeze-and-Excitation)
- Attention gates on skip connections
- ASPP module at bottleneck for multi-scale context
- Upsample + Conv (no transposed conv) to avoid checkerboard artifacts
- Deep supervision outputs at multiple decoder depths (return list of outputs)
- Designed for 2D slices (ACDC typical setup)

Usage:
    model = UNetPP_Attn_SE_ASPP(in_channels=1, num_classes=4)
    outputs = model(x)

Returns:
    If deep_supervision=True: list of torch tensors [out_d1, out_d2, out_d3] (each upsampled to input size)
    else: single tensor out

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- Basic building blocks ----------------------------
class ConvGNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, groups=8, use_se=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False)
        # GroupNorm: choose groups such that group size ~ 16-32 features, but not larger than channels
        gn_groups = min(groups, out_ch) if out_ch >= groups else 1
        self.gn = nn.GroupNorm(gn_groups, out_ch)
        self.act = nn.ReLU(inplace=True)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        if self.use_se:
            x = self.se(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        r = max(1, min(reduction, channels // 2))
        self.fc1 = nn.Conv2d(channels, channels // r, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // r, channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        z = x.mean(dim=(2, 3), keepdim=True)
        z = self.act(self.fc1(z))
        z = self.sig(self.fc2(z))
        return x * z


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=False):
        super().__init__()
        self.conv1 = ConvGNReLU(in_ch, out_ch, use_se=use_se)
        self.conv2 = ConvGNReLU(out_ch, out_ch, use_se=use_se)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AttentionGate(nn.Module):
    """Attention Gate from Attention U-Net (spatial attention using gating signal)
    x: skip connection feature (from encoder)
    g: gating feature (from decoder)
    returns: filtered skip features
    """
    def __init__(self, in_ch, gating_ch, inter_ch):
        super().__init__()
        self.W_x = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, inter_ch), inter_ch)
        )
        self.W_g = nn.Sequential(
            nn.Conv2d(gating_ch, inter_ch, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, inter_ch), inter_ch)
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, g):
        # x: encoder feature, g: decoder feature
        x1 = self.W_x(x)
        g1 = self.W_g(g)
        psi = self.psi(x1 + g1)
        return x * psi


class UpsampleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvGNReLU(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


# ---------------------------- ASPP Module ----------------------------
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256, rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        # 1x1 conv
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True)
        ))
        # atrous convs
        for r in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=r, dilation=r, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.ReLU(inplace=True)
            ))
        # image pooling
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (2 + len(rates)), out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        res = []
        for b in self.branches:
            res.append(b(x))
        img = self.image_pool(x)
        img = F.interpolate(img, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(img)
        x = torch.cat(res, dim=1)
        x = self.project(x)
        return x


# ---------------------------- UNet++ with Attention and Deep Supervision ----------------------------
class UNetPP_Attn_SE_ASPP(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, filters=(32, 64, 128, 256, 512), use_se=True, deep_supervision=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_se = use_se
        self.deep_supervision = deep_supervision

        f1, f2, f3, f4, f5 = filters

        # Encoder
        self.conv00 = ConvBlock(in_channels, f1, use_se=use_se)
        self.pool0 = nn.MaxPool2d(2)

        self.conv10 = ConvBlock(f1, f2, use_se=use_se)
        self.pool1 = nn.MaxPool2d(2)

        self.conv20 = ConvBlock(f2, f3, use_se=use_se)
        self.pool2 = nn.MaxPool2d(2)

        self.conv30 = ConvBlock(f3, f4, use_se=use_se)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck (ASPP)
        self.conv40 = ConvBlock(f4, f5, use_se=use_se)
        self.aspp = ASPP(f5, out_ch=f5//2, rates=(6, 12, 18))
        self.bottle_proj = nn.Conv2d(f5//2, f5, kernel_size=1, bias=False)

        # Decoder nested nodes (UNet++) X_ij style
        # Level 3
        self.up31 = UpsampleConv(f5, f4)
        self.att31 = AttentionGate(f4, f4, inter_ch=f4//2)
        self.conv31 = ConvBlock(f4*2, f4, use_se=use_se)

        # Level 2
        self.up21 = UpsampleConv(f4, f3)
        self.att21 = AttentionGate(f3, f3, inter_ch=f3//2)
        self.conv21 = ConvBlock(f3*2, f3, use_se=use_se)

        self.up22 = UpsampleConv(f3, f3)
        self.conv22 = ConvBlock(f3*2, f3, use_se=use_se)

        # Level 1
        self.up11 = UpsampleConv(f3, f2)
        self.att11 = AttentionGate(f2, f2, inter_ch=f2//2)
        self.conv11 = ConvBlock(f2*2, f2, use_se=use_se)

        self.up12 = UpsampleConv(f2, f2)
        self.conv12 = ConvBlock(f2*2, f2, use_se=use_se)

        self.up13 = UpsampleConv(f2, f2)
        self.conv13 = ConvBlock(f2*2, f2, use_se=use_se)

        # Level 0
        self.up01 = UpsampleConv(f2, f1)
        self.att01 = AttentionGate(f1, f1, inter_ch=f1//2)
        self.conv01 = ConvBlock(f1*2, f1, use_se=use_se)

        self.up02 = UpsampleConv(f1, f1)
        self.conv02 = ConvBlock(f1*2, f1, use_se=use_se)

        self.up03 = UpsampleConv(f1, f1)
        self.conv03 = ConvBlock(f1*2, f1, use_se=use_se)

        self.up04 = UpsampleConv(f1, f1)
        self.conv04 = ConvBlock(f1*2, f1, use_se=use_se)

        # Final classifiers for deep supervision
        self.final1 = nn.Conv2d(f1, num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(f1, num_classes, kernel_size=1)
        self.final3 = nn.Conv2d(f1, num_classes, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        x00 = self.conv00(x)          # size: f1
        x10 = self.conv10(self.pool0(x00))
        x20 = self.conv20(self.pool1(x10))
        x30 = self.conv30(self.pool2(x20))

        # Bottleneck
        x40 = self.conv40(self.pool3(x30))
        x40 = self.aspp(x40)
        x40 = self.bottle_proj(x40)

        # Decoder (nested) - building progressively
        # Level 3 (first decoder depth)
        d31 = self.up31(x40)
        # attention over encoder x30 using decoder gating d31
        x30_att = self.att31(x30, d31)
        x31 = self.conv31(torch.cat([d31, x30_att], dim=1))

        # Level 2
        d21 = self.up21(x31)
        x20_att = self.att21(x20, d21)
        x21 = self.conv21(torch.cat([d21, x20_att], dim=1))

        d22 = self.up22(x21)
        # skip connection from x10 and x11 composed later
        x22 = self.conv22(torch.cat([d22, x10], dim=1))

        # Level 1
        d11 = self.up11(x22)
        x10_att = self.att11(x10, d11)
        x11 = self.conv11(torch.cat([d11, x10_att], dim=1))

        d12 = self.up12(x11)
        x12 = self.conv12(torch.cat([d12, x00], dim=1))

        d13 = self.up13(x12)
        x13 = self.conv13(torch.cat([d13, x00], dim=1))

        # Level 0
        d01 = self.up01(x13)
        x00_att = self.att01(x00, d01)
        x01 = self.conv01(torch.cat([d01, x00_att], dim=1))

        d02 = self.up02(x01)
        x02 = self.conv02(torch.cat([d02, x00], dim=1))

        d03 = self.up03(x02)
        x03 = self.conv03(torch.cat([d03, x00], dim=1))

        d04 = self.up04(x03)
        x04 = self.conv04(torch.cat([d04, x00], dim=1))

        # Deep supervision outputs (take three different decoder depths)
        out1 = self.final1(x01)
        out2 = self.final2(x02)
        out3 = self.final3(x04)

        # Upsample all to input size
        out1_up = F.interpolate(out1, size=x.shape[2:], mode='bilinear', align_corners=False)
        out2_up = F.interpolate(out2, size=x.shape[2:], mode='bilinear', align_corners=False)
        out3_up = F.interpolate(out3, size=x.shape[2:], mode='bilinear', align_corners=False)

        if self.deep_supervision:
            # return list for supervision (combine in loss externally)
            return [out1_up, out2_up, out3_up]
        else:
            return out3_up


if __name__ == '__main__':
    # quick sanity test
    model = UNetPP_Attn_SE_ASPP(in_channels=1, num_classes=4, deep_supervision=True)
    x = torch.randn(2, 1, 256, 256)
    outs = model(x)
    assert isinstance(outs, list) and len(outs) == 3
    for o in outs:
        print(o.shape)

    print('Model ready')

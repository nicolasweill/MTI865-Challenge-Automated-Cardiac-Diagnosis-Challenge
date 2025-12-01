import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    Bloc Résiduel : Conv -> Norm -> LeakyReLU -> Conv -> Norm -> LeakyReLU + Residual
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.norm1 = nn.InstanceNorm2d(out_channels) # InstanceNorm > BatchNorm pour le médical
        self.act1  = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.act2  = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        # Ajustement pour la connexion résiduelle si les dimensions changent
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.InstanceNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        
        out += residual # Addition résiduelle
        out = self.act2(out)
        return out

class ImprovedUNet(nn.Module):
    def __init__(self, num_classes, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # Filtres de base (commencer à 32 est ok, 64 capture plus de détails)
        filters = [32, 64, 128, 256, 512]

        # -------- Encoder (Downsampling) --------
        # On utilise stride=2 dans le premier conv du bloc au lieu de MaxPool 
        # pour préserver l'information spatiale.
        self.enc1 = ResBlock(1, filters[0])
        self.enc2 = ResBlock(filters[0], filters[1], stride=2)
        self.enc3 = ResBlock(filters[1], filters[2], stride=2)
        self.enc4 = ResBlock(filters[2], filters[3], stride=2)
        
        # -------- Bottleneck --------
        self.bottleneck = ResBlock(filters[3], filters[4], stride=2)

        # -------- Decoder (Upsampling) --------
        # ConvTranspose est bien, mais Bilinear + Conv réduit les artefacts en damier
        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.dec4 = ResBlock(filters[3] + filters[3], filters[3]) # Concaténation double les canaux d'entrée

        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.dec3 = ResBlock(filters[2] + filters[2], filters[2])

        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.dec2 = ResBlock(filters[1] + filters[1], filters[1])

        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.dec1 = ResBlock(filters[0] + filters[0], filters[0])

        # -------- Deep Supervision Heads --------
        # Têtes de segmentation à différentes échelles
        self.final1 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(filters[1], num_classes, kernel_size=1)
        self.final3 = nn.Conv2d(filters[2], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)       # [B, 32, H, W]
        e2 = self.enc2(e1)      # [B, 64, H/2, W/2]
        e3 = self.enc3(e2)      # [B, 128, H/4, W/4]
        e4 = self.enc4(e3)      # [B, 256, H/8, W/8]

        # Bottleneck
        b = self.bottleneck(e4) # [B, 512, H/16, W/16]

        # Decoder 4
        d4 = self.up4(b)
        # Gestion des légers décalages de dimension si H/W ne sont pas puissances de 2
        if d4.shape != e4.shape:
             d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        # Decoder 3
        d3 = self.up3(d4)
        if d3.shape != e3.shape:
             d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        # Decoder 2
        d2 = self.up2(d3)
        if d2.shape != e2.shape:
             d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        # Decoder 1
        d1 = self.up1(d2)
        if d1.shape != e1.shape:
             d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Sortie finale (Résolution native)
        out1 = self.final1(d1)

        # Si Deep Supervision activée, on retourne une liste de masques
        if self.deep_supervision and self.training:
            out2 = self.final2(d2)
            out3 = self.final3(d3)
            # On retourne [Full_Res, Half_Res, Quarter_Res]
            return [out1, out2, out3]
        
        # En mode inférence, on ne retourne que le masque haute résolution
        return out1

# Test rapide pour vérifier les dimensions
if __name__ == "__main__":
    model = ImprovedUNet(num_classes=4) # 0:BG, 1:RV, 2:Myo, 3:LV
    x = torch.randn(1, 1, 224, 224)
    outputs = model(x)
    print(f"Sortie principale shape: {outputs[0].shape}") # Doit être [1, 4, 224, 224]
    print(f"Sortie supervision 2 shape: {outputs[1].shape}") # Doit être [1, 4, 112, 112]
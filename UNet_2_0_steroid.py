import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels):
    """
    Bloc standard de U-Net composé de deux convolutions successives
       
    Args:
        in_channels (int): Nombre de canaux entrant dans le bloc
        out_channels (int): Nombre de filtres en sortie
    
    Returns:
        nn.Sequential: conteneur séquentiel des couches
    """
    return nn.Sequential(
        # 1ere convolution : change le nombre de canaux 
        # padding=1 conserve la taille de l'image 
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),#normaliser pour accélerer l'apprentissage
        nn.ReLU(inplace=True),#fct d'activation ReLU

        # 2eme convolution : affine les features
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet_2_dope(nn.Module):
    """
    
    Ce modèle prend en entrée une concaténation de l'image IRM et des masques guides
    
    Args:
        num_classes (int): Nombre de classes à prédire 
    
    """
    def __init__(self, num_classes):
        super().__init__()

        # entrée -->32 filtres
        self.enc1 = conv_block(5, 32)
        self.pool1 = nn.MaxPool2d(2)# Divise la taille par 2 (256 -> 128)

        #32 -> 64 filtres
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)# 128 -> 64

        # 64 -> 128 filtres
        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)# 64 -> 32

        #  128 -> 256 filtres
        self.enc4 = conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(2)# 32 -> 16

        # 256 -> 512 filtres. Taille spatiale minimale (16x16).
        self.bottleneck = conv_block(256, 512)

        #remontée vers 32x32
        # opération inverse de la convolution/pooling,
        # apprend à agrandir l'image.        
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = conv_block(512, 256)

        #remontée vers 64x64
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = conv_block(256, 128)

        #remontée vers 128x128
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(128, 64)

        #remontée vers 256x256
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = conv_block(64, 32)

        # Convolution 1x1 pour projeter les 32 derniers features vers les 4 classes de sortie.
        # On n'utilise pas de Softmax ici si on utilise CrossEntropyLoss plus tard (qui l'inclut).
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        """        
        Args:
            x (torch.Tensor): Contient l'image concaténée avec les guides
        
        Returns:
            torch.Tensor: Carte de segmentation
        """
        #encodage
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # point le plus profond du réseau.
        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)#agrandir le bottleneck
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.final(d1)

        return F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)

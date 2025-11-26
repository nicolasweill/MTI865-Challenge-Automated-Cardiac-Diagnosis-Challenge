import torch

def getTargetSegmentation(batch):
    """
    Convertit les masques normalisés (0, 0.33, 0.66, 1) en classes entières (0, 1, 2, 3).
    Cette fonction remplace la logique complexe de utils.py pour être plus directe.
    """
    # Le dénominateur correspond au pas entre les classes normalisées
    # (0, 0.33, 0.66, 1.0) -> (0, 1, 2, 3)
    denom = 0.33333334
    return (batch / denom).round().long().squeeze()

def dice_score(pred, target, num_classes, eps=1e-7):
    """
    Calcule le Dice score par classe de manière compatible avec PyTorch (MPS/CPU).
    
    Args:
        pred (Tensor): Prédictions du modèle [Batch, Classes, H, W] (Logits ou Softmax)
        target (Tensor): Vérité terrain [Batch, H, W] (Classes 0,1,2...)
        num_classes (int): Nombre de classes
        eps (float): Petite valeur pour éviter la division par zéro
        
    Returns:
        Tensor: Un tenseur de taille [num_classes] contenant le score Dice pour chaque classe.
    """
    # On récupère la classe prédite pour chaque pixel (Argmax)
    # pred passe de [B, C, H, W] -> [B, H, W]
    pred_classes = torch.argmax(pred, dim=1)
    
    dice_per_class = []
    
    # On itère sur chaque classe pour calculer son score spécifique (One-vs-Rest)
    for c in range(num_classes):
        # Création des masques binaires pour la classe c
        pred_c = (pred_classes == c).float()
        target_c = (target == c).float()
        
        # Calcul de l'intersection et de l'union
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        # Formule du Dice : 2*Inter / (Union)
        dice = (2 * intersection + eps) / (union + eps)
        dice_per_class.append(dice)
        
    # On empile les résultats pour renvoyer un seul tenseur
    return torch.stack(dice_per_class)

import os
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import argparse

def elastic_transform(image, mask, alpha, sigma, random_state=42):
    """
    Applique une déformation élastique à une image et son masque 
    
    Args:
        image (numpy array):image d'entrée, niveaux de gris ou RGB
        mask (numpy array): masque de segmentation, étiquettes de classes 0, 1, 2, 3
        alpha (float): intensité de la déformation
        sigma (float): lissage de la déformation
        random_state (int): Graine fixe

    Returns:
        tuple: (distorted_image, distorted_mask) les versions déformées
    
    """
    # générateur aléatoire si il est a None
    if random_state is None:
        random_state = np.random.RandomState(None)
        
    # graine pour créer le générateur si c'est un int donné
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    shape = image.shape
    
    # Génération du champ de vecteurs aléatoire 
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))#grille de coordonnées X Y de chaque pixel de l'image
    
    # calcul des coordonnées en ajoutant le déplacement
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    #interpolation
    distorted_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    #pas d'interpolation, ordre 0 pour le plus proche voisin
    distorted_mask = map_coordinates(mask, indices, order=0, mode='reflect').reshape(shape)

    return distorted_image, distorted_mask


def run_augmentation(
    source_dir='./Data_augment/train', 
    target_dir='./Data_augment/PreAugmented_2',
    do_flip=True,        
    do_rotate=True,     
    n_elastic=3           
):
    """
    Exécute le pipeline complet d'augmentation de données
    
    Lit un dataset, applique des transformations géométriques et sauvegarde, le but est de multiplier la taille du dataset d'entrainement.
    
    Args:
        source_dir (str): Dossier entrainement avec 'Img' (images originales) et 'GT' (masques)
        target_dir (str): Dossier de sortie
        do_flip (bool):   Activer le miroir horizontal
        do_rotate (bool): Activer les rotations
        n_elastic (int):  Nombre de déformations élastiques à générer par image 
    """
    
    if not os.path.exists(target_dir):
        os.makedirs(os.path.join(target_dir, 'Img'))
        os.makedirs(os.path.join(target_dir, 'GT'))
    else:
        print(f"le dossier {target_dir} existe déjà.")

    img_dir = os.path.join(source_dir, 'Img')
    mask_dir = os.path.join(source_dir, 'GT')
    
    files = sorted([f for f in os.listdir(img_dir) if not f.startswith('.')])#liste tous les fichiers sauf ceux cachées 
    
    print(f" Démarrage de l'augmentation")
    print(f" Source: {len(files)} images")
    print(f" Paramètres: Flip={do_flip}, Rotate={do_rotate}, Elastic x{n_elastic}")

    count_total = 0

    # affichage de la barre de progression
    for f in tqdm(files, desc="Processing"):
        
        img_path = os.path.join(img_dir, f)
        mask_path = os.path.join(mask_dir, f)
        
        try:
            img = Image.open(img_path)
            mask = Image.open(mask_path)
        except Exception as e:
            print(f"Erreur lecture {f}: {e}")
            continue

        #fonction interne de sauvegarde
        def save(im, mk, suffix):
            name, ext = os.path.splitext(f)
            new_name = f"{name}_{suffix}{ext}"
            #sauvegarde de l'image
            im.save(os.path.join(target_dir, 'Img', new_name))
            #sauvegarde du masque
            mk.save(os.path.join(target_dir, 'GT', new_name))
            return 1

        count_total += save(img, mask, "orig")

        if do_flip:
            #miroir
            save(ImageOps.mirror(img), ImageOps.mirror(mask), "flipH")
            count_total += 1

        if do_rotate:
            # rotation légère aléatoire entre -10 et 10 degrés
            angle = np.random.uniform(-10, 10)
            save(img.rotate(angle, resample=Image.BILINEAR), 
                 mask.rotate(angle, resample=Image.NEAREST), "rot")
            count_total += 1

        if n_elastic > 0:
            img_np = np.array(img)
            mask_np = np.array(mask)
            
            for i in range(n_elastic):
                # Paramètres aléatoires pour unicité

                alpha_rnd = np.random.uniform(20, 45)#intensité
                sigma_rnd = np.random.uniform(3.5, 5.5)#lissage
                
                img_aug, mask_aug = elastic_transform(
                    img_np, mask_np, alpha=alpha_rnd, sigma=sigma_rnd
                )
                
                save(Image.fromarray(img_aug), 
                     Image.fromarray(mask_aug.astype(np.uint8)), 
                     f"elas_{i+1}")
                count_total += 1

    print(f"fin,  {count_total} images générées dans {target_dir}")

if __name__ == "__main__":
    run_augmentation()
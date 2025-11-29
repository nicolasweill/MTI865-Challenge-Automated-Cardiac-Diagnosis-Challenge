import os
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import argparse

def elastic_transform(image, mask, alpha, sigma, random_state=42):
    """
    Applique une déformation élastique.
    Gère random_state que ce soit None, un entier (seed), ou un objet RandomState.
    """
    # 1. Si c'est None -> On crée un générateur aléatoire
    if random_state is None:
        random_state = np.random.RandomState(None)
        
    # 2. Si c'est un Entier (ex: 42) -> On l'utilise comme graine pour créer le générateur
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    shape = image.shape
    
    # Génération du champ de vecteurs aléatoire (utilise random_state.rand qui existe maintenant forcément)
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    # Interpolation
    distorted_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    distorted_mask = map_coordinates(mask, indices, order=0, mode='reflect').reshape(shape)

    return distorted_image, distorted_mask


def run_augmentation(
    source_dir='./Data_augment/train', 
    target_dir='./Data_augment/PreAugmented_2',
    do_flip=True,        # Activer/Désactiver Flip
    do_rotate=True,       # Activer/Désactiver Rotation
    n_elastic=3           # Nombre de versions élastiques par image
):
    """
    Génère un dataset augmenté avec contrôle fin des transformations.
    """
    
    # 1. Préparation des dossiers
    if not os.path.exists(target_dir):
        os.makedirs(os.path.join(target_dir, 'Img'))
        os.makedirs(os.path.join(target_dir, 'GT'))
    else:
        print(f" Attention : Le dossier {target_dir} existe déjà.")

    img_dir = os.path.join(source_dir, 'Img')
    mask_dir = os.path.join(source_dir, 'GT')
    
    # Filtrer les fichiers cachés
    files = sorted([f for f in os.listdir(img_dir) if not f.startswith('.')])
    
    print(f" Démarrage de l'augmentation...")
    print(f" Source: {len(files)} images")
    print(f" Paramètres: Flip={do_flip}, Rotate={do_rotate}, Elastic x{n_elastic}")

    count_total = 0

    for f in tqdm(files, desc="Processing"):
        # Chemins
        img_path = os.path.join(img_dir, f)
        mask_path = os.path.join(mask_dir, f)
        
        try:
            img = Image.open(img_path)
            mask = Image.open(mask_path)
        except Exception as e:
            print(f"Erreur lecture {f}: {e}")
            continue

        # Helper de sauvegarde
        def save(im, mk, suffix):
            name, ext = os.path.splitext(f)
            new_name = f"{name}_{suffix}{ext}"
            im.save(os.path.join(target_dir, 'Img', new_name))
            mk.save(os.path.join(target_dir, 'GT', new_name))
            return 1

        # --- A. ORIGINAL (Toujours) ---
        count_total += save(img, mask, "orig")

        # --- B. FLIP (Optionnel) ---
        # Uniquement vertical pour respecter l'anatomie (ou horizontal si vous le souhaitez finalement)
        if do_flip:
            # Flip Miroir (Horizontal) - À utiliser avec prudence en médical
            save(ImageOps.mirror(img), ImageOps.mirror(mask), "flipH")
            count_total += 1

        # --- C. ROTATION (Optionnel) ---
        if do_rotate:
            # Rotation légère aléatoire (+/- 10 deg)
            angle = np.random.uniform(-10, 10)
            save(img.rotate(angle, resample=Image.BILINEAR), 
                 mask.rotate(angle, resample=Image.NEAREST), "rot")
            count_total += 1

        # --- D. ELASTIC (Multiple & Aléatoire) ---
        if n_elastic > 0:
            img_np = np.array(img)
            mask_np = np.array(mask)
            
            for i in range(n_elastic):
                # Paramètres aléatoires pour varier les effets
                # Alpha : Intensité (20 à 40)
                # Sigma : Lissage (3 à 5)
                # Ces plages sont idéales pour des images 256x256
                alpha_rnd = np.random.uniform(20, 45)
                sigma_rnd = np.random.uniform(3.5, 5.5)
                
                img_aug, mask_aug = elastic_transform(
                    img_np, mask_np, alpha=alpha_rnd, sigma=sigma_rnd
                )
                
                # Suffixe numéroté : elas_1, elas_2, etc.
                save(Image.fromarray(img_aug), 
                     Image.fromarray(mask_aug.astype(np.uint8)), 
                     f"elas_{i+1}")
                count_total += 1

    print(f" Terminé ! {count_total} images générées dans {target_dir}")

if __name__ == "__main__":
    # Permet d'appeler le script avec des arguments si besoin
    run_augmentation()
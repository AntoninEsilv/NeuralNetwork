import os
import random
import shutil
from pathlib import Path

# --- Paramètres à adapter ---
source_dir = Path("seg_test")        # Ton dossier contenant les sous-dossiers de classes
output_dir = Path("dataset_split")   # Nouveau dossier qui contiendra train/ et test/
train_ratio = 0.70                   # 70% train, 30% test
seed = 42                            # Graine aléatoire pour reproductibilité
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Optionnel : limiter le nombre d'images max par classe (ex: 500, ou None pour tout prendre)
max_per_class = None

random.seed(seed)

# Vérification du dossier source
if not source_dir.exists():
    raise FileNotFoundError(f"Le dossier source '{source_dir}' est introuvable.")

# Récupération des sous-dossiers de classes
class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
class_dirs.sort()

print(f"Classes détectées ({len(class_dirs)}) : {[d.name for d in class_dirs]}\n")

total_train = 0
total_test = 0

for class_dir in class_dirs:
    class_name = class_dir.name

    # Liste des images valides de la classe
    images = [
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ]
    images.sort()
    random.shuffle(images)

    if max_per_class is not None:
        images = images[:max_per_class]

    n_total = len(images)
    n_train = int(n_total * train_ratio)

    train_images = images[:n_train]
    test_images = images[n_train:]

    # Création des dossiers de destination
    train_class_dir = output_dir / "train" / class_name
    test_class_dir = output_dir / "test" / class_name
    train_class_dir.mkdir(parents=True, exist_ok=True)
    test_class_dir.mkdir(parents=True, exist_ok=True)

    # Copie des fichiers
    for img_path in train_images:
        shutil.copy2(img_path, train_class_dir / img_path.name)

    for img_path in test_images:
        shutil.copy2(img_path, test_class_dir / img_path.name)

    total_train += len(train_images)
    total_test += len(test_images)

    print(
        f"{class_name:<12} | Total: {n_total:>4} | "
        f"Train (70%): {len(train_images):>4} | Test (30%): {len(test_images):>4}"
    )

print("-" * 58)
print(f"Total général | Train: {total_train} | Test: {total_test} | Somme: {total_train + total_test}")
print(f"\nNouveau dataset prêt dans : '{output_dir.resolve()}'")
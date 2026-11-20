# NeuralNetwork
Projet de classification de Neural Network

Classification des 10 classes de Fashion-MNIST, sur un sous-ensemble de 10 000 images
(`dataset/fashion-mnist_test.csv`) pour limiter le temps de calcul.

## Fichiers

| Fichier | Rôle |
|---|---|
| `Separation.py` | Découpage commun : 70 % train / 30 % test, stratifié, seed 42 -> `dataset_split/` |
| `network.py` | CNN (Keras) entraîné from scratch |
| `densenet_transfer.py` | Transfer learning DenseNet121 pré-entraîné sur ImageNet (PyTorch) |
| `figures/` | Courbes d'apprentissage et matrices de confusion |

Les deux modèles utilisent le même découpage : 5 600 train / 1 400 validation / 3 000 test.

## Transfer learning DenseNet121

```
python Separation.py          # si dataset_split/ n'existe pas
python densenet_transfer.py   # ~5 min sur RTX 2060 (--quick pour un test rapide)
```

- Images 28x28 agrandies en 224x224 et dupliquées sur 3 canaux, normalisation ImageNet
- Phase 1 : backbone gelé, seule la nouvelle couche de sortie (10 classes) est entraînée (3 époques)
- Phase 2 : fine-tuning de tout le réseau, lr = 1e-4 (8 époques)

Résultat : **92,2 % d'accuracy sur le test** (3 000 images). Classe la plus difficile : Shirt.
Détail dans `train_log.txt`.

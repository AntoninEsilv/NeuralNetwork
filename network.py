import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

chemin = Path(__file__).resolve().parent / "dataset" / "fashion-mnist_test.csv"
# https://www.kaggle.com/datasets/zalando-research/fashionmnist

df = pd.read_csv(chemin)
X = df.drop("label", axis=1).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

"""
Labels
Each training and test example is assigned to one of the following labels:
0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot
"""

#CNN
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# 1. Normalisation des pixels rbg->[0, 1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# 2. Transformation du vecteur plat de 784 valeurs en image 28 x 28 x 1
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

y_train = y_train.astype("int64")
y_test = y_test.astype("int64")

print("Train :", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)

# 3. Construction du CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(28, 28, 1)
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.summary()

# 4. Configuration de l'apprentissage
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"],)

# 5. Entraînement du modèle
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=3,restore_best_weights=True,)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.30,
    epochs=20,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1,
)

# 6. Évaluation sur les données de test
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# 7. Prédiction sur une image
predictions = model.predict(X_test[:10], verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

for index, predicted_label in enumerate(predicted_labels):
    real_label = y_test[index]
    print(
        f"Image {index}: predicted={class_names[predicted_label]}, "
        f"real={class_names[real_label]}"
    )


#transfert learning approach


import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

tf.random.set_seed(42)

IMG_SIZE = 96
BATCH_SIZE = 64
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 10

X_train_tl = X_train
X_test_tl = X_test

base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = keras.Input(shape=(28, 28, 1))
x = layers.Resizing(IMG_SIZE, IMG_SIZE)(inputs)
x = layers.Concatenate()([x, x, x])
x = layers.Rescaling(scale=2.0, offset=-1.0)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation="softmax")(x)

tl_model = keras.Model(inputs, outputs, name="mobilenetv2_transfer")
tl_model.summary()

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

tl_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history_phase1 = tl_model.fit(
    X_train_tl, y_train,
    validation_split=0.3,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS_PHASE1,
    callbacks=[early_stop],
    verbose=2,
)

base_model.trainable = True
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
tl_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history_phase2 = tl_model.fit(
    X_train_tl, y_train,
    validation_split=0.3,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS_PHASE2,
    callbacks=[early_stop],
    verbose=2,
)

tl_loss, tl_acc = tl_model.evaluate(X_test_tl, y_test, verbose=0)
print(f"\n[Transfer learning] Test loss : {tl_loss:.4f} | Test accuracy : {tl_acc:.4f}")

y_pred_tl = np.argmax(tl_model.predict(X_test_tl, verbose=0), axis=1)
print(classification_report(y_test, y_pred_tl, target_names=class_names,
                            digits=3, zero_division=0))
print("Matrice de confusion (lignes = vraie classe, colonnes = prédiction) :")
with pd.option_context("display.width", 200, "display.max_columns", None):
    print(pd.DataFrame(confusion_matrix(y_test, y_pred_tl),
                       index=class_names, columns=class_names))

acc = history_phase1.history["accuracy"] + history_phase2.history["accuracy"]
val_acc = history_phase1.history["val_accuracy"] + history_phase2.history["val_accuracy"]
n1 = len(history_phase1.history["accuracy"])

plt.figure(figsize=(7, 4))
plt.plot(acc, label="Train")
plt.plot(val_acc, label="Validation")
plt.axvline(n1 - 0.5, color="gray", linestyle="--", label="Début du fine-tuning")
plt.title("Transfer learning (MobileNetV2) : accuracy")
plt.xlabel("Époque")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
courbes_path = Path(__file__).resolve().parent / "transfer_learning_curves.png"
plt.savefig(courbes_path)
print(f"Courbes enregistrées dans {courbes_path}")

print("\n=== Comparaison sur le même test set ===")
print(f"CNN from scratch  : accuracy = {test_accuracy:.4f} | {model.count_params():,} paramètres")
print(f"Transfer learning : accuracy = {tl_acc:.4f} | {tl_model.count_params():,} paramètres")
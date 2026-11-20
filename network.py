import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

chemin = Path(__file__).resolve().parent / "dataset" / "fashion-mnist_test.csv"

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
    validation_split=0.20,
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



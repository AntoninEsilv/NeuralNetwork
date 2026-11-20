import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#CNN



#transfert learning approach



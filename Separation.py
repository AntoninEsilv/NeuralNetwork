from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


base_dir = Path(__file__).resolve().parent
source_csv = base_dir / "dataset" / "fashion-mnist_test.csv"
output_dir = base_dir / "dataset_split"
train_ratio = 0.70
seed = 42

df = pd.read_csv(source_csv)

if "label" not in df.columns:
    raise ValueError("Le CSV doit contenir une colonne 'label'.")

pixel_columns = [column for column in df.columns if column != "label"]
if len(pixel_columns) != 28 * 28:
    raise ValueError(
        f"Le CSV doit contenir 784 pixels, mais {len(pixel_columns)} ont ete trouves."
    )

train_df, test_df = train_test_split(
    df,
    train_size=train_ratio,
    random_state=seed,
    shuffle=True,
    stratify=df["label"],
)

output_dir.mkdir(parents=True, exist_ok=True)
train_path = output_dir / "train.csv"
test_path = output_dir / "test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Train : {len(train_df)} images (70%) -> {train_path}")
print(f"Test  : {len(test_df)} images (30%) -> {test_path}")

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_image_files(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def load_and_preprocess_image(image_path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def build_augmentation_pipeline() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.12),
            tf.keras.layers.RandomZoom(0.10),
        ],
        name="augmentation",
    )


def train_test_split_records(
    records: Iterable,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[list, list]:
    records_list = list(records)
    train_items, test_items = train_test_split(
        records_list,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return train_items, test_items


def build_datasets(
    data_dir: str | Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 16,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    files = discover_image_files(data_dir)
    if not files:
        raise FileNotFoundError(
            f"No image files found in '{data_dir}'. Place data in class-wise folders, e.g. data/raw/species_a/*.jpg"
        )

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    normalization = tf.keras.layers.Rescaling(1.0 / 255)
    augmentation = build_augmentation_pipeline()

    train_ds = train_ds.map(lambda x, y: (augmentation(normalization(x), training=True), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    return train_ds, val_ds, class_names


def load_locations_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required_columns = {"species", "latitude", "longitude"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")
    return frame

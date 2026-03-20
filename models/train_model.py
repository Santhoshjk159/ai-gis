from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from src.preprocessing import build_datasets
from src.utils import (
    compute_classification_metrics,
    ensure_dir,
    plot_and_save_confusion_matrix,
    save_json,
    set_seed,
    utc_timestamp,
)


def build_model(
    num_classes: int,
    image_size: tuple[int, int] = (224, 224),
    base_model_name: str = "efficientnetb0",
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    input_layer = tf.keras.Input(shape=(image_size[0], image_size[1], 3))

    if base_model_name.lower() == "resnet50":
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            input_tensor=input_layer,
            weights="imagenet",
            pooling="avg",
        )
    else:
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            input_tensor=input_layer,
            weights="imagenet",
            pooling="avg",
        )

    base_model.trainable = False

    x = tf.keras.layers.Dropout(0.2)(base_model.output)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=input_layer, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def train(
    data_dir: str,
    output_dir: str,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 16,
    epochs: int = 8,
    val_split: float = 0.2,
    base_model_name: str = "efficientnetb0",
    seed: int = 42,
) -> Dict[str, Any]:
    set_seed(seed)
    output_root = ensure_dir(output_dir)

    train_ds, val_ds, class_names = build_datasets(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        val_split=val_split,
        seed=seed,
    )

    model = build_model(
        num_classes=len(class_names),
        image_size=image_size,
        base_model_name=base_model_name,
    )

    checkpoint_path = output_root / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_path), save_best_only=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    predictions = model.predict(val_ds)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.concatenate([labels.numpy() for _, labels in val_ds], axis=0)

    _ = confusion_matrix(y_true, y_pred)
    metrics = compute_classification_metrics(y_true, y_pred)
    metrics_payload = {
        "timestamp": utc_timestamp(),
        "num_classes": len(class_names),
        "class_names": class_names,
        "epochs": epochs,
        **metrics,
        "history": {k: [float(vv) for vv in v] for k, v in history.history.items()},
    }

    model_path = output_root / "model.keras"
    metrics_path = output_root / "metrics.json"
    confusion_path = output_root / "confusion_matrix.png"

    model.save(model_path)
    save_json(metrics_payload, metrics_path)
    plot_and_save_confusion_matrix(y_true, y_pred, class_names, confusion_path)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "confusion_matrix_path": str(confusion_path),
    }


if __name__ == "__main__":
    artifacts = train(
        data_dir="data/raw",
        output_dir="outputs/results",
    )
    print("Training complete:", artifacts)

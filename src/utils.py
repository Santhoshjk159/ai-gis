import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        cmap="Blues",
        ax=axis,
        xticks_rotation=45,
    )
    display.ax_.set_title("Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def ecological_authenticity_index(
    model_confidence: float,
    latitude: float,
    longitude: float,
    elevation_m: float,
    environmental_suitability: float,
) -> float:
    location_valid = 1.0 if -90 <= latitude <= 90 and -180 <= longitude <= 180 else 0.0
    elevation_score = 1.0 - min(abs(elevation_m - 900.0) / 2000.0, 1.0)
    suitability_score = float(np.clip(environmental_suitability, 0.0, 1.0))
    confidence_score = float(np.clip(model_confidence, 0.0, 1.0))

    final_score = (
        0.45 * confidence_score
        + 0.20 * location_valid
        + 0.15 * elevation_score
        + 0.20 * suitability_score
    )
    return float(np.clip(final_score, 0.0, 1.0))


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf

from src.preprocessing import load_and_preprocess_image


def load_trained_model(model_path: str | Path) -> tf.keras.Model:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path)


def predict_image(
    model: tf.keras.Model,
    image_path: str | Path,
    image_size: tuple[int, int] = (224, 224),
    class_names: Optional[list[str]] = None,
) -> Dict[str, Any]:
    image = load_and_preprocess_image(image_path, image_size)
    batched = np.expand_dims(image, axis=0)
    probabilities = model.predict(batched, verbose=0)[0]

    predicted_idx = int(np.argmax(probabilities))
    predicted_label = class_names[predicted_idx] if class_names and predicted_idx < len(class_names) else str(predicted_idx)

    return {
        "predicted_index": predicted_idx,
        "predicted_label": predicted_label,
        "confidence": float(np.max(probabilities)),
        "probabilities": [float(p) for p in probabilities.tolist()],
    }

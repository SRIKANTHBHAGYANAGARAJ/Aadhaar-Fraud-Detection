# advanced_model.py

from transformers import pipeline
from PIL import Image
import torch
import numpy as np

device = 0 if torch.cuda.is_available() else -1

forgery_pipeline = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224",
)

def advanced_model_analysis(image_path):

    try:
        image = Image.open(image_path).convert("RGB")

        predictions = forgery_pipeline(image)

        # Sort predictions
        predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)

        top = predictions[0]
        label = top["label"]
        confidence = float(top["score"])

        # Softmax confidence calibration
        calibrated_conf = np.clip(confidence * 1.1, 0, 1)

        risk_score = calibrated_conf * 100

        return {
            "predicted_label": label,
            "raw_confidence": round(confidence, 4),
            "calibrated_confidence": round(float(calibrated_conf), 4),
            "risk_score": round(float(risk_score), 2)
        }

    except Exception as e:
        return {"risk_score": 45, "error": str(e)}

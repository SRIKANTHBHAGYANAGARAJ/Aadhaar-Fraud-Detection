# advanced_ela.py

import os
import numpy as np
from PIL import Image, ImageChops

def advanced_ela_analysis(image_path, quality=90, block_size=32):

    try:
        original = Image.open(image_path).convert("RGB")

        temp_path = "temp_ela.jpg"
        original.save(temp_path, "JPEG", quality=quality)
        compressed = Image.open(temp_path)

        ela_image = ImageChops.difference(original, compressed)
        ela_array = np.array(ela_image).astype("float32") / 255.0

        os.remove(temp_path)

        # Global statistics
        mean_intensity = np.mean(ela_array)
        variance = np.var(ela_array)

        # Block inconsistency detection
        h, w, _ = ela_array.shape
        block_variances = []

        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = ela_array[i:i+block_size, j:j+block_size]
                if block.size > 0:
                    block_variances.append(np.var(block))

        block_variance_score = np.std(block_variances)

        # Risk computation (normalized scaling)
        risk_score = (
            (variance * 200) +
            (mean_intensity * 100) +
            (block_variance_score * 300)
        )

        return {
            "mean_intensity": round(float(mean_intensity), 6),
            "global_variance": round(float(variance), 6),
            "block_inconsistency": round(float(block_variance_score), 6),
            "risk_score": round(float(risk_score), 2)
        }

    except Exception as e:
        return {"error": str(e), "risk_score": 35}

import cv2
import numpy as np
import os


def noise_analysis(image_path, save_output=True):
    """
    Performs noise inconsistency detection on an image.
    Returns:
        noise_residual (image)
        heatmap (image)
        suspicious_mask (binary image)
    """

    # 1️⃣ Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found!")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2️⃣ Apply Gaussian Blur (remove noise approximation)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3️⃣ Extract noise residual
    noise_residual = cv2.subtract(gray, blurred)

    # 4️⃣ Normalize residual for visualization
    residual_norm = cv2.normalize(
        noise_residual, None, 0, 255, cv2.NORM_MINMAX
    )

    # 5️⃣ Create heatmap
    heatmap = cv2.applyColorMap(residual_norm, cv2.COLORMAP_JET)

    # 6️⃣ Detect abnormal noise regions
    abs_noise = cv2.absdiff(gray, blurred)
    _, suspicious_mask = cv2.threshold(abs_noise, 25, 255, cv2.THRESH_BINARY)

    # 7️⃣ Optional: Morphological cleaning
    kernel = np.ones((3, 3), np.uint8)
    suspicious_mask = cv2.morphologyEx(
        suspicious_mask, cv2.MORPH_CLOSE, kernel
    )

    # 8️⃣ Save results
    if save_output:
        os.makedirs("assets/noise", exist_ok=True)
        cv2.imwrite("assets/noise/noise_residual.jpg", residual_norm)
        cv2.imwrite("assets/noise/noise_heatmap.jpg", heatmap)
        cv2.imwrite("assets/noise/suspicious_mask.jpg", suspicious_mask)

    return residual_norm, heatmap, suspicious_mask


if __name__ == "__main__":
    image_path = "test.jpg"
    noise_analysis(image_path)
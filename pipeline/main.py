import os
from PIL import Image, ImageChops, ImageEnhance, ExifTags
from datetime import datetime
import numpy as np

# ==============================
# 1️⃣ EXIF + METADATA ANALYSIS
# ==============================

def analyze_metadata(image_path):
    result = {
        "has_exif": False,
        "camera_model": None,
        "software": None,
        "datetime_original": None,
        "metadata_issues": [],
        "decision": "Unknown"
    }

    try:
        image = Image.open(image_path)
        exif_data = image._getexif()

        if not exif_data:
            result["metadata_issues"].append("No EXIF metadata found")
            result["decision"] = "Likely Tampered"
            return result

        result["has_exif"] = True

        exif = {}
        for tag_id, value in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            exif[tag] = value

        result["camera_model"] = exif.get("Model")
        result["software"] = exif.get("Software")
        result["datetime_original"] = exif.get("DateTimeOriginal")

        # Missing important fields
        if not result["camera_model"]:
            result["metadata_issues"].append("Camera model missing")

        if not result["datetime_original"]:
            result["metadata_issues"].append("Original capture date missing")

        # Suspicious software check
        suspicious_softwares = ["photoshop", "gimp", "canva", "snapseed", "pixlr", "editor"]

        if result["software"]:
            for sw in suspicious_softwares:
                if sw in result["software"].lower():
                    result["metadata_issues"].append(
                        f"Edited using suspicious software: {result['software']}"
                    )
                    break

        if len(result["metadata_issues"]) > 0:
            result["decision"] = "Likely Tampered"
        else:
            result["decision"] = "Likely Real"

        return result

    except Exception as e:
        return {"error": f"Metadata analysis failed: {str(e)}"}


# ==============================
# 2️⃣ ELA ANALYSIS
# ==============================

def perform_ela(image_path, quality=90):
    try:
        original = Image.open(image_path).convert("RGB")
        temp_path = "temp_resaved.jpg"
        original.save(temp_path, "JPEG", quality=quality)

        resaved = Image.open(temp_path)
        ela_image = ImageChops.difference(original, resaved)

        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])

        if max_diff == 0:
            max_diff = 1

        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        # Convert to numpy to measure intensity
        ela_array = np.array(ela_image)
        mean_intensity = np.mean(ela_array)

        os.remove(temp_path)

        # Threshold logic (tune if needed)
        if mean_intensity > 25:
            decision = "Likely Tampered"
        else:
            decision = "Likely Real"

        return {
            "ela_score": round(float(mean_intensity), 2),
            "decision": decision
        }

    except Exception as e:
        return {"error": f"ELA analysis failed: {str(e)}"}


# ==============================
# 3️⃣ FINAL DECISION ENGINE
# ==============================

def final_image_verification(image_path):
    metadata_result = analyze_metadata(image_path)
    ela_result = perform_ela(image_path)

    tamper_flags = 0

    if metadata_result.get("decision") == "Likely Real":
        tamper_flags += 1

    if ela_result.get("decision") == "Likely Tampered":
        tamper_flags += 1

    # Final logic
    if tamper_flags >= 1:
        final_decision = "⚠️ IMAGE IS LIKELY TAMPERED"
    else:
        final_decision = "✅ IMAGE APPEARS AUTHENTIC"

    return {
        "metadata_analysis": metadata_result,
        "ela_analysis": ela_result,
        "final_decision": final_decision
    }


# ==============================
# 4️⃣ RUN TEST
# ==============================

if __name__ == "__main__":
    image_path = "sample\doggy.jpg"  # Change to your image path

    result = final_image_verification(image_path)

    print("\n========== IMAGE VERIFICATION RESULT ==========")
    print(result)

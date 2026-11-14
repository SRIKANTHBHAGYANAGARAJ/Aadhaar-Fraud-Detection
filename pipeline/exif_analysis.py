# advanced_exif.py

from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime

def advanced_exif_analysis(image_path):

    risk = 0
    reasons = []

    try:
        img = Image.open(image_path)
        exif_raw = img._getexif()

        if not exif_raw:
            return {
                "risk_score": 50,
                "reasons": ["Metadata stripped or missing"]
            }

        exif = {}
        for tag_id, value in exif_raw.items():
            tag = TAGS.get(tag_id, tag_id)
            exif[tag] = value

        # 1️⃣ Editing software detection
        software = exif.get("Software", "")
        suspicious_tools = [
            "photoshop", "gimp", "canva",
            "snapseed", "pixlr", "lightroom",
            "editor"
        ]

        if any(tool in software.lower() for tool in suspicious_tools):
            risk += 60
            reasons.append(f"Edited using {software}")

        # 2️⃣ Timestamp anomaly detection
        create_time = exif.get("DateTimeOriginal")
        modify_time = exif.get("DateTime")

        if create_time and modify_time:
            if create_time != modify_time:
                risk += 20
                reasons.append("Creation and modification timestamps mismatch")

            try:
                create_dt = datetime.strptime(create_time, "%Y:%m:%d %H:%M:%S")
                if create_dt.year < 2000 or create_dt.year > datetime.now().year:
                    risk += 15
                    reasons.append("Abnormal creation date")
            except:
                pass

        # 3️⃣ Camera consistency check
        make = exif.get("Make")
        model = exif.get("Model")

        if not make or not model:
            risk += 15
            reasons.append("Camera information incomplete")

        # 4️⃣ GPS anomaly
        if "GPSInfo" not in exif:
            risk += 10
            reasons.append("GPS metadata missing")

        return {
            "risk_score": risk,
            "camera_make": make,
            "camera_model": model,
            "software": software,
            "reasons": reasons
        }

    except Exception as e:
        return {"risk_score": 40, "reasons": [str(e)]}

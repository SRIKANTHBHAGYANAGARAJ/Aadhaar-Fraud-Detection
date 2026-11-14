# app.py
"""
Final Flask app for Aadhar card analysis.
- Uses OpenCV's QRCodeDetector (NO pyzbar / NO pyaadhaar / NO zbar).
- Uses YOLO (ultralytics) for general/object/text detection when available.
- Performs OCR with pytesseract.
- Parses Aadhaar-like QR payloads via XML/JSON/regex fallback.
- Keeps same model paths you confirmed (yolov8n.pt + HF hub repo paths).
"""

import os
import re
import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from datetime import datetime

from flask import Flask, request, render_template, jsonify, flash, redirect, url_for

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from PIL.ExifTags import TAGS

import pytesseract
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

import supervision as sv
from supervision import Detections

import xml.etree.ElementTree as ET
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# Configuration
# -------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'secret-key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Tesseract (Windows default)
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# -------------------------
# Helper functions
# -------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_str(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return ""


SUSPICIOUS_SOFTWARE_TERMS = ("photoshop", "gimp", "canva", "editor", "snapseed", "pixlr", "lightroom")
OBJECT_TAMPER_MIN_CONFIDENCE = float(os.getenv("OBJECT_TAMPER_MIN_CONFIDENCE", "0.55"))
OBJECT_TAMPER_STRONG_CONFIDENCE = float(os.getenv("OBJECT_TAMPER_STRONG_CONFIDENCE", "0.75"))
NAME_MISMATCH_SIMILARITY_THRESHOLD = float(os.getenv("NAME_MISMATCH_SIMILARITY_THRESHOLD", "0.58"))
ADDRESS_MISMATCH_SIMILARITY_THRESHOLD = float(os.getenv("ADDRESS_MISMATCH_SIMILARITY_THRESHOLD", "0.55"))
FAST_ANALYSIS_MODE = True
FAST_OCR_MAX_ATTEMPTS = 1
FAST_FALLBACK_OCR_MAX_ATTEMPTS = 1
QR_MAX_DECODE_DIM = 800


def _rotate_image(image, angle):
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _ensure_bgr(image):
    if image is None:
        return None
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _resize_to_max_dim(image, max_dim):
    if image is None:
        return None
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = float(max_dim) / float(longest)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _normalize_space(text):
    return re.sub(r"\s+", " ", safe_str(text)).strip()


def _normalize_compare_text(value):
    cleaned = re.sub(r"[^a-z0-9\s]", " ", safe_str(value).lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _text_similarity(value_a, value_b):
    left = _normalize_compare_text(value_a)
    right = _normalize_compare_text(value_b)
    if not left or not right:
        return 0.0
    return float(SequenceMatcher(None, left, right).ratio())


def _token_overlap_ratio(value_a, value_b):
    left_tokens = {token for token in _normalize_compare_text(value_a).split() if len(token) >= 2}
    right_tokens = {token for token in _normalize_compare_text(value_b).split() if len(token) >= 2}
    if not left_tokens or not right_tokens:
        return 0.0
    return float(len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens))))


def _dob_consistent(ocr_dob, qr_dob):
    left = _normalize_space(ocr_dob).replace("-", "/")
    right = _normalize_space(qr_dob).replace("-", "/")
    if not left or not right:
        return True
    if left == right:
        return True

    left_year = re.search(r"(19|20)\d{2}", left)
    right_year = re.search(r"(19|20)\d{2}", right)
    if left_year and right_year and left_year.group(0) == right_year.group(0):
        left_full = bool(re.search(r"\d{2}/\d{2}/\d{4}", left))
        right_full = bool(re.search(r"\d{2}/\d{2}/\d{4}", right))
        if not (left_full and right_full):
            return True

    return False


def _format_aadhaar_number(digits):
    digits_only = re.sub(r"\D", "", safe_str(digits))
    if len(digits_only) > 12:
        digits_only = digits_only[:12]
    if len(digits_only) == 12:
        return f"{digits_only[:4]} {digits_only[4:8]} {digits_only[8:]}"
    return digits_only


def _sanitize_ocr_value(field_name, value):
    value = _normalize_space(value)
    if not value:
        return ""

    if field_name == "Aadhaar Number":
        return _format_aadhaar_number(value)

    if field_name == "Date of Birth":
        match = re.search(r"(?:\d{2}[/-]\d{2}[/-]\d{4})|(?:\d{4})", value)
        return match.group(0) if match else value

    if field_name == "Gender":
        lower = value.lower()
        if re.search(r"\bmale\b|\bm\b", lower):
            return "Male"
        if re.search(r"\bfemale\b|\bf\b", lower):
            return "Female"
        return value.title()

    if field_name == "Name":
        cleaned = re.sub(r"[^A-Za-z.\s]", " ", value)
        return _normalize_space(cleaned).title()

    if field_name == "Address":
        return value

    return value


def _ocr_variants_from_array(image_array, include_inverted=False):
    if image_array is None or getattr(image_array, "size", 0) == 0:
        return []

    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array.copy()

    h, w = gray.shape[:2]
    min_dim = min(h, w)
    scale = 2.0 if min_dim < 900 else 1.4
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=55, sigmaSpace=55)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    _, otsu = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        equalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )

    if FAST_ANALYSIS_MODE:
        return [gray]

    variants = [equalized, otsu, adaptive]
    if include_inverted:
        variants.append(cv2.bitwise_not(otsu))
    return variants


def _extract_text_and_confidence(roi, config, lang="eng+hin", allow_string_fallback=False):
    try:
        data = pytesseract.image_to_data(
            roi,
            lang=lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        tokens = []
        confs = []
        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            token = _normalize_space(text)
            if token:
                tokens.append(token)
            try:
                conf_value = float(conf)
                if conf_value >= 0:
                    confs.append(conf_value)
            except Exception:
                continue

        text_value = _normalize_space(" ".join(tokens))
        if not text_value and allow_string_fallback:
            text_value = _normalize_space(pytesseract.image_to_string(roi, lang=lang, config=config))
        mean_conf = float(np.mean(confs)) if confs else (30.0 if text_value else 0.0)
        return text_value, mean_conf
    except Exception:
        return "", 0.0


def _score_ocr_candidate(field_name, text, mean_conf):
    text = _normalize_space(text)
    if not text:
        return 0.0

    score = float(np.clip(mean_conf, 0.0, 100.0))
    score += min(len(text), 60) * 0.35

    if field_name == "Aadhaar Number":
        digits = re.sub(r"\D", "", text)
        if len(digits) == 12:
            score += 55
        elif len(digits) >= 10:
            score += 20
        else:
            score -= 20
        if len(set(digits)) <= 2:
            score -= 10

    elif field_name == "Date of Birth":
        if re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", text):
            score += 45
        elif re.search(r"\b\d{4}\b", text):
            score += 20

    elif field_name == "Gender":
        if re.search(r"\bmale\b|\bfemale\b|\bm\b|\bf\b", text, re.IGNORECASE):
            score += 45
        else:
            score -= 10

    elif field_name == "Name":
        words = re.findall(r"[A-Za-z]{2,}", text)
        if len(words) >= 2:
            score += 25
        if any(token.lower() in text.lower() for token in ("uidai", "government", "india")):
            score -= 15

    elif field_name == "Address":
        if len(text) >= 20:
            score += 20

    return float(score)


def _run_adaptive_ocr(roi, field_name):
    if roi is None or getattr(roi, "size", 0) == 0:
        return "", 0.0

    if field_name == "Aadhaar Number":
        configs = [
            ("--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789 ", "eng"),
            ("--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789 ", "eng"),
            ("--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789 ", "eng"),
        ]
        variants = _ocr_variants_from_array(roi, include_inverted=True)
        target_score = 110.0
        max_attempts = min(len(variants) * len(configs), FAST_OCR_MAX_ATTEMPTS if FAST_ANALYSIS_MODE else 10_000)
    elif field_name == "Date of Birth":
        configs = [
            ("--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789/- ", "eng"),
            ("--oem 3 --psm 6", "eng+hin"),
            ("--oem 3 --psm 11", "eng+hin"),
        ]
        variants = _ocr_variants_from_array(roi, include_inverted=True)
        target_score = 95.0
        max_attempts = min(len(variants) * len(configs), FAST_OCR_MAX_ATTEMPTS if FAST_ANALYSIS_MODE else 10_000)
    elif field_name == "Gender":
        configs = [
            ("--oem 3 --psm 7", "eng"),
            ("--oem 3 --psm 8", "eng"),
            ("--oem 3 --psm 6", "eng"),
        ]
        variants = _ocr_variants_from_array(roi, include_inverted=False)
        target_score = 85.0
        max_attempts = min(len(variants) * len(configs), 2 if FAST_ANALYSIS_MODE else 10_000)
    else:
        configs = [
            ("--oem 3 --psm 6", "eng+hin"),
            ("--oem 3 --psm 11", "eng+hin"),
            ("--oem 3 --psm 7", "eng+hin"),
        ]
        variants = _ocr_variants_from_array(roi, include_inverted=False)
        target_score = 90.0 if field_name == "Name" else 95.0
        max_attempts = min(len(variants) * len(configs), FAST_OCR_MAX_ATTEMPTS if FAST_ANALYSIS_MODE else 10_000)

    if FAST_ANALYSIS_MODE and variants:
        if field_name in ("Aadhaar Number", "Date of Birth"):
            preferred_order = [1, 0, 3, 2]
        elif field_name == "Gender":
            preferred_order = [1, 0]
        else:
            preferred_order = [0, 1, 2]
        ordered_variants = [variants[idx] for idx in preferred_order if idx < len(variants)]
        if ordered_variants:
            variants = ordered_variants

    best_text = ""
    best_score = 0.0
    attempts = 0
    total_candidates = len(variants) * len(configs)
    for variant in variants:
        for config, lang in configs:
            attempts += 1
            allow_string_fallback = (not FAST_ANALYSIS_MODE) and (attempts == total_candidates)
            candidate, mean_conf = _extract_text_and_confidence(
                variant,
                config=config,
                lang=lang,
                allow_string_fallback=allow_string_fallback,
            )
            score = _score_ocr_candidate(field_name, candidate, mean_conf)
            if score > best_score:
                best_text = candidate
                best_score = score
                if best_score >= target_score:
                    return _sanitize_ocr_value(field_name, best_text), best_score
            if attempts >= max_attempts:
                return _sanitize_ocr_value(field_name, best_text), best_score

    return _sanitize_ocr_value(field_name, best_text), best_score


def _extract_labeled_value(lines, label_terms, max_lookahead=2):
    label_terms = tuple(term.lower() for term in label_terms)
    for idx, raw_line in enumerate(lines):
        line = _normalize_space(raw_line)
        line_lower = line.lower()
        if not line_lower:
            continue
        if not any(term in line_lower for term in label_terms):
            continue

        parts = re.split(r"[:\-]", line, maxsplit=1)
        if len(parts) == 2:
            value = _normalize_space(parts[1])
            if value:
                return value, idx

        for step in range(1, max_lookahead + 1):
            if idx + step >= len(lines):
                break
            follow = _normalize_space(lines[idx + step])
            follow_lower = follow.lower()
            if not follow:
                continue
            if any(term in follow_lower for term in label_terms):
                continue
            return follow, idx + step

    return "", -1


def _score_document_field(field_name, value, base_confidence=0.0):
    value = _sanitize_ocr_value(field_name, value)
    if not value:
        return 0.0

    score = float(np.clip(base_confidence, 0.0, 100.0))
    score += min(len(value), 80) * 0.25

    if field_name == "Aadhaar Number":
        digits = re.sub(r"\D", "", value)
        if len(digits) == 12:
            score += 70
        elif len(digits) >= 10:
            score += 25
        else:
            score -= 15

    elif field_name == "Date of Birth":
        if re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", value):
            score += 45
        elif re.search(r"\b(?:19|20)\d{2}\b", value):
            score += 25

    elif field_name == "Gender":
        if re.search(r"\bmale\b|\bfemale\b|\bm\b|\bf\b", value, re.IGNORECASE):
            score += 35

    elif field_name == "Name":
        words = re.findall(r"[A-Za-z]{2,}", value)
        if len(words) >= 2:
            score += 30

    elif field_name == "Address":
        if len(value) >= 18:
            score += 22

    return float(score)


def _parse_fallback_fields_from_text(raw_text):
    text = safe_str(raw_text)
    if not text:
        return {}

    normalized_text = text.replace("\r", "\n")
    normalized_text = re.sub(r"[ \t]+", " ", normalized_text)
    fields = {}
    lines = [_normalize_space(line) for line in normalized_text.split("\n")]
    lines = [line for line in lines if line]

    # Prefer explicit labels when available
    name_labeled, _ = _extract_labeled_value(lines, ("name",))
    if name_labeled:
        fields["Name"] = _sanitize_ocr_value("Name", name_labeled)

    dob_labeled, _ = _extract_labeled_value(lines, ("dob", "date of birth", "year of birth", "yob"))
    if dob_labeled:
        fields["Date of Birth"] = _sanitize_ocr_value("Date of Birth", dob_labeled)

    gender_labeled, _ = _extract_labeled_value(lines, ("gender", "sex"))
    if gender_labeled:
        fields["Gender"] = _sanitize_ocr_value("Gender", gender_labeled)

    aadhaar_labeled, _ = _extract_labeled_value(lines, ("aadhaar", "aadhar", "uid"))
    if aadhaar_labeled:
        fields["Aadhaar Number"] = _sanitize_ocr_value("Aadhaar Number", aadhaar_labeled)

    address_labeled, address_idx = _extract_labeled_value(lines, ("address", "addr"), max_lookahead=1)
    if address_idx >= 0:
        address_chunks = []
        if address_labeled:
            address_chunks.append(address_labeled)
        for j in range(address_idx + 1, min(address_idx + 5, len(lines))):
            follow = _normalize_space(lines[j])
            follow_lower = follow.lower()
            if not follow:
                continue
            if any(token in follow_lower for token in ("uidai", "government", "aadhaar", "aadhar")):
                break
            if re.search(r"^\d{4}\s?\d{4}\s?\d{4}$", follow):
                break
            address_chunks.append(follow)
        merged_addr = _normalize_space(" ".join(address_chunks))
        if len(merged_addr) >= 10:
            fields["Address"] = merged_addr

    aadhaar_match = re.search(r"(?<!\d)(\d[\d\s-]{10,16}\d)(?!\d)", normalized_text)
    if aadhaar_match and not fields.get("Aadhaar Number"):
        fields["Aadhaar Number"] = _format_aadhaar_number(aadhaar_match.group(1))

    dob_match = re.search(r"(?:\d{2}[/-]\d{2}[/-]\d{4})|(?:\b(?:19|20)\d{2}\b)", normalized_text)
    if dob_match and not fields.get("Date of Birth"):
        fields["Date of Birth"] = dob_match.group(0)

    gender_match = re.search(r"\b(male|female|m|f)\b", normalized_text, re.IGNORECASE)
    if gender_match and not fields.get("Gender"):
        gender_token = gender_match.group(1).lower()
        fields["Gender"] = "Male" if gender_token in ("male", "m") else "Female"

    ignore_terms = (
        "government",
        "india",
        "uidai",
        "aadhaar",
        "aadhar",
        "dob",
        "year of birth",
        "male",
        "female",
        "address",
        "enrolment",
    )

    if not fields.get("Name"):
        for line in lines:
            line_lower = line.lower()
            if any(term in line_lower for term in ignore_terms):
                continue
            if re.search(r"\d", line):
                continue
            words = re.findall(r"[A-Za-z]{2,}", line)
            if len(words) >= 2:
                fields["Name"] = _sanitize_ocr_value("Name", line)
                break

    return fields


def _extract_fallback_fields_from_image(image_array, required_fields=None, fast_mode=FAST_ANALYSIS_MODE):
    target_fields = {
        safe_str(item).strip() for item in (required_fields or ("Name", "Date of Birth", "Gender", "Aadhaar Number", "Address"))
        if safe_str(item).strip()
    }

    best_fields = {}
    best_scores = {}
    variants = _ocr_variants_from_array(image_array, include_inverted=False)
    if image_array is not None and getattr(image_array, "size", 0) > 0:
        variants = [image_array] + variants

    if not variants:
        return {}

    if fast_mode:
        configs = ("--oem 3 --psm 6", "--oem 3 --psm 11")
        variants = variants[:2]
        max_attempts = min(len(variants) * len(configs), FAST_FALLBACK_OCR_MAX_ATTEMPTS)
    else:
        configs = ("--oem 3 --psm 6", "--oem 3 --psm 11", "--oem 3 --psm 4")
        variants = variants[:4]
        max_attempts = len(variants) * len(configs)

    attempts = 0
    total_candidates = max(1, len(variants) * len(configs))
    for variant in variants:
        for config in configs:
            attempts += 1
            allow_string_fallback = (not fast_mode) and (attempts == total_candidates)
            text_value, mean_conf = _extract_text_and_confidence(
                variant,
                config=config,
                lang="eng+hin",
                allow_string_fallback=allow_string_fallback,
            )
            if not text_value:
                if attempts >= max_attempts:
                    return best_fields
                continue

            parsed = _parse_fallback_fields_from_text(text_value)
            for field_name, value in parsed.items():
                score = _score_document_field(field_name, value, mean_conf)
                if score >= float(best_scores.get(field_name, -1.0)):
                    best_scores[field_name] = float(score)
                    best_fields[field_name] = _sanitize_ocr_value(field_name, value)

            if target_fields and target_fields.issubset(set(best_fields.keys())):
                return best_fields
            if attempts >= max_attempts:
                return best_fields

    return best_fields


def _merge_ocr_field(existing_value, candidate_value, field_name):
    existing = _sanitize_ocr_value(field_name, existing_value)
    candidate = _sanitize_ocr_value(field_name, candidate_value)
    if not candidate:
        return existing
    if not existing:
        return candidate

    if field_name in ("Name", "Gender", "Date of Birth"):
        existing_score = _score_document_field(field_name, existing, 35.0)
        candidate_score = _score_document_field(field_name, candidate, 55.0)
        return candidate if candidate_score >= existing_score else existing

    if field_name == "Address":
        return candidate if len(candidate) > len(existing) else existing

    if field_name == "Aadhaar Number":
        existing_digits = re.sub(r"\D", "", existing)
        candidate_digits = re.sub(r"\D", "", candidate)
        if len(candidate_digits) == 12 and len(existing_digits) != 12:
            return candidate
        return existing

    return existing

# -------------------------
# Try loading YOLO models (best-effort)
# -------------------------
general_model = None
object_detection_model = None
aadhaar_model = None

try:
    general_model = YOLO("yolov8n.pt")
    print("Loaded general yolov8n.pt")
except Exception as e:
    general_model = None
    print(f"Warning: general YOLO load failed: {e}")

# custom object detection from HF (user-confirmed paths)
try:
    REPO_ID = "ConiferousYogi/Weights_for_Aadhar_Card_Fraud_Detection"
    FILENAME = "models/best.pt"
    COMMIT_SHA = "8e491271abe6e223322b307f6a5f33892a0914b6"
    local_model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, revision=COMMIT_SHA)
    object_detection_model = YOLO(local_model_path)
    print("Loaded custom object detection model from HF.")
except Exception as e:
    object_detection_model = None
    print(f"Warning: custom object model load failed: {e}")

# Aadhaar-specific OCR/text model from HF
try:
    repo_config = dict(repo_id="arnabdhar/YOLOv8-nano-aadhar-card", filename="model.pt")
    aadhaar_model = YOLO(hf_hub_download(**repo_config))
    print("Loaded Aadhaar YOLO text model.")
except Exception as e:
    aadhaar_model = None
    print(f"Warning: aadhaar text model load failed: {e}")

# -------------------------
# Verhoeff checksum tables for Aadhaar validation
# -------------------------
_d = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
)
_p = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8)
)
_inv = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)

def validate_checksum(num_str_with_checksum):
    try:
        c = 0
        num_digits = [int(d) for d in num_str_with_checksum]
        for i, digit in enumerate(reversed(num_digits)):
            c = _d[c][_p[(i % 8)][digit]]
        return c == 0
    except Exception:
        return False

# -------------------------
# EXIF extraction
# -------------------------
def get_exif_data(image_path):
    try:
        img = Image.open(image_path)
        exif = {}
        if hasattr(img, "_getexif"):
            info = img._getexif()
            if info:
                for tag, value in info.items():
                    decoded = TAGS.get(tag, tag)
                    exif[decoded] = value
        return exif
    except Exception as e:
        return {"error": str(e)}


def _save_pil_image_to_static(image, prefix):
    try:
        output_dir = os.path.join("static", "analysis")
        os.makedirs(output_dir, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        filename = f"{prefix}_{stamp}.png"
        rel_path = os.path.join("analysis", filename).replace("\\", "/")
        image.save(os.path.join("static", rel_path))
        return rel_path
    except Exception:
        return None


def _save_cv2_image_to_static(image_array, prefix):
    try:
        output_dir = os.path.join("static", "analysis")
        os.makedirs(output_dir, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        filename = f"{prefix}_{stamp}.png"
        rel_path = os.path.join("analysis", filename).replace("\\", "/")
        if cv2.imwrite(os.path.join("static", rel_path), image_array):
            return rel_path
        return None
    except Exception:
        return None


def _static_rel_path_to_url(rel_path):
    clean = safe_str(rel_path).replace("\\", "/").lstrip("/")
    if not clean:
        return None
    try:
        return url_for("static", filename=clean)
    except Exception:
        return f"/static/{clean}"


def _classify_forensic_risk(score):
    if score >= 70:
        return "Likely Tampered"
    if score >= 40:
        return "Suspicious"
    return "Likely Real"


def _compute_cross_side_analysis(front_path, back_path):
    try:
        front = cv2.imread(front_path, cv2.IMREAD_GRAYSCALE)
        back = cv2.imread(back_path, cv2.IMREAD_GRAYSCALE)
        if front is None or back is None:
            return {
                "available": False,
                "risk_score": 0.0,
                "decision": "Unavailable",
                "reason": "Could not read one or both side images for cross-side consistency check.",
            }

        target_size = (512, 320)
        front = cv2.resize(front, target_size, interpolation=cv2.INTER_AREA)
        back = cv2.resize(back, target_size, interpolation=cv2.INTER_AREA)

        front_f = front.astype(np.float32) / 255.0
        back_f = back.astype(np.float32) / 255.0

        front_std = float(np.std(front_f))
        back_std = float(np.std(back_f))
        if front_std < 1e-6 or back_std < 1e-6:
            corr = 0.0
        else:
            corr = float(np.corrcoef(front_f.flatten(), back_f.flatten())[0, 1])

        hist_front = cv2.calcHist([front], [0], None, [64], [0, 256])
        hist_back = cv2.calcHist([back], [0], None, [64], [0, 256])
        hist_corr = float(cv2.compareHist(hist_front, hist_back, cv2.HISTCMP_CORREL))

        mad = float(np.mean(np.abs(front_f - back_f)))

        if corr >= 0.97 and hist_corr >= 0.90 and mad <= 0.03:
            risk_score = 95.0
            reason = "Front and back images are near-identical."
        elif corr >= 0.90 and hist_corr >= 0.75 and mad <= 0.06:
            risk_score = 80.0
            reason = "Front and back images are highly similar."
        elif corr >= 0.78 and hist_corr >= 0.45 and mad <= 0.10:
            risk_score = 55.0
            reason = "Front and back images are unusually similar."
        else:
            risk_score = 8.0
            reason = "Front and back images show expected visual differences."

        return {
            "available": True,
            "risk_score": round(risk_score, 2),
            "decision": _classify_forensic_risk(risk_score),
            "correlation": round(corr, 4),
            "histogram_correlation": round(hist_corr, 4),
            "mean_abs_difference": round(mad, 4),
            "reason": reason,
        }
    except Exception as e:
        return {
            "available": False,
            "risk_score": 0.0,
            "decision": "Unavailable",
            "reason": f"Cross-side analysis failed: {safe_str(e)}",
        }


def _compute_ela_metrics(original_image, quality):
    temp_jpg = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            temp_jpg = tmp.name

        original_image.save(temp_jpg, "JPEG", quality=int(quality))
        compressed = Image.open(temp_jpg).convert("RGB")
        ela_image = ImageChops.difference(original_image, compressed)
        extrema = ela_image.getextrema()
        max_diff = max((ch[1] for ch in extrema), default=1)
        if max_diff <= 0:
            max_diff = 1

        ela_enhanced = ImageEnhance.Brightness(ela_image).enhance(255.0 / float(max_diff))
        ela_rgb = np.asarray(ela_enhanced, dtype=np.uint8)
        ela_gray = cv2.cvtColor(ela_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        mean_intensity = float(np.mean(ela_gray))
        std_intensity = float(np.std(ela_gray))
        p95 = float(np.percentile(ela_gray, 95))
        p99 = float(np.percentile(ela_gray, 99))
        hot_threshold = float(max(26.0, p95))
        hot_ratio = float(np.mean(ela_gray >= hot_threshold))
        laplacian_var = float(cv2.Laplacian(ela_gray, cv2.CV_32F).var())
        laplacian_norm = float(np.clip(np.log1p(laplacian_var) / 7.5, 0.0, 1.0))

        risk_score = float(
            np.clip(
                ((mean_intensity / 255.0) * 20.0)
                + ((std_intensity / 255.0) * 24.0)
                + ((p95 / 255.0) * 32.0)
                + ((p99 / 255.0) * 26.0)
                + (hot_ratio * 42.0)
                + (laplacian_norm * 14.0),
                0.0,
                100.0,
            )
        )

        return {
            "quality": int(quality),
            "risk_score": risk_score,
            "mean_intensity": mean_intensity,
            "std_intensity": std_intensity,
            "p95_intensity": p95,
            "p99_intensity": p99,
            "hot_ratio": hot_ratio,
            "laplacian_variance": laplacian_var,
            "ela_rgb": ela_rgb,
        }
    finally:
        if temp_jpg and os.path.exists(temp_jpg):
            try:
                os.unlink(temp_jpg)
            except Exception:
                pass


def perform_ela_analysis(image_path, side_label, quality=88):
    try:
        original = Image.open(image_path).convert("RGB")
        qualities = [int(quality)] if FAST_ANALYSIS_MODE else sorted(set([84, int(quality), 92]))
        quality_metrics = []
        for q in qualities:
            try:
                quality_metrics.append(_compute_ela_metrics(original, q))
            except Exception:
                continue

        if not quality_metrics:
            raise ValueError("ELA computation failed for all quality levels")

        risks = [float(item.get("risk_score", 0.0) or 0.0) for item in quality_metrics]
        mean_risk = float(np.mean(risks))
        max_risk = float(np.max(risks))
        min_risk = float(np.min(risks))
        spread = max_risk - min_risk

        max_p95 = float(np.max([item.get("p95_intensity", 0.0) for item in quality_metrics]))
        max_p99 = float(np.max([item.get("p99_intensity", 0.0) for item in quality_metrics]))
        max_hot = float(np.max([item.get("hot_ratio", 0.0) for item in quality_metrics]))
        max_lap = float(np.max([item.get("laplacian_variance", 0.0) for item in quality_metrics]))

        risk_score = float(np.clip((mean_risk * 0.6) + (max_risk * 0.4), 0.0, 100.0))

        # Guardrail: strong ELA spikes should not be classified as fully real.
        if max_hot >= 0.045 or max_p99 >= 88 or spread >= 22:
            risk_score = max(risk_score, 32.0)
        if max_hot >= 0.09 or max_p99 >= 125 or spread >= 34:
            risk_score = max(risk_score, 60.0)

        ext = os.path.splitext(safe_str(image_path))[1].lower()
        if ext in (".png", ".bmp", ".tif", ".tiff", ".webp"):
            risk_score = max(risk_score, 30.0)

        decision = _classify_forensic_risk(risk_score)
        representative = quality_metrics[min(len(quality_metrics) // 2, len(quality_metrics) - 1)]
        ela_rgb = representative.get("ela_rgb")
        ela_image_path = _save_cv2_image_to_static(cv2.cvtColor(ela_rgb, cv2.COLOR_RGB2BGR), f"ela_{side_label.lower()}") if isinstance(ela_rgb, np.ndarray) else None

        return {
            "side": side_label,
            "decision": decision,
            "risk_score": round(risk_score, 2),
            "mean_intensity": round(float(representative.get("mean_intensity", 0.0) or 0.0), 2),
            "std_intensity": round(float(representative.get("std_intensity", 0.0) or 0.0), 2),
            "p95_intensity": round(max_p95, 2),
            "p99_intensity": round(max_p99, 2),
            "hot_ratio": round(max_hot, 4),
            "laplacian_variance": round(max_lap, 4),
            "quality_spread": round(spread, 2),
            "quality_scores": [round(v, 2) for v in risks],
            "ela_image_path": ela_image_path,
        }
    except Exception as e:
        return {
            "side": side_label,
            "decision": "Unavailable",
            "risk_score": 0.0,
            "mean_intensity": 0.0,
            "std_intensity": 0.0,
            "p95_intensity": 0.0,
            "p99_intensity": 0.0,
            "hot_ratio": 0.0,
            "laplacian_variance": 0.0,
            "quality_spread": 0.0,
            "quality_scores": [],
            "ela_image_path": None,
            "error": str(e),
        }


def perform_noise_analysis(image_path, side_label):
    try:
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("Could not read image for noise analysis")
        gray = np.squeeze(gray)
        if gray.ndim != 2:
            gray = cv2.cvtColor(_ensure_bgr(gray), cv2.COLOR_BGR2GRAY)
        gray = np.asarray(gray, dtype=np.float32)

        gray_float = gray / 255.0
        smooth = cv2.GaussianBlur(gray_float, (0, 0), sigmaX=1.2, sigmaY=1.2)
        residual = gray_float - smooth
        residual_abs = np.abs(residual)

        local_mean = cv2.GaussianBlur(residual_abs, (0, 0), sigmaX=2.0, sigmaY=2.0)
        local_sq_mean = cv2.GaussianBlur(residual_abs * residual_abs, (0, 0), sigmaX=2.0, sigmaY=2.0)
        local_std = np.sqrt(np.clip(local_sq_mean - (local_mean * local_mean), 0.0, None))
        local_std = np.squeeze(local_std)

        grad_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
        texture = cv2.magnitude(np.squeeze(grad_x), np.squeeze(grad_y))
        texture = cv2.GaussianBlur(texture, (0, 0), sigmaX=1.5, sigmaY=1.5)
        texture = np.squeeze(texture)

        noise_ratio = local_std / (texture + 0.015)
        median_ratio = float(np.median(noise_ratio))
        mad_ratio = float(np.median(np.abs(noise_ratio - median_ratio)))
        robust_scale = (1.4826 * mad_ratio) + 1e-6
        robust_z = np.abs(noise_ratio - median_ratio) / robust_scale

        anomaly_threshold = float(max(2.8, np.percentile(robust_z, 90)))
        anomaly_mask = (robust_z >= anomaly_threshold).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        anomaly_ratio = float(np.mean(anomaly_mask > 0))

        connected_components = cv2.connectedComponentsWithStats(anomaly_mask, connectivity=8)
        _, _, stats, _ = connected_components
        if stats.shape[0] > 1:
            largest_component = float(np.max(stats[1:, cv2.CC_STAT_AREA]))
        else:
            largest_component = 0.0
        largest_component_ratio = largest_component / float(gray.shape[0] * gray.shape[1])
        z95 = float(np.percentile(robust_z, 95))
        dispersion = float(np.std(noise_ratio))

        risk_score = float(
            np.clip(
                (anomaly_ratio * 170.0)
                + (largest_component_ratio * 240.0)
                + (z95 * 8.0)
                + (dispersion * 35.0),
                0.0,
                100.0,
            )
        )
        decision = _classify_forensic_risk(risk_score)

        heatmap_base = cv2.normalize(np.clip(robust_z, 0, 8), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap_base, cv2.COLORMAP_INFERNO)
        mask_img = (anomaly_mask * 255).astype(np.uint8)
        heatmap_path = _save_cv2_image_to_static(heatmap, f"noise_heatmap_{side_label.lower()}")
        mask_path = _save_cv2_image_to_static(mask_img, f"noise_mask_{side_label.lower()}")

        return {
            "side": side_label,
            "decision": decision,
            "risk_score": round(risk_score, 2),
            "anomaly_ratio": round(anomaly_ratio, 4),
            "largest_component_ratio": round(largest_component_ratio, 4),
            "z95_score": round(z95, 2),
            "dispersion": round(dispersion, 4),
            "heatmap_path": heatmap_path,
            "mask_path": mask_path,
        }
    except Exception as e:
        return {
            "side": side_label,
            "decision": "Unavailable",
            "risk_score": 0.0,
            "anomaly_ratio": 0.0,
            "largest_component_ratio": 0.0,
            "z95_score": 0.0,
            "dispersion": 0.0,
            "heatmap_path": None,
            "mask_path": None,
            "error": str(e),
        }


def summarize_dual_side_analysis(front_result, back_result, label):
    front_score = float(front_result.get("risk_score", 0.0) or 0.0)
    back_score = float(back_result.get("risk_score", 0.0) or 0.0)
    avg_score = float(np.mean([front_score, back_score]))

    return {
        "label": label,
        "decision": _classify_forensic_risk(avg_score),
        "risk_score": round(avg_score, 2),
        "front_score": round(front_score, 2),
        "back_score": round(back_score, 2),
    }


def _parse_exif_datetime(value):
    raw = safe_str(value).strip()
    if not raw:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except Exception:
            continue
    return None


def run_metadata_analysis(front_exif, back_exif):
    records = [("Front", front_exif), ("Back", back_exif)]
    risk_score = 0.0
    signals = []
    valid_count = 0
    suspicious_software_hits = []
    software_values = []
    timestamps = []

    for side_label, exif_data in records:
        is_valid = isinstance(exif_data, dict) and exif_data and "error" not in exif_data
        if not is_valid:
            signals.append(f"{side_label} side has missing or unreadable EXIF metadata.")
            risk_score += 12.0
            continue

        valid_count += 1
        software_value = _normalize_space(exif_data.get("Software", ""))
        if software_value:
            software_values.append(software_value)
            if any(token in software_value.lower() for token in SUSPICIOUS_SOFTWARE_TERMS):
                suspicious_software_hits.append(f"{side_label}: {software_value}")
                risk_score += 36.0
            else:
                risk_score += 8.0
        else:
            risk_score += 5.0

        make = _normalize_space(exif_data.get("Make", ""))
        model = _normalize_space(exif_data.get("Model", ""))
        if not make and not model:
            signals.append(f"{side_label} side has no camera make/model metadata.")
            risk_score += 8.0

        exif_time = (
            _parse_exif_datetime(exif_data.get("DateTimeOriginal", ""))
            or _parse_exif_datetime(exif_data.get("DateTimeDigitized", ""))
            or _parse_exif_datetime(exif_data.get("DateTime", ""))
            or _parse_exif_datetime(exif_data.get("ModifyDate", ""))
        )
        if exif_time is not None:
            timestamps.append((side_label, exif_time))
        else:
            risk_score += 4.0

    if suspicious_software_hits:
        signals.append(f"Editing software markers found: {', '.join(suspicious_software_hits)}")
    elif software_values:
        signals.append(f"Software tag present: {', '.join(sorted(set(software_values)))}")

    if len(timestamps) == 2:
        delta_seconds = abs((timestamps[0][1] - timestamps[1][1]).total_seconds())
        if delta_seconds > (365 * 24 * 3600 * 5):
            signals.append("Front/back capture timestamps differ by several years.")
            risk_score += 15.0
        elif delta_seconds > (180 * 24 * 3600):
            signals.append("Front/back capture timestamps differ significantly.")
            risk_score += 8.0

    if valid_count == 0:
        risk_score = max(risk_score, 45.0)
        signals.append("Both sides are missing EXIF metadata. This can happen after image export/compression.")
    elif valid_count == 1:
        risk_score = max(risk_score, 30.0)
        signals.append("Only one side contains valid EXIF metadata.")

    risk_score = float(np.clip(risk_score, 0.0, 100.0))
    decision = _classify_forensic_risk(risk_score)
    confidence = float(np.clip(55.0 + (abs(risk_score - 50.0) * 0.7), 35.0, 98.0))

    if not signals:
        signals.append("Metadata appears consistent across available EXIF fields.")

    return {
        "decision": decision,
        "risk_score": round(risk_score, 2),
        "confidence": round(confidence, 2),
        "signals": signals,
        "valid_exif_sides": valid_count,
    }


def run_image_forgery_detection(results, ela_summary, noise_summary):
    front_obj = results.get("front", {}).get("object_verification", {}) or {}
    back_obj = results.get("back", {}).get("object_verification", {}) or {}

    tampered_front = bool(front_obj.get("is_tampered"))
    tampered_back = bool(back_obj.get("is_tampered"))
    tamper_suspected = bool(front_obj.get("tamper_suspected")) or bool(back_obj.get("tamper_suspected"))
    tampered_regions_detected = tampered_front or tampered_back
    tamper_confidence = max(
        float(front_obj.get("tamper_confidence", 0.0) or 0.0),
        float(back_obj.get("tamper_confidence", 0.0) or 0.0),
    )

    human_detected = bool(results.get("front", {}).get("face_detection", {}).get("human_detected", False))

    front_exif = results.get("front", {}).get("exif_analysis", {}) or {}
    back_exif = results.get("back", {}).get("exif_analysis", {}) or {}
    exif_records = [front_exif, back_exif]

    has_any_exif = any(isinstance(item, dict) and item and "error" not in item for item in exif_records)
    software_values = []
    for record in exif_records:
        if isinstance(record, dict) and "error" not in record:
            software = safe_str(record.get("Software", "")).strip()
            if software:
                software_values.append(software)

    suspicious_software_values = [
        item for item in software_values if any(token in item.lower() for token in SUSPICIOUS_SOFTWARE_TERMS)
    ]
    qr_analysis = results.get("back", {}).get("qr_analysis", {}) or {}
    qr_decode_failed = bool(isinstance(qr_analysis, dict) and "error" in qr_analysis)
    cross_side_analysis = results.get("cross_side_analysis", {}) or {}
    cross_side_risk = float(cross_side_analysis.get("risk_score", 0.0) or 0.0)

    ela_score = float(ela_summary.get("risk_score", 0.0) or 0.0)
    noise_score = float(noise_summary.get("risk_score", 0.0) or 0.0)
    if tampered_regions_detected:
        object_risk = float(np.clip(55.0 + (tamper_confidence * 35.0), 60.0, 100.0))
    elif tamper_suspected:
        object_risk = float(np.clip(25.0 + (tamper_confidence * 25.0), 25.0, 55.0))
    else:
        object_risk = 12.0 if not human_detected else 6.0

    metadata_analysis = results.get("metadata_analysis", {}) or {}
    metadata_risk_from_module = float(metadata_analysis.get("risk_score", 0.0) or 0.0)
    if metadata_risk_from_module > 0:
        metadata_risk = metadata_risk_from_module
    elif suspicious_software_values:
        metadata_risk = 80.0
    elif software_values:
        metadata_risk = 55.0
    elif not has_any_exif:
        metadata_risk = 45.0
    else:
        metadata_risk = 15.0

    weighted_risk = float(
        np.clip(
            (ela_score * 0.30)
            + (noise_score * 0.26)
            + (object_risk * 0.17)
            + (metadata_risk * 0.09)
            + (cross_side_risk * 0.18),
            0.0,
            100.0,
        )
    )

    boost = 0.0
    if ela_score >= 60 and noise_score >= 55:
        boost += 12.0
    if tampered_regions_detected and (ela_score >= 50 or noise_score >= 50):
        boost += 8.0
    if suspicious_software_values and (ela_score >= 40 or noise_score >= 40):
        boost += 8.0
    if not has_any_exif and (ela_score >= 50 or noise_score >= 50):
        boost += 6.0
    if metadata_risk >= 60 and (ela_score >= 30 or noise_score >= 30):
        boost += 8.0
    if cross_side_risk >= 80:
        boost += 18.0
    elif cross_side_risk >= 55:
        boost += 8.0

    weighted_risk = float(np.clip(weighted_risk + boost, 0.0, 100.0))
    if tampered_regions_detected:
        weighted_risk = max(weighted_risk, 58.0)
    if ela_score >= 55 or noise_score >= 55:
        weighted_risk = max(weighted_risk, 52.0)
    elif ela_score >= 35 or noise_score >= 35:
        weighted_risk = max(weighted_risk, 35.0)
    if not has_any_exif:
        weighted_risk = max(weighted_risk, 28.0)
    if qr_decode_failed:
        weighted_risk = max(weighted_risk, 30.0)
    if metadata_risk >= 70:
        weighted_risk = max(weighted_risk, 40.0)
    if cross_side_risk >= 80:
        weighted_risk = max(weighted_risk, 67.0)
    elif cross_side_risk >= 55:
        weighted_risk = max(weighted_risk, 45.0)

    if weighted_risk >= 67:
        decision = "Likely Forged"
    elif weighted_risk >= 42:
        decision = "Needs Manual Review"
    else:
        decision = "Likely Authentic"

    signals = []
    if tampered_regions_detected:
        signals.append("Object detector flagged one or more tampered regions.")
        if tamper_confidence > 0:
            signals.append(f"Tamper-class confidence peaked at {round(tamper_confidence * 100.0, 1)}%.")
    elif tamper_suspected:
        signals.append("Object detector found weak tamper-like regions; confidence was below strong threshold.")
    if not human_detected:
        signals.append("No human face candidate was detected on the front side.")
    if suspicious_software_values:
        signals.append(f"Editing software marker found in EXIF: {', '.join(sorted(set(suspicious_software_values)))}")
    elif software_values:
        signals.append(f"EXIF software tag present: {', '.join(sorted(set(software_values)))}")
    elif not has_any_exif:
        signals.append("No EXIF metadata found on either side.")
    if qr_decode_failed:
        signals.append("QR data could not be decoded, reducing trust in machine verification.")
    if cross_side_risk >= 55:
        signals.append(safe_str(cross_side_analysis.get("reason", "Front/back similarity is unusually high.")))
    if ela_score >= 30:
        signals.append(f"ELA average risk is elevated ({round(ela_score, 1)}%).")
    if noise_score >= 30:
        signals.append(f"Noise inconsistency level is elevated ({round(noise_score, 1)}%).")
    if not signals:
        signals.append("No strong forgery signals were detected across current checks.")

    base_confidence = float(results.get("average_confidence", 0.0) * 100.0)
    confidence = float(
        np.clip(
            (base_confidence * 0.50) + (50.0 + (abs(weighted_risk - 50.0) * 0.8)),
            0.0,
            99.0,
        )
    )

    return {
        "decision": decision,
        "risk_score": round(weighted_risk, 2),
        "confidence": round(confidence, 2),
        "signals": signals,
        "tampered_regions_detected": tampered_regions_detected,
        "component_scores": {
            "ela": round(ela_score, 2),
            "noise": round(noise_score, 2),
            "object_detection": round(object_risk, 2),
            "metadata": round(metadata_risk, 2),
            "cross_side_similarity": round(cross_side_risk, 2),
        },
    }


def _derive_image_authenticity(results):
    forgery = results.get("image_forgery_detection", {}) or {}
    forgery_decision = safe_str(forgery.get("decision", ""))
    forgery_risk = float(forgery.get("risk_score", 0.0) or 0.0)
    fraud_score = int(results.get("fraud_score", 0) or 0)

    if forgery_decision == "Likely Forged" or forgery_risk >= 67 or fraud_score >= 5:
        verdict = "TAMPERED IMAGE"
        is_real = False
    elif forgery_decision == "Needs Manual Review" or forgery_risk >= 42 or fraud_score >= 2:
        verdict = "SUSPICIOUS IMAGE (MANUAL REVIEW REQUIRED)"
        is_real = False
    else:
        verdict = "LIKELY REAL IMAGE"
        is_real = True

    return {
        "verdict": verdict,
        "is_real": is_real,
        "forgery_decision": forgery_decision or "Unavailable",
        "forgery_risk_score": round(forgery_risk, 2),
        "fraud_score": fraud_score,
    }

# -------------------------
# QR decode using OpenCV QRCodeDetector and Aadhaar parsing
# -------------------------
def _decode_qr_payload_with_variants(image_bgr):
    detector = cv2.QRCodeDetector()

    image_bgr = _resize_to_max_dim(image_bgr, QR_MAX_DECODE_DIM)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    _, otsu = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        clahe_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        8,
    )

    quick_variants = [
        ("orig", image_bgr),
        ("gray", gray),
        ("clahe", clahe_img),
    ]

    for variant_name, base in quick_variants:
        angles = (0,) if FAST_ANALYSIS_MODE else (0, 90, 270)
        for angle in angles:
            rotated = _rotate_image(base, angle)
            scales = (1.0,) if FAST_ANALYSIS_MODE else (1.0, 1.35)
            for scale in scales:
                if scale == 1.0:
                    candidate = rotated
                else:
                    candidate = cv2.resize(rotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                candidate = _ensure_bgr(candidate)
                try:
                    decoded, _, _ = detector.detectAndDecode(candidate)
                except Exception:
                    decoded = ""
                if decoded:
                    return decoded, f"{variant_name}_rot{angle}_x{scale:.1f}"

    fallback_variants = [
        ("otsu", otsu),
        ("adaptive", adaptive),
        ("clahe", clahe_img),
    ]
    if FAST_ANALYSIS_MODE:
        fallback_variants = fallback_variants[:1]
    for variant_name, base in fallback_variants:
        for angle in (0, 180):
            rotated = _rotate_image(base, angle)
            for scale in (1.0, 1.6):
                if scale == 1.0:
                    candidate = rotated
                else:
                    candidate = cv2.resize(rotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                candidate = _ensure_bgr(candidate)
                try:
                    decoded, _, _ = detector.detectAndDecode(candidate)
                except Exception:
                    decoded = ""
                if decoded:
                    return decoded, f"{variant_name}_rot{angle}_x{scale:.1f}"

                if FAST_ANALYSIS_MODE:
                    continue
                try:
                    decoded_curved, _, _ = detector.detectAndDecodeCurved(candidate)
                except Exception:
                    decoded_curved = ""
                if decoded_curved:
                    return decoded_curved, f"{variant_name}_curved_rot{angle}_x{scale:.1f}"

                if FAST_ANALYSIS_MODE:
                    continue
                try:
                    retval, decoded_info, _, _ = detector.detectAndDecodeMulti(candidate)
                except Exception:
                    retval, decoded_info = False, []
                if retval and decoded_info:
                    for entry in decoded_info:
                        if entry:
                            return entry, f"{variant_name}_multi_rot{angle}_x{scale:.1f}"

    return "", ""


def _enrich_qr_parsed_fields(parsed, payload):
    if not isinstance(parsed, dict):
        return parsed

    enriched = dict(parsed)
    lower_map = {}
    for key, value in parsed.items():
        key_text = safe_str(key).strip().lower()
        value_text = _normalize_space(value)
        if key_text and value_text:
            lower_map[key_text] = value_text

    def pick(*keys):
        for key in keys:
            val = lower_map.get(key)
            if val:
                return val
        return ""

    uid = pick("uid", "uidai", "aadhaar", "aadhar", "uidnum", "uidnumber")
    if not uid:
        uid_match = re.search(r"(?<!\d)(\d{12})(?!\d)", safe_str(payload))
        if uid_match:
            uid = uid_match.group(1)
    if uid:
        enriched["uid"] = re.sub(r"\D", "", uid)[:12]

    name = pick("name", "nm", "n")
    if name:
        enriched["name"] = name

    dob = pick("dob", "dateofbirth", "birthdate", "yob")
    if dob:
        enriched["dob"] = dob

    gender = pick("gender", "sex", "g")
    if gender:
        gender_lower = gender.lower()
        if gender_lower in ("m", "male"):
            enriched["gender"] = "Male"
        elif gender_lower in ("f", "female"):
            enriched["gender"] = "Female"
        else:
            enriched["gender"] = gender

    address = pick("address", "addr", "location")
    if address:
        enriched["address"] = address

    return enriched


def _parse_qr_payload(payload):
    payload = safe_str(payload).strip()
    if not payload:
        return {"error": "Decoded QR payload was empty"}

    try:
        root = ET.fromstring(payload)
        parsed = {}
        if root.attrib:
            parsed.update(root.attrib)
        else:
            for child in root:
                parsed[safe_str(child.tag)] = safe_str(child.text)
        if parsed:
            return {"raw": payload, "parsed": _enrich_qr_parsed_fields(parsed, payload), "format": "xml"}
    except Exception:
        pass

    try:
        parsed_json = json.loads(payload)
        if isinstance(parsed_json, dict):
            return {"raw": payload, "parsed": _enrich_qr_parsed_fields(parsed_json, payload), "format": "json"}
    except Exception:
        pass

    parsed = {}
    uid_match = re.search(r"(?<!\d)(\d{12})(?!\d)", payload)
    if uid_match:
        parsed["uid"] = uid_match.group(1)

    name_match = re.search(
        r"(?:name|nm|n)[=:\s\"']{1,3}([A-Za-z][A-Za-z.\-'\s]{2,120})",
        payload,
        re.IGNORECASE,
    )
    if name_match:
        parsed["name"] = _normalize_space(name_match.group(1))

    dob_match = re.search(r"(?:\d{2}[-/]\d{2}[-/]\d{4})|(?:\d{4}[-/]\d{2}[-/]\d{2})", payload)
    if dob_match:
        parsed["dob"] = dob_match.group(0)

    gender_match = re.search(r"\b(?:gender|sex|g)[=:\s\"']{1,3}(male|female|m|f)\b", payload, re.IGNORECASE)
    if gender_match:
        parsed["gender"] = gender_match.group(1)

    addr_match = re.search(r"(?:addr|address|house|loc|street)[=:\s\"']{1,3}(.{10,250})", payload, re.IGNORECASE)
    if addr_match:
        parsed["address"] = _normalize_space(addr_match.group(1))

    if parsed:
        return {"raw": payload, "parsed": _enrich_qr_parsed_fields(parsed, payload), "format": "fallback"}

    return {"raw": payload, "format": "raw"}


def decode_qr_with_opencv(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read image for QR decoding."}

        payload, decode_variant = _decode_qr_payload_with_variants(img)
        if not payload:
            return {"error": "QR Code not found or could not be read"}

        parsed_result = _parse_qr_payload(payload)
        if "error" in parsed_result:
            return parsed_result
        parsed_result["decode_variant"] = decode_variant
        return parsed_result
    except Exception as e:
        return {"error": str(e)}


def decode_qr_from_card_images(back_image_path, front_image_path):
    errors = []
    attempts = [("Back", back_image_path)]
    if safe_str(front_image_path) != safe_str(back_image_path):
        attempts.append(("Front", front_image_path))

    for side_label, image_path in attempts:
        qr_result = decode_qr_with_opencv(image_path)
        if isinstance(qr_result, dict) and "error" not in qr_result:
            qr_result["source_side"] = side_label
            return qr_result
        errors.append(f"{side_label}: {safe_str(qr_result.get('error', 'QR decode failed'))}")

    return {
        "error": "QR code could not be decoded from either side image",
        "attempts": errors,
    }

# -------------------------
# OCR extraction from YOLO text detections
# -------------------------
def extract_aadhaar_data(image_path, text_model_raw_results, side_label="front"):
    try:
        image = np.array(Image.open(image_path).convert("RGB"))
        aadhaar_data = {}
        confidences = []
        field_scores = {}

        key_mapping = {
            'NAME': 'Name',
            'AADHAR_NUMBER': 'Aadhaar Number',
            'AADHAAR_NUMBER': 'Aadhaar Number',
            'AADHARNO': 'Aadhaar Number',
            'GENDER': 'Gender',
            'DATE_OF_BIRTH': 'Date of Birth',
            'DOB': 'Date of Birth',
            'ADDRESS': 'Address'
        }

        if text_model_raw_results is not None:
            detections = Detections.from_ultralytics(text_model_raw_results)
            xyxy_raw = getattr(detections, "xyxy", None)
            confs_raw = getattr(detections, "confidence", None)
            xyxy = list(xyxy_raw) if xyxy_raw is not None else []
            confs = list(confs_raw) if confs_raw is not None else []
            class_names = []
            if hasattr(detections, "data"):
                names_raw = detections.data.get("class_name", [])
                class_names = list(names_raw) if names_raw is not None else []
            if len(class_names) < len(xyxy):
                class_names.extend(["Unknown"] * (len(xyxy) - len(class_names)))

            ordered_regions = []
            for bbox, conf, cls_name in zip(xyxy, confs, class_names):
                try:
                    conf_value = float(conf)
                except Exception:
                    conf_value = 0.0
                confidences.append(conf_value)

                x1, y1, x2, y2 = map(int, bbox)
                h, w = image.shape[:2]
                x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
                y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
                if y2 <= y1 or x2 <= x1:
                    continue

                roi = image[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                cls_name_str = re.sub(r"[^A-Z_]", "_", safe_str(cls_name).upper()).strip("_")
                normalized_key = key_mapping.get(cls_name_str, cls_name_str.title().replace("_", " "))
                ordered_regions.append((y1, x1, normalized_key, roi))

            ordered_regions.sort(key=lambda item: (item[0], item[1]))
            for _, _, field_name, roi in ordered_regions:
                text, ocr_score = _run_adaptive_ocr(roi, field_name)
                if not text:
                    continue
                previous_score = float(field_scores.get(field_name, -1.0))
                if ocr_score >= previous_score:
                    aadhaar_data[field_name] = text
                    field_scores[field_name] = float(ocr_score)

        primary_fields = ("Name", "Date of Birth", "Gender", "Aadhaar Number")
        side_hint = safe_str(side_label).strip().lower()
        missing_fields = {name for name in primary_fields if not _normalize_space(aadhaar_data.get(name, ""))}
        if not _normalize_space(aadhaar_data.get("Address", "")):
            missing_fields.add("Address")

        should_run_fallback = bool(missing_fields)
        if side_hint == "back":
            # Back side typically contributes address/QR context; avoid unnecessary full fallback OCR.
            should_run_fallback = "Address" in missing_fields or len(missing_fields) >= 3

        if should_run_fallback:
            fallback_fast_mode = FAST_ANALYSIS_MODE and bool(text_model_raw_results is not None)
            fallback_fields = _extract_fallback_fields_from_image(
                image,
                required_fields=missing_fields,
                fast_mode=fallback_fast_mode,
            )
            for field_name, candidate_value in fallback_fields.items():
                existing_value = aadhaar_data.get(field_name, "")
                existing_score = _score_document_field(
                    field_name,
                    existing_value,
                    float(field_scores.get(field_name, 0.0) or 0.0),
                )
                candidate_score = _score_document_field(field_name, candidate_value, 55.0)
                if (not existing_value) or (candidate_score >= existing_score):
                    merged = _merge_ocr_field(existing_value, candidate_value, field_name)
                    if merged:
                        aadhaar_data[field_name] = merged

        for field_name in list(aadhaar_data.keys()):
            aadhaar_data[field_name] = _sanitize_ocr_value(field_name, aadhaar_data[field_name])

        return {"data": aadhaar_data, "confidences": confidences}
    except Exception as e:
        return {"error": f"OCR failed: {e}"}

# -------------------------
# Object verification wrapper
# -------------------------
def run_object_verification(image_path, object_model_raw_results):
    if object_detection_model is None:
        return {"error": "Object detection model not available."}
    try:
        detected_objects = []
        confidences = []
        tamper_confidences = []
        tamper_labels = []
        tamper_keywords = ("tampered", "tamper", "forged", "forgery", "edited", "manipulated", "fake")
        if object_model_raw_results is None:
            return {
                "detected_objects": [],
                "is_tampered": False,
                "tamper_suspected": False,
                "tamper_detection_count": 0,
                "confidences": [],
                "tamper_confidence": 0.0,
            }
        for box in getattr(object_model_raw_results, "boxes", []):
            try:
                class_id = int(box.cls[0])
                class_name = object_detection_model.names.get(class_id, str(class_id)) if isinstance(object_detection_model.names, dict) else object_detection_model.names[class_id]
            except Exception:
                class_name = "Unknown"
            try:
                confidence = float(box.conf[0])
            except Exception:
                confidence = 0.0
            detected_objects.append(class_name)
            class_lower = safe_str(class_name).lower()
            if any(token in class_lower for token in tamper_keywords) and confidence >= 0.30:
                tamper_confidences.append(confidence)
                tamper_labels.append(class_name)
            confidences.append(confidence)

        strong_hits = [c for c in tamper_confidences if c >= OBJECT_TAMPER_STRONG_CONFIDENCE]
        moderate_hits = [c for c in tamper_confidences if c >= OBJECT_TAMPER_MIN_CONFIDENCE]
        is_tampered = bool(strong_hits) or len(moderate_hits) >= 2
        tamper_suspected = bool(moderate_hits) and not is_tampered
        tamper_confidence = max(tamper_confidences) if tamper_confidences else 0.0

        return {
            "detected_objects": list(set(detected_objects)),
            "is_tampered": is_tampered,
            "tamper_suspected": tamper_suspected,
            "tamper_detection_count": len(moderate_hits),
            "confidences": confidences,
            "tamper_confidence": round(float(tamper_confidence), 4),
            "tamper_labels": list(set(tamper_labels)),
        }
    except Exception as e:
        return {"error": f"Object verification failed: {e}"}

# -------------------------
# Annotate image using supervision
# -------------------------
def create_annotated_image(image_path, text_model_results, object_model_results):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        try:
            if text_model_results is not None:
                text_detections = Detections.from_ultralytics(text_model_results)
                text_box_annotator = sv.BoxAnnotator(color=sv.Color.BLUE, thickness=2)
                text_label_annotator = sv.LabelAnnotator(color=sv.Color.BLUE, text_color=sv.Color.WHITE, text_scale=0.5)
                image = text_box_annotator.annotate(scene=image.copy(), detections=text_detections)
                image = text_label_annotator.annotate(scene=image, detections=text_detections)
        except Exception as e:
            print(f"Warning annotate text: {e}")
        try:
            if object_model_results is not None:
                object_detections = Detections.from_ultralytics(object_model_results)
                object_box_annotator = sv.BoxAnnotator(color=sv.Color.RED, thickness=2)
                object_label_annotator = sv.LabelAnnotator(color=sv.Color.RED, text_color=sv.Color.WHITE, text_scale=0.5)
                image = object_box_annotator.annotate(scene=image, detections=object_detections)
                image = object_label_annotator.annotate(scene=image, detections=object_detections)
        except Exception as e:
            print(f"Warning annotate object: {e}")

        annotated_filename = "annotated_" + os.path.basename(image_path)
        save_dir = "static"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, annotated_filename)
        cv2.imwrite(save_path, image)
        return annotated_filename
    except Exception as e:
        print(f"Error creating annotated image: {e}")
        return None

# -------------------------
# Main analysis pipeline
# -------------------------
def analyze_aadhar_pair(front_path, back_path):
    results = {}
    try:
        # Text model predictions (best-effort)
        text_front = None
        text_back = None
        if aadhaar_model is not None:
            try:
                text_batch = aadhaar_model.predict([front_path, back_path], verbose=False)
                if isinstance(text_batch, (list, tuple)):
                    if len(text_batch) > 0:
                        text_front = text_batch[0]
                    if len(text_batch) > 1:
                        text_back = text_batch[1]
            except Exception:
                try:
                    text_front = aadhaar_model.predict(front_path, verbose=False)[0]
                except Exception as e:
                    print(f"Text model front fail: {e}")
                try:
                    text_back = aadhaar_model.predict(back_path, verbose=False)[0]
                except Exception as e:
                    print(f"Text model back fail: {e}")

        with ThreadPoolExecutor(max_workers=2) as executor:
            front_ocr_future = executor.submit(extract_aadhaar_data, front_path, text_front, "front")
            back_ocr_future = executor.submit(extract_aadhaar_data, back_path, text_back, "back")
            front_ocr = front_ocr_future.result()
            back_ocr = back_ocr_future.result()

        front_ocr_data = front_ocr.get("data", {}) if isinstance(front_ocr, dict) else {}
        back_ocr_data = back_ocr.get("data", {}) if isinstance(back_ocr, dict) else {}

        # object detection (tamper)
        obj_front = None
        obj_back = None
        if object_detection_model is not None:
            try:
                object_batch = object_detection_model.predict([front_path, back_path], verbose=False)
                if isinstance(object_batch, (list, tuple)):
                    if len(object_batch) > 0:
                        obj_front = object_batch[0]
                    if len(object_batch) > 1:
                        obj_back = object_batch[1]
            except Exception:
                try:
                    obj_front = object_detection_model(front_path, verbose=False)[0]
                except Exception as e:
                    print(f"Object detect front fail: {e}")
                try:
                    obj_back = object_detection_model(back_path, verbose=False)[0]
                except Exception as e:
                    print(f"Object detect back fail: {e}")

        object_results_front = run_object_verification(front_path, obj_front)
        object_results_back = run_object_verification(back_path, obj_back)
        if text_back is None and text_front is not None and safe_str(front_path) == safe_str(back_path):
            text_back = text_front
        if obj_back is None and obj_front is not None and safe_str(front_path) == safe_str(back_path):
            obj_back = obj_front

        # confidences
        all_conf = []
        for r in (obj_front, obj_back, text_front, text_back):
            try:
                if r and getattr(r, "boxes", None):
                    all_conf.extend(r.boxes.conf.tolist())
            except Exception:
                pass
        average_confidence = float(np.mean(all_conf)) if all_conf else 0.0

        # exif
        exif_front = get_exif_data(front_path)
        exif_back = get_exif_data(back_path)
        metadata_analysis = run_metadata_analysis(exif_front, exif_back)
        cross_side_analysis = _compute_cross_side_analysis(front_path, back_path)

        # qr decode with robust fallbacks (back first, then front)
        qr_results = decode_qr_from_card_images(back_path, front_path)

        # general YOLO detection to find person/human
        general_labels = []
        human_detected = False
        if general_model is not None:
            try:
                gres = general_model(front_path, verbose=False)[0]
                if getattr(gres, "boxes", None):
                    general_labels = [general_model.names[int(cls)] for cls in gres.boxes.cls]
                    human_detected = "person" in general_labels
            except Exception as e:
                print(f"General model inference failed: {e}")

        # assemble results
        results = {
            "front": {
                "object_verification": object_results_front,
                "exif_analysis": exif_front,
                "ocr_analysis": front_ocr_data,
                "face_detection": {"human_detected": human_detected, "detected_objects": general_labels},
            },
            "back": {
                "object_verification": object_results_back,
                "exif_analysis": exif_back,
                "ocr_analysis": back_ocr_data,
                "qr_analysis": qr_results,
                "general_detection": {"human_detected": human_detected, "detected_objects": general_labels}
            },
            "average_confidence": average_confidence,
            "fraud_indicators": [],
            "metadata_analysis": metadata_analysis,
            "cross_side_analysis": cross_side_analysis,
            "raw_results": {
                "text_front": text_front,
                "text_back": text_back,
                "object_front": obj_front,
                "object_back": obj_back
            }
        }

        # combined OCR
        combined_ocr = dict(front_ocr_data) if isinstance(front_ocr_data, dict) else {}
        if isinstance(back_ocr_data, dict):
            for key, value in back_ocr_data.items():
                normalized_value = _sanitize_ocr_value(key, value)
                if not normalized_value:
                    continue
                if key not in combined_ocr or not safe_str(combined_ocr.get(key)).strip():
                    combined_ocr[key] = normalized_value
                    continue
                if key == "Address" and len(normalized_value) > len(safe_str(combined_ocr.get(key))):
                    combined_ocr[key] = normalized_value
        results["combined_ocr"] = combined_ocr

        # image forensics (ELA, noise, forgery synthesis)
        forensic_futures = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            forensic_futures["ela_front"] = executor.submit(perform_ela_analysis, front_path, "Front")
            forensic_futures["ela_back"] = executor.submit(perform_ela_analysis, back_path, "Back")
            forensic_futures["noise_front"] = executor.submit(perform_noise_analysis, front_path, "Front")
            forensic_futures["noise_back"] = executor.submit(perform_noise_analysis, back_path, "Back")

            ela_front = forensic_futures["ela_front"].result()
            ela_back = forensic_futures["ela_back"].result()
            noise_front = forensic_futures["noise_front"].result()
            noise_back = forensic_futures["noise_back"].result()

        ela_summary = summarize_dual_side_analysis(ela_front, ela_back, "ELA")
        noise_summary = summarize_dual_side_analysis(noise_front, noise_back, "Noise")

        results["ela_analysis"] = {
            "front": ela_front,
            "back": ela_back,
            "summary": ela_summary,
        }
        results["noise_analysis"] = {
            "front": noise_front,
            "back": noise_back,
            "summary": noise_summary,
        }
        results["image_forgery_detection"] = run_image_forgery_detection(results, ela_summary, noise_summary)

        # heuristics & fraud scoring
        fraud_score = 0
        front_tampered = isinstance(object_results_front, dict) and object_results_front.get("is_tampered")
        back_tampered = isinstance(object_results_back, dict) and object_results_back.get("is_tampered")
        front_tamper_suspected = isinstance(object_results_front, dict) and object_results_front.get("tamper_suspected")
        back_tamper_suspected = isinstance(object_results_back, dict) and object_results_back.get("tamper_suspected")

        if front_tampered:
            results["fraud_indicators"].append("Tampered region detected on the front of the card.")
            fraud_score += 2

        if back_tampered:
            results["fraud_indicators"].append("Tampered region detected on the back of the card.")
            fraud_score += 1

        cross_side_risk = float((cross_side_analysis or {}).get("risk_score", 0.0) or 0.0)
        if cross_side_risk >= 80:
            results["fraud_indicators"].append("Front and back images appear near-identical, which is highly suspicious.")
            fraud_score += 3
        elif cross_side_risk >= 55:
            results["fraud_indicators"].append("Front and back images are unusually similar; verify document sides manually.")
            fraud_score += 2

        if (front_tamper_suspected or back_tamper_suspected) and not (front_tampered or back_tampered):
            results["fraud_indicators"].append("Weak tamper-like regions were detected; confidence is below strong threshold.")
        elif not human_detected:
            results["fraud_indicators"].append("No human detected in the photo area; this may happen on low-quality uploads.")

        # compare QR parsed values if available
        if isinstance(qr_results, dict) and 'error' not in qr_results:
            parsed = qr_results.get("parsed", {}) if isinstance(qr_results.get("parsed", {}), dict) else {}
            ocr_name = safe_str(combined_ocr.get("Name", "")).strip().lower()
            qr_name = safe_str(parsed.get("name") or parsed.get("Name") or qr_results.get("raw", "")).strip().lower()
            name_similarity = _text_similarity(ocr_name, qr_name)
            name_overlap = _token_overlap_ratio(ocr_name, qr_name)
            if (
                ocr_name and qr_name
                and len(ocr_name) >= 4 and len(qr_name) >= 4
                and name_similarity < NAME_MISMATCH_SIMILARITY_THRESHOLD
                and name_overlap < 0.34
            ):
                results["fraud_indicators"].append("Mismatch: Printed Name vs. QR Code Name.")
                fraud_score += 1

            ocr_gender = safe_str(combined_ocr.get("Gender", "")).strip().lower()
            qr_gender = safe_str(parsed.get("gender") or parsed.get("sex") or parsed.get("g", "")).strip().lower()
            if ocr_gender and qr_gender and ocr_gender != qr_gender:
                results["fraud_indicators"].append("Mismatch between printed gender and QR code gender.")
                fraud_score += 1

            ocr_dob = safe_str(combined_ocr.get("Date of Birth", "")).replace("-", "/").strip()
            qr_dob = safe_str(parsed.get("dob") or parsed.get("yob") or parsed.get("dob_raw", "")).replace("-", "/").strip()
            if ocr_dob and qr_dob and not _dob_consistent(ocr_dob, qr_dob):
                results["fraud_indicators"].append("Mismatch between printed DOB and QR DOB.")
                fraud_score += 1

            ocr_num = re.sub(r'\s+', '', safe_str(combined_ocr.get("Aadhaar Number", ""))).strip()
            qr_num = re.sub(r'\s+', '', safe_str(parsed.get("uid") or parsed.get("aadhar") or parsed.get("aadhaar") or parsed.get("uidnum", ""))).strip()
            if ocr_num and qr_num and ocr_num != qr_num:
                results["fraud_indicators"].append("Mismatch between printed Aadhaar number and QR Aadhaar number.")
                fraud_score += 1

            ocr_addr = safe_str(combined_ocr.get("Address", "")).strip().lower()
            qr_addr = safe_str(parsed.get("address") or parsed.get("addr") or parsed.get("location", "")).strip().lower()
            addr_similarity = _text_similarity(ocr_addr, qr_addr)
            if (
                ocr_addr and qr_addr
                and len(ocr_addr) >= 12 and len(qr_addr) >= 12
                and ocr_addr not in qr_addr and qr_addr not in ocr_addr
                and addr_similarity < ADDRESS_MISMATCH_SIMILARITY_THRESHOLD
            ):
                results["fraud_indicators"].append("Mismatch between printed address and QR code address.")
                fraud_score += 1
        else:
            results["fraud_indicators"].append("QR code could not be decoded from either side image.")
            fraud_score += 1

        # Aadhaar checksum validation on extracted front OCR Aadhaar number
        try:
            front_aad = combined_ocr.get("Aadhaar Number", "")
            clean_num = re.sub(r'\D', '', front_aad)
            if clean_num and len(clean_num) == 12 and clean_num.isdigit():
                is_valid = validate_checksum(clean_num)
                results["aadhaar_validation"] = {"valid": is_valid, "reason": "Valid Aadhaar number" if is_valid else "Invalid checksum", "clean_number": clean_num}
                if not is_valid:
                    qr_num = ""
                    if isinstance(qr_results, dict) and 'error' not in qr_results:
                        parsed = qr_results.get("parsed", {}) if isinstance(qr_results.get("parsed", {}), dict) else {}
                        qr_num = re.sub(r'\D', '', safe_str(parsed.get("uid") or parsed.get("aadhar") or parsed.get("aadhaar") or parsed.get("uidnum", ""))).strip()
                    if not qr_num or qr_num != clean_num:
                        results["fraud_indicators"].append(f"Invalid Aadhaar number: {results['aadhaar_validation']['reason']}")
                        fraud_score += 1
            else:
                results["aadhaar_validation"] = {"valid": False, "reason": "Aadhaar not found or invalid format"}
        except Exception:
            results["aadhaar_validation"] = {"valid": False, "reason": "Validation error"}

        # EXIF presence
        exif_front_ok = isinstance(exif_front, dict) and 'error' not in exif_front and len(exif_front) > 0
        exif_back_ok = isinstance(exif_back, dict) and 'error' not in exif_back and len(exif_back) > 0
        if not exif_front_ok and not exif_back_ok:
            results["fraud_indicators"].append("No EXIF metadata found")

        metadata_decision = safe_str(metadata_analysis.get("decision", ""))
        if metadata_decision == "Likely Tampered":
            results["fraud_indicators"].append("Metadata analysis indicates likely tampered image history.")
            fraud_score += 1
        elif metadata_decision == "Suspicious":
            results["fraud_indicators"].append("Metadata analysis indicates suspicious EXIF consistency.")

        # integrate forensic module outputs into indicator list
        if ela_summary.get("risk_score", 0) >= 60:
            results["fraud_indicators"].append("ELA analysis shows high recompression inconsistency.")
            fraud_score += 1
        elif ela_summary.get("risk_score", 0) >= 30:
            results["fraud_indicators"].append("ELA analysis suggests moderate inconsistency.")

        if noise_summary.get("risk_score", 0) >= 60:
            results["fraud_indicators"].append("Noise analysis detected unusual local variance patterns.")
            fraud_score += 1
        elif noise_summary.get("risk_score", 0) >= 30:
            results["fraud_indicators"].append("Noise analysis suggests mild inconsistency.")

        forgery_detection = results.get("image_forgery_detection", {}) or {}
        forgery_decision = safe_str(forgery_detection.get("decision", ""))
        if forgery_decision == "Likely Forged":
            results["fraud_indicators"].append("Image forgery detector indicates likely forged visuals.")
            fraud_score += 2
        elif forgery_decision == "Needs Manual Review":
            results["fraud_indicators"].append("Image forgery detector recommends manual review.")
            fraud_score += 0

        # final score/assessment
        results["fraud_score"] = fraud_score
        results["assessment"] = "HIGH FRAUD RISK" if fraud_score >= 5 else ("MODERATE FRAUD RISK" if fraud_score >= 2 else "LOW FRAUD RISK")
        results["image_authenticity"] = _derive_image_authenticity(results)

        return results
    except Exception as e:
        return {"error": str(e)}

# -------------------------
# Template transform helper
# -------------------------
def transform_results_for_template(results):
    risk_level = safe_str(results.get("assessment", "UNKNOWN")).replace(" FRAUD RISK", "")
    risk_score = int(np.clip(results.get("fraud_score", 0) * 20, 0, 100))
    confidence_score = int(np.clip(results.get("average_confidence", 0) * 100, 0, 100))

    color_map = {
        "HIGH": "border-red-500 text-red-900 bg-red-50",
        "MODERATE": "border-amber-500 text-amber-900 bg-amber-50",
        "LOW": "border-green-500 text-green-900 bg-green-50",
        "UNKNOWN": "border-slate-500 text-slate-900 bg-slate-50",
    }

    def risk_badge_class(score):
        if score >= 60:
            return "bg-red-50 border-red-200 text-red-700"
        if score >= 30:
            return "bg-amber-50 border-amber-200 text-amber-700"
        return "bg-green-50 border-green-200 text-green-700"

    def risk_label(score):
        if score >= 60:
            return "High Risk"
        if score >= 30:
            return "Review"
        return "Normal"

    indicators = []
    for desc in results.get("fraud_indicators", []):
        lower_desc = safe_str(desc).lower()
        if any(token in lower_desc for token in ("tampered", "mismatch", "invalid", "forgery", "high")):
            severity = "high"
        elif any(token in lower_desc for token in ("suspicious", "review", "no exif", "noise", "ela")):
            severity = "medium"
        else:
            severity = "low"
        badge_map = {
            "high": "border-red-300 bg-red-100 text-red-800",
            "medium": "border-amber-300 bg-amber-100 text-amber-800",
            "low": "border-green-300 bg-green-100 text-green-800",
        }
        indicators.append(
            {
                "type": safe_str(desc).split(":")[0],
                "severity": severity,
                "description": desc,
                "badge_class": badge_map.get(severity, ""),
            }
        )

    if not indicators:
        indicators = [
            {
                "type": "No major anomalies",
                "severity": "low",
                "description": "No major fraud indicators were detected by the current rule set.",
                "badge_class": "border-green-300 bg-green-100 text-green-800",
            }
        ]

    icon_map = {
        "Name": "M5.121 17.804A11.953 11.953 0 0112 15c2.5 0 4.847.765 6.879 2.074M15 11a3 3 0 11-6 0 3 3 0 016 0z",
        "Date of Birth": "M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z",
        "Gender": "M7 7h10M7 12h6m-6 5h10",
        "Aadhaar Number": "M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z",
    }
    ocr_front = results.get("front", {}).get("ocr_analysis", {}) or {}
    ocr_front_list = [
        {"label": "Name", "value": ocr_front.get("Name", "N/A"), "icon": icon_map["Name"]},
        {"label": "Date of Birth", "value": ocr_front.get("Date of Birth", "N/A"), "icon": icon_map["Date of Birth"]},
        {"label": "Gender", "value": ocr_front.get("Gender", "N/A"), "icon": icon_map["Gender"]},
        {"label": "Aadhaar Number", "value": ocr_front.get("Aadhaar Number", "N/A"), "icon": icon_map["Aadhaar Number"]},
    ]

    qr_analysis = results.get("back", {}).get("qr_analysis", {}) or {}
    qr_ok = isinstance(qr_analysis, dict) and "error" not in qr_analysis
    qr_items = []
    if qr_ok:
        parsed = qr_analysis.get("parsed", {}) if isinstance(qr_analysis.get("parsed", {}), dict) else {}
        if parsed:
            for key, value in parsed.items():
                qr_items.append({"label": str(key).capitalize(), "value": value})
        elif "raw" in qr_analysis:
            qr_items.append({"label": "Raw", "value": qr_analysis.get("raw")})
    if not qr_items:
        qr_items.append({"label": "Details", "value": "No QR fields extracted"})

    metadata_analysis = results.get("metadata_analysis", {}) or {}
    metadata_score = int(round(float(metadata_analysis.get("risk_score", 0) or 0)))
    metadata_decision = safe_str(metadata_analysis.get("decision", "Unavailable")) or "Unavailable"
    metadata_confidence = round(float(metadata_analysis.get("confidence", 0) or 0), 2)
    metadata_signals = metadata_analysis.get("signals", []) if isinstance(metadata_analysis.get("signals", []), list) else []

    metadata_fields = []
    for side_key, side_label in (("front", "Front"), ("back", "Back")):
        exif_data = results.get(side_key, {}).get("exif_analysis", {}) or {}
        if isinstance(exif_data, dict) and "error" not in exif_data:
            for key, value in exif_data.items():
                key_text = safe_str(key)
                value_text = safe_str(value)
                key_lower = key_text.lower()
                value_lower = value_text.lower()
                warning = ("software" in key_lower) or any(term in value_lower for term in SUSPICIOUS_SOFTWARE_TERMS)
                metadata_fields.append(
                    {
                        "label": f"{side_label} - {key_text}",
                        "value": value_text,
                        "warning": warning,
                    }
                )

    has_metadata_warning = any(field.get("warning") for field in metadata_fields)
    if metadata_decision == "Likely Tampered":
        metadata_warning = True
        metadata_warning_title = "Metadata Analysis: Likely Tampered"
        metadata_warning_text = "EXIF evidence suggests image edits or inconsistent capture history."
    elif metadata_decision == "Suspicious":
        metadata_warning = True
        metadata_warning_title = "Metadata Analysis: Suspicious"
        metadata_warning_text = "Metadata contains irregularities and should be manually reviewed."
    elif not metadata_fields:
        metadata_warning = True
        metadata_warning_title = "No EXIF Metadata Found"
        metadata_warning_text = "Missing metadata can happen after editing, screenshot capture, or image re-export."
    elif has_metadata_warning:
        metadata_warning = True
        metadata_warning_title = "Potential Metadata Irregularities"
        metadata_warning_text = "Software/edit markers were found in one or both uploaded images."
    else:
        metadata_warning = False
        metadata_warning_title = ""
        metadata_warning_text = ""

    ela_analysis = results.get("ela_analysis", {}) or {}
    noise_analysis = results.get("noise_analysis", {}) or {}
    forgery_analysis = results.get("image_forgery_detection", {}) or {}

    ela_summary = ela_analysis.get("summary", {}) or {}
    noise_summary = noise_analysis.get("summary", {}) or {}

    ela_summary_score = int(round(float(ela_summary.get("risk_score", 0) or 0)))
    noise_summary_score = int(round(float(noise_summary.get("risk_score", 0) or 0)))
    forgery_score = int(round(float(forgery_analysis.get("risk_score", 0) or 0)))

    ela_panels = []
    for side_key, side_label in (("front", "Front"), ("back", "Back")):
        side_data = ela_analysis.get(side_key, {}) or {}
        side_score = int(round(float(side_data.get("risk_score", 0) or 0)))
        ela_panels.append(
            {
                "side": side_label,
                "decision": side_data.get("decision", "Unavailable"),
                "risk_score": side_score,
                "mean_intensity": side_data.get("mean_intensity", "N/A"),
                "std_intensity": side_data.get("std_intensity", "N/A"),
                "image_url": _static_rel_path_to_url(side_data.get("ela_image_path")),
                "risk_badge_class": risk_badge_class(side_score),
            }
        )

    noise_panels = []
    for side_key, side_label in (("front", "Front"), ("back", "Back")):
        side_data = noise_analysis.get(side_key, {}) or {}
        side_score = int(round(float(side_data.get("risk_score", 0) or 0)))
        anomaly_ratio = float(side_data.get("anomaly_ratio", 0) or 0)
        noise_panels.append(
            {
                "side": side_label,
                "decision": side_data.get("decision", "Unavailable"),
                "risk_score": side_score,
                "anomaly_ratio_pct": round(anomaly_ratio * 100, 2),
                "heatmap_url": _static_rel_path_to_url(side_data.get("heatmap_path")),
                "mask_url": _static_rel_path_to_url(side_data.get("mask_path")),
                "risk_badge_class": risk_badge_class(side_score),
            }
        )

    forgery_components_raw = forgery_analysis.get("component_scores", {}) or {}
    forgery_components = [
        {"label": "ELA Signal", "score": round(float(forgery_components_raw.get("ela", 0) or 0), 2)},
        {"label": "Noise Signal", "score": round(float(forgery_components_raw.get("noise", 0) or 0), 2)},
        {"label": "Object Detection", "score": round(float(forgery_components_raw.get("object_detection", 0) or 0), 2)},
        {"label": "Metadata Signal", "score": round(float(forgery_components_raw.get("metadata", 0) or 0), 2)},
    ]
    forgery_signals = forgery_analysis.get("signals", []) or []

    ocr_status = "Success" if ocr_front else "Failed"
    ocr_message = f"{len(ocr_front)} fields extracted" if ocr_front else "Extraction failed"
    qr_status = "Decoded" if qr_ok else "Failed"
    qr_error = qr_analysis.get("error", "Unknown error") if isinstance(qr_analysis, dict) else "Unknown error"
    if qr_ok:
        qr_context = []
        source_side = safe_str(qr_analysis.get("source_side", "")).strip()
        decode_variant = safe_str(qr_analysis.get("decode_variant", "")).strip()
        if source_side:
            qr_context.append(f"{source_side} side")
        if decode_variant:
            qr_context.append(decode_variant.replace("_", " "))
        qr_message = "Data parsed" if not qr_context else f"Data parsed ({', '.join(qr_context)})"
    else:
        qr_message = qr_error
    exif_status = "Found" if metadata_fields else "Not Found"
    exif_message = f"{len(metadata_fields)} fields found" if metadata_fields else "No EXIF data"

    overview_status_cards = [
        {
            "title": "OCR Extraction",
            "status": ocr_status,
            "message": ocr_message,
            "icon": "M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z",
            "badge_class": "bg-green-50 border-green-200 text-green-700" if ocr_status == "Success" else "bg-red-50 border-red-200 text-red-700",
        },
        {
            "title": "QR Validation",
            "status": qr_status,
            "message": qr_message,
            "icon": "M7 7h2v2H7V7zm0 8h2v2H7v-2zm8-8h2v2h-2V7zm0 8h2v2h-2v-2zM4 4h6v6H4V4zm10 0h6v6h-6V4zM4 14h6v6H4v-6zm10 0h6v6h-6v-6z",
            "badge_class": "bg-green-50 border-green-200 text-green-700" if qr_status == "Decoded" else "bg-red-50 border-red-200 text-red-700",
        },
        {
            "title": "Metadata Analysis",
            "status": metadata_decision,
            "message": f"Risk: {metadata_score}% | EXIF fields: {len(metadata_fields)}",
            "icon": "M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7",
            "badge_class": risk_badge_class(metadata_score),
        },
        {
            "title": "ELA Analysis",
            "status": risk_label(ela_summary_score),
            "message": f"Average risk: {ela_summary_score}%",
            "icon": "M4 16l4.586-4.586a2 2 0 012.828 0L16 16m4-4l-2 2m0 0l-2-2m2 2V6",
            "badge_class": risk_badge_class(ela_summary_score),
        },
        {
            "title": "Noise Analysis",
            "status": risk_label(noise_summary_score),
            "message": f"Average risk: {noise_summary_score}%",
            "icon": "M9.75 3v1.5m4.5-1.5v1.5M4.5 9.75h1.5m12 0h1.5M6.22 6.22l1.06 1.06m9.44 9.44l1.06 1.06M6.22 17.78l1.06-1.06m9.44-9.44l1.06-1.06",
            "badge_class": risk_badge_class(noise_summary_score),
        },
        {
            "title": "Forgery Detection",
            "status": safe_str(forgery_analysis.get("decision", "Unavailable")),
            "message": f"Composite risk: {forgery_score}%",
            "icon": "M12 8c-2.21 0-4 1.79-4 4m8 0a4 4 0 00-4-4m0 0V4m0 8v8",
            "badge_class": risk_badge_class(forgery_score),
        },
    ]

    if risk_level == "HIGH":
        action_title = "Manual Verification Required"
        action_text = "High-risk anomalies were detected. Perform manual review before accepting this document."
        action_class = "border-red-200 bg-red-50 text-red-800"
        action_icon_class = "text-red-600"
    elif risk_level == "MODERATE":
        action_title = "Additional Review Recommended"
        action_text = "Some suspicious signals were detected. Validate key fields with trusted sources."
        action_class = "border-amber-200 bg-amber-50 text-amber-800"
        action_icon_class = "text-amber-600"
    else:
        action_title = "No Strong Fraud Signal"
        action_text = "Current checks indicate lower risk, but continue with standard verification controls."
        action_class = "border-green-200 bg-green-50 text-green-800"
        action_icon_class = "text-green-600"

    transformed = {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "confidence_score": confidence_score,
        "risk_color_class": color_map.get(risk_level, "border-slate-500 text-slate-900 bg-slate-50"),
        "fraud_indicators": indicators,
        "ocr_status": ocr_status,
        "ocr_message": ocr_message,
        "qr_status": qr_status,
        "qr_message": qr_message,
        "exif_status": exif_status,
        "exif_message": exif_message,
        "overview_status_cards": overview_status_cards,
        "ocr_front": ocr_front_list,
        "ocr_address": results.get("combined_ocr", {}).get("Address", "N/A"),
        "qr_decode_status": qr_status,
        "qr_data_items": qr_items,
        "qr_mismatch": any("mismatch" in safe_str(d.get("description", "")).lower() and "address" in safe_str(d.get("description", "")).lower() for d in indicators),
        "qr_mismatch_address": "Printed address appears inconsistent with QR-derived address.",
        "metadata_warning": metadata_warning,
        "metadata_warning_title": metadata_warning_title,
        "metadata_warning_text": metadata_warning_text,
        "metadata_decision": metadata_decision,
        "metadata_score": metadata_score,
        "metadata_confidence": metadata_confidence,
        "metadata_signals": metadata_signals,
        "metadata_badge_class": risk_badge_class(metadata_score),
        "metadata_fields_left": metadata_fields[::2],
        "metadata_fields_right": metadata_fields[1::2],
        "ela_summary_decision": ela_summary.get("decision", "Unavailable"),
        "ela_summary_score": ela_summary_score,
        "ela_summary_badge_class": risk_badge_class(ela_summary_score),
        "ela_panels": ela_panels,
        "noise_summary_decision": noise_summary.get("decision", "Unavailable"),
        "noise_summary_score": noise_summary_score,
        "noise_summary_badge_class": risk_badge_class(noise_summary_score),
        "noise_panels": noise_panels,
        "forgery_decision": safe_str(forgery_analysis.get("decision", "Unavailable")),
        "forgery_score": forgery_score,
        "forgery_confidence": round(float(forgery_analysis.get("confidence", 0) or 0), 2),
        "forgery_badge_class": risk_badge_class(forgery_score),
        "forgery_signals": forgery_signals,
        "forgery_components": forgery_components,
        "action_title": action_title,
        "action_text": action_text,
        "action_class": action_class,
        "action_icon_class": action_icon_class,
        "report_year": datetime.utcnow().year,
    }
    return transformed

# -------------------------
# Flask routes
# -------------------------
@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'front_image' not in request.files or 'back_image' not in request.files:
        flash('Please upload both front and back of the Aadhar card')
        return redirect(request.url)

    front_file = request.files['front_image']
    back_file = request.files['back_image']

    if front_file.filename == '' or back_file.filename == '':
        flash('Either one or both images are missing')
        return redirect(request.url)

    if (front_file and allowed_file(front_file.filename)) and (back_file and allowed_file(back_file.filename)):
        tmp_front = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(front_file.filename)[1])
        front_path = tmp_front.name
        tmp_front.close()
        front_file.save(front_path)

        tmp_back = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(back_file.filename)[1])
        back_path = tmp_back.name
        tmp_back.close()
        back_file.save(back_path)

        try:
            analysis_results = analyze_aadhar_pair(front_path, back_path)
            template_data = transform_results_for_template(analysis_results)

            raw = analysis_results.get('raw_results', {})
            annotated_front = create_annotated_image(front_path, raw.get('text_front'), raw.get('object_front'))
            annotated_back = create_annotated_image(back_path, raw.get('text_back'), raw.get('object_back'))

            template_data['front_annotated_image'] = url_for('static', filename=annotated_front) if annotated_front else None
            template_data['back_annotated_image'] = url_for('static', filename=annotated_back) if annotated_back else None

            return render_template('results.html', **template_data)
        finally:
            try:
                os.unlink(front_path)
            except Exception:
                pass
            try:
                os.unlink(back_path)
            except Exception:
                pass
    else:
        flash('Invalid file type. Please upload an image file.')
        return redirect(request.url)

@app.route('/analyzing')
def analyzing():
    return render_template('analyzing.html')

@app.route('/analyze', methods=['POST'])
def api_analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    tmp_file_path = tmp_file.name
    tmp_file.close()
    try:
        file.save(tmp_file_path)
        # For API convenience use same image as both front/back
        analysis_results = analyze_aadhar_pair(tmp_file_path, tmp_file_path)
        return jsonify(analysis_results)
    finally:
        try:
            os.unlink(tmp_file_path)
        except Exception:
            pass

# -------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv("PORT", 7860)))


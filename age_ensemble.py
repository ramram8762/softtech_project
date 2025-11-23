"""Age ensemble backend using UniFace only (option 3-1).

- DeepFace / TensorFlow는 전혀 사용하지 않는다.
- UniFace (RetinaFace + AgeGender) + OpenCV만 사용한다.
- 클라이언트(AgeRunner.js)가 기대하는 JSON 형식을 유지한다.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from uniface import RetinaFace, AgeGender

# Lazy-loaded global models (프로세스당 1번만 로드)
_DETECTOR: RetinaFace | None = None
_AGE_GENDER: AgeGender | None = None


def _get_models() -> Tuple[RetinaFace, AgeGender]:
    """Create or reuse UniFace models."""
    global _DETECTOR, _AGE_GENDER

    if _DETECTOR is None:
        _DETECTOR = RetinaFace()  # default: retinaface_mnet_v2

    if _AGE_GENDER is None:
        _AGE_GENDER = AgeGender()  # age_gender.onnx

    return _DETECTOR, _AGE_GENDER


def _decode_base64_image(image_b64: str) -> np.ndarray:
    """Decode base64 string to OpenCV BGR image."""
    if not image_b64:
        raise ValueError("Empty base64 string")

    # Strip data URI prefix if present
    if "," in image_b64 and image_b64.strip().startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]

    try:
        binary = base64.b64decode(image_b64, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid base64 image") from exc

    nparr = np.frombuffer(binary, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img


def _analyze_uniface(image: np.ndarray) -> Dict[str, Any]:
    """Run UniFace detector + age/gender model on a BGR image."""
    detector, age_gender = _get_models()

    faces = detector.detect(image) or []
    if not faces:
        return {
            "ok": False,
            "error": "NO_FACE",
            "message": "No face detected in the image.",
        }

    # Pick the most confident face (가장 높은 confidence)
    best = max(faces, key=lambda f: float(f.get("confidence", 0.0)))

    bbox = best.get("bbox") or [0, 0, 0, 0]
    confidence = float(best.get("confidence", 0.0))

    # UniFace AgeGender API: gender(str), age(int/float)
    gender_label, age_value = age_gender.predict(image, bbox)

    # Age
    try:
        age_int = int(round(float(age_value)))
    except Exception:  # noqa: BLE001
        age_int = int(age_value or 0)

    # Gender → 확률(score)로 변환 (단순 매핑)
    g_str = str(gender_label or "").lower()
    if g_str.startswith("m"):  # "male"
        man_score = 99.0
        woman_score = 1.0
        gender_label_out = "Male"
    elif g_str.startswith("f"):  # "female"
        man_score = 1.0
        woman_score = 99.0
        gender_label_out = "Female"
    else:
        man_score = 50.0
        woman_score = 50.0
        gender_label_out = "Unknown"

    gender_scores = {
        "Man": float(man_score),
        "Woman": float(woman_score),
    }

    # bbox를 float 리스트로 정규화
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x1, y1, x2, y2 = bbox[:4]
        bbox_list = [float(x1), float(y1), float(x2), float(y2)]
    else:
        bbox_list = [0.0, 0.0, 0.0, 0.0]

    result: Dict[str, Any] = {
        "signature": "SOFTTECH_UNIFACE_ONLY_V3_1",
        "ok": True,
        "error": None,
        "age": age_int,
        "final_age": age_int,
        "gender": gender_scores,  # top-level gender scores
        "ages": {
            "uniface": age_int,
        },
        "models": {
            "uniface": {
                "age": age_int,
                "gender_label": gender_label_out,
                "gender_scores": gender_scores,
                "bbox": bbox_list,
                "confidence": float(confidence),
            },
        },
        "model_count": 1,
        "model_errors": {
            # DeepFace는 완전히 제거되었지만, JSON 구조 호환성을 위해 키만 남겨둔다.
            "deepface": "DISABLED (not installed)",
            "uniface": None,
        },
    }
    return result


def analyze_age_ensemble(image_b64: str) -> Dict[str, Any]:
    """Public API called from app.py.

    Parameters
    ----------
    image_b64 : str
        Base64-encoded image string (with or without data URI prefix).

    Returns
    -------
    Dict[str, Any]
        JSON-serializable result for the mobile app.
    """
    base_result: Dict[str, Any] = {
        "signature": "SOFTTECH_UNIFACE_ONLY_V3_1",
        "ok": False,
        "error": "UNKNOWN",
    }

    try:
        image = _decode_base64_image(image_b64)

        # 간단한 해상도/품질 체크 (너무 작은 경우)
        h, w = image.shape[:2]
        if min(h, w) < 128:
            base_result.update(
                {
                    "ok": False,
                    "error": "IMAGE_TOO_SMALL",
                    "message": f"Image too small for stable analysis: {w}x{h}px",
                }
            )
            return base_result

        result = _analyze_uniface(image)
        base_result.update(result)
        return base_result

    except Exception as exc:  # noqa: BLE001
        base_result.update(
            {
                "ok": False,
                "error": "PIPELINE_ERROR",
                "message": f"{exc.__class__.__name__}: {exc}",
            }
        )
        return base_result

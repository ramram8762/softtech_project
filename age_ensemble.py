"""
Age ensemble module.

- Uses UniFace (RetinaFace + AgeGender ONNX models) as the primary estimator.
- Optionally uses DeepFace (TensorFlow) as a secondary estimator.
- Returns a unified JSON-serializable dictionary that the React Native client expects.

This module is intentionally self-contained so that app.py only needs to call
`analyze_age_ensemble(image_bytes: bytes)`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

# These imports will fail if the corresponding packages are not installed.
# They are wrapped in lazy initialisation so that the server can still run
# even if one of the models is unavailable.
try:
    from uniface import RetinaFace, AgeGender  # type: ignore
except Exception:  # pragma: no cover - handled lazily
    RetinaFace = AgeGender = None  # type: ignore

try:
    from deepface import DeepFace  # type: ignore
except Exception:  # pragma: no cover - handled lazily
    DeepFace = None  # type: ignore


LOGGER = logging.getLogger("age_ensemble")
LOGGER.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _bytes_to_cv2_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes (JPEG/PNG) to an OpenCV BGR image."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩 실패: 지원되지 않는 형식이거나 손상된 파일입니다.")
    return img


def _normalize_gender_keys(dist: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize a raw gender probability dictionary into {'Woman': x, 'Man': y}.

    The input may contain keys like 'Male' / 'Female' / 'man' / 'woman'.
    Missing keys are filled with 0.0.
    """
    woman = 0.0
    man = 0.0

    for k, v in (dist or {}).items():
        key = k.lower()
        try:
            val = float(v)
        except Exception:
            continue

        if key.startswith("f") or key.startswith("w"):  # female / woman
            woman = max(woman, val)
        elif key.startswith("m"):  # male / man
            man = max(man, val)

    # If everything is zero, make it a neutral 50/50.
    if woman <= 0.0 and man <= 0.0:
        woman = man = 50.0

    total = woman + man
    if total <= 0.0:
        return {"Woman": 50.0, "Man": 50.0}

    return {"Woman": 100.0 * woman / total, "Man": 100.0 * man / total}


def _label_to_gender_scores(label: str) -> Dict[str, float]:
    """Fallback when a model only returns a gender label."""
    if not label:
        return {"Woman": 50.0, "Man": 50.0}
    low = label.lower()
    if low.startswith("f") or low.startswith("w"):
        return {"Woman": 99.0, "Man": 1.0}
    return {"Woman": 1.0, "Man": 99.0}


# -----------------------------------------------------------------------------
# UniFace backend
# -----------------------------------------------------------------------------

_UNIFACE_DETECTOR: Optional["RetinaFace"] = None
_UNIFACE_AGE_GENDER: Optional["AgeGender"] = None


def _ensure_uniface_loaded() -> None:
    global _UNIFACE_DETECTOR, _UNIFACE_AGE_GENDER

    if RetinaFace is None or AgeGender is None:
        raise ImportError("uniface 패키지가 설치되지 않았습니다. requirements.txt 에 'uniface' 를 추가하세요.")

    if _UNIFACE_DETECTOR is None or _UNIFACE_AGE_GENDER is None:
        LOGGER.info("[UniFace] Initializing RetinaFace + AgeGender models...")
        _UNIFACE_DETECTOR = RetinaFace()
        _UNIFACE_AGE_GENDER = AgeGender()


def _run_uniface(image_bgr: np.ndarray) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Run UniFace RetinaFace + AgeGender on the given BGR image.

    Returns:
        (model_result_dict, error_message)
        If error_message is not None, model_result_dict is empty.
    """
    try:
        _ensure_uniface_loaded()
        assert _UNIFACE_DETECTOR is not None
        assert _UNIFACE_AGE_GENDER is not None

        faces = _UNIFACE_DETECTOR.detect(image_bgr)
        if not faces:
            raise RuntimeError("얼굴을 찾지 못했습니다.")

        # Choose the most confident face.
        best = max(
            faces,
            key=lambda f: float(f.get("score") or f.get("confidence") or 0.0),
        )

        bbox = best.get("bbox") or best.get("box") or best.get("bbox_xyxy")
        if bbox is None:
            raise RuntimeError("UniFace 결과에 bbox가 없습니다.")

        gender_label, age_value = _UNIFACE_AGE_GENDER.predict(image_bgr, bbox)

        gender_scores = _label_to_gender_scores(str(gender_label))
        model_res: Dict[str, Any] = {
            "age": float(age_value),
            "gender_label": str(gender_label),
            "gender_scores": gender_scores,
            "bbox": [float(x) for x in bbox],
            "confidence": float(best.get("score") or best.get("confidence") or 1.0),
        }
        return model_res, None
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("[UniFace] Error during inference: %s", exc)
        return {}, f"UniFace error: {exc}"


# -----------------------------------------------------------------------------
# DeepFace backend
# -----------------------------------------------------------------------------

_DEEPFACE_READY: bool = False
_DEEPFACE_IMPORT_ERROR: Optional[str] = None


def _ensure_deepface_loaded() -> None:
    global _DEEPFACE_READY, _DEEPFACE_IMPORT_ERROR

    if _DEEPFACE_READY:
        return
    if DeepFace is None:
        _DEEPFACE_IMPORT_ERROR = "DeepFace import failed (패키지 설치 확인 필요)"
        raise ImportError(_DEEPFACE_IMPORT_ERROR)

    # Try a trivial call to make sure TensorFlow / tf-keras are consistent.
    try:
        # We don't actually run a model here; we just touch the module.
        _ = DeepFace  # noqa: F841
        _DEEPFACE_READY = True
    except Exception as exc:  # noqa: BLE001
        _DEEPFACE_IMPORT_ERROR = f"DeepFace import error: {exc}"
        raise


def _run_deepface(image_bgr: np.ndarray) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Run DeepFace age+gender analysis on the given BGR image.

    Returns:
        (model_result_dict, error_message)
        If error_message is not None, model_result_dict is empty.
    """
    try:
        _ensure_deepface_loaded()
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("DeepFace import error: %s", exc)
        return {}, f"DeepFace import error: {exc}"

    if DeepFace is None:
        return {}, "DeepFace module not available."

    try:
        # Downscale large images to reduce memory usage.
        h, w = image_bgr.shape[:2]
        max_side = max(h, w)
        if max_side > 800:
            scale = 800.0 / float(max_side)
            image_bgr = cv2.resize(
                image_bgr,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        # DeepFace expects RGB by default.
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        analysis = DeepFace.analyze(
            img_path=image_rgb,
            actions=["age", "gender"],
            enforce_detection=False,
            prog_bar=False,
        )

        if isinstance(analysis, list) and analysis:
            analysis = analysis[0]

        age_val = float(analysis.get("age", 0.0))
        gender_dist = analysis.get("gender") or analysis.get("gender_prob") or {}
        gender_final = _normalize_gender_keys(gender_dist)
        gender_label = analysis.get("dominant_gender") or (
            "Woman" if gender_final["Woman"] >= gender_final["Man"] else "Man"
        )

        model_res: Dict[str, Any] = {
            "age": age_val,
            "gender_label": str(gender_label),
            "gender_scores": gender_final,
            "bbox": None,
            "confidence": float(analysis.get("face_confidence") or 1.0),
        }
        return model_res, None
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("[DeepFace] Error during inference: %s", exc)
        return {}, f"DeepFace error: {exc}"


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

SIGNATURE = "SOFTTECH_DEEPFACE_UNIFACE_V1"


def analyze_age_ensemble(image_bytes: bytes) -> Dict[str, Any]:
    """
    Main entry point.

    Args:
        image_bytes: Raw JPEG/PNG bytes from the client.

    Returns:
        JSON-serializable dictionary with keys:
          - signature
          - ok
          - error
          - age
          - final_age
          - gender {Woman, Man}
          - ages {model_name: age}
          - models {model_name: {detail...}}
          - model_count
          - model_errors {model_name: error_message_or_None}
    """
    image = _bytes_to_cv2_image(image_bytes)

    models: Dict[str, Dict[str, Any]] = {}
    model_errors: Dict[str, Optional[str]] = {}
    ages: Dict[str, float] = {}

    gender_acc = {"Woman": 0.0, "Man": 0.0}
    successful_models = 0

    # UniFace
    uniface_res, uniface_err = _run_uniface(image)
    model_errors["uniface"] = uniface_err
    if not uniface_err and uniface_res:
        models["uniface"] = uniface_res
        ages["uniface"] = float(uniface_res["age"])
        gender = uniface_res["gender_scores"]
        gender_acc["Woman"] += float(gender.get("Woman", 0.0))
        gender_acc["Man"] += float(gender.get("Man", 0.0))
        successful_models += 1

    # DeepFace
    deepface_res, deepface_err = _run_deepface(image)
    model_errors["deepface"] = deepface_err
    if not deepface_err and deepface_res:
        models["deepface"] = deepface_res
        ages["deepface"] = float(deepface_res["age"])
        gender = deepface_res["gender_scores"]
        gender_acc["Woman"] += float(gender.get("Woman", 0.0))
        gender_acc["Man"] += float(gender.get("Man", 0.0))
        successful_models += 1

    if successful_models == 0:
        # No model succeeded: return a graceful error payload.
        return {
            "signature": SIGNATURE,
            "ok": False,
            "error": "No model could estimate age / gender (모델 추론 실패)",
            "age": None,
            "final_age": None,
            "gender": {"Woman": 50.0, "Man": 50.0},
            "ages": {},
            "models": {},
            "model_count": 0,
            "model_errors": model_errors,
        }

    # Aggregate final age and gender.
    final_age = float(sum(ages.values()) / float(len(ages)))
    gender = {
        "Woman": float(gender_acc["Woman"] / successful_models),
        "Man": float(gender_acc["Man"] / successful_models),
    }

    return {
        "signature": SIGNATURE,
        "ok": True,
        "error": None,
        "age": final_age,
        "final_age": final_age,
        "gender": gender,
        "ages": ages,
        "models": models,
        "model_count": successful_models,
        "model_errors": model_errors,
    }

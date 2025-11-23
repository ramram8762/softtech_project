from __future__ import annotations
"""
DeepFace + UniFace 앙상블 모듈 (DeepFace import 에러 노출 버전)
"""

from typing import Any, Dict, Optional, Tuple

import base64
import cv2
import numpy as np

from uniface import RetinaFace, AgeGender

_DETECTOR: Optional[RetinaFace] = None
_AGE_GENDER: Optional[AgeGender] = None

# DeepFace import 는 lazy 하게, 에러 메시지를 전역에 보관
_DEEPFACE_OBJ = None
_DEEPFACE_ERR: Optional[str] = None


def _get_uniface_models() -> Tuple[RetinaFace, AgeGender]:
    global _DETECTOR, _AGE_GENDER

    if _DETECTOR is None:
        _DETECTOR = RetinaFace()

    if _AGE_GENDER is None:
        _AGE_GENDER = AgeGender()

    return _DETECTOR, _AGE_GENDER


def _decode_base64_to_bgr(image_base64: str) -> Optional[np.ndarray]:
    try:
        if not image_base64:
            return None
        if image_base64.startswith("data:"):
            image_base64 = image_base64.split(",", 1)[-1]
        binary = base64.b64decode(image_base64)
        arr = np.frombuffer(binary, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return img
    except Exception:
        return None


def _softmax_2(logits: np.ndarray) -> Tuple[float, float]:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    if logits.size < 2:
        return 0.5, 0.5
    m = float(np.max(logits[:2]))
    e0 = float(np.exp(logits[0] - m))
    e1 = float(np.exp(logits[1] - m))
    s = e0 + e1
    if s <= 0.0:
        return 0.5, 0.5
    return e0 / s, e1 / s


def _predict_uniface_from_bgr(bgr: np.ndarray) -> Dict[str, Any]:
    det, ag = _get_uniface_models()

    faces = det.detect(bgr, max_num=1)
    if not faces:
        return {
            "ok": False,
            "error": "UniFace 에서 얼굴을 찾지 못했습니다.",
            "age": None,
            "gender_label": None,
            "gender_scores": None,
            "bbox": None,
            "confidence": None,
        }

    face0 = faces[0]
    bbox = None
    conf = None

    if isinstance(face0, dict):
        bbox = face0.get("bbox") or face0.get("box") or face0.get("bbox_xyxy")
        conf = face0.get("confidence")
    elif isinstance(face0, (list, tuple, np.ndarray)):
        bbox = face0

    if bbox is None:
        return {
            "ok": False,
            "error": "UniFace bbox 포맷 해석 실패",
            "age": None,
            "gender_label": None,
            "gender_scores": None,
            "bbox": None,
            "confidence": None,
        }

    bbox_array = np.asarray(bbox, dtype=np.float32).reshape(-1)

    try:
        blob = ag.preprocess(bgr, bbox_array)
        raw_out = ag.session.run(ag.output_names, {ag.input_name: blob})[0][0]
        raw_out = np.asarray(raw_out, dtype=np.float32).reshape(-1)
    except Exception as e:
        return {
            "ok": False,
            "error": f"UniFace AgeGender 추론 실패: {e}",
            "age": None,
            "gender_label": None,
            "gender_scores": None,
            "bbox": bbox_array.tolist(),
            "confidence": float(conf) if conf is not None else None,
        }

    gender_label, age_years = ag.postprocess(raw_out)
    p_female, p_male = _softmax_2(raw_out[:2])
    gender_scores = {
        "Woman": float(p_female * 100.0),
        "Man": float(p_male * 100.0),
    }

    return {
        "ok": True,
        "error": None,
        "age": int(age_years),
        "gender_label": str(gender_label),
        "gender_scores": gender_scores,
        "bbox": bbox_array.tolist(),
        "confidence": float(conf) if conf is not None else None,
    }


def _get_deepface():
    global _DEEPFACE_OBJ, _DEEPFACE_ERR
    if _DEEPFACE_OBJ is not None:
        return _DEEPFACE_OBJ, _DEEPFACE_ERR

    try:
        from deepface import DeepFace  # type: ignore
        _DEEPFACE_OBJ = DeepFace
        _DEEPFACE_ERR = None
    except Exception as e:
        _DEEPFACE_OBJ = None
        _DEEPFACE_ERR = f"DeepFace import error: {e!r}"
    return _DEEPFACE_OBJ, _DEEPFACE_ERR


def _predict_deepface_from_bgr(bgr: np.ndarray) -> Dict[str, Any]:
    DeepFace, imp_err = _get_deepface()
    if DeepFace is None:
        return {
            "ok": False,
            "error": imp_err or "DeepFace 라이브러리가 서버에 설치되어 있지 않습니다.",
            "age": None,
            "gender_label": None,
            "gender_scores": None,
            "bbox": None,
            "confidence": None,
        }

    try:
        obj = DeepFace.analyze(
            img_path=bgr,
            actions=["age", "gender"],
            enforce_detection=False,
        )
    except Exception as e:
        return {
            "ok": False,
            "error": f"DeepFace 분석 실패: {e}",
            "age": None,
            "gender_label": None,
            "gender_scores": None,
            "bbox": None,
            "confidence": None,
        }

    if isinstance(obj, list):
        if not obj:
            return {
                "ok": False,
                "error": "DeepFace 결과가 비어 있습니다.",
                "age": None,
                "gender_label": None,
                "gender_scores": None,
                "bbox": None,
                "confidence": None,
            }
        result = obj[0]
    else:
        result = obj

    age = result.get("age")
    try:
        if age is not None:
            age = int(age)
    except Exception:
        age = None

    gender_label = result.get("gender") or result.get("dominant_gender")
    if gender_label is not None:
        gender_label = str(gender_label)

    gender_scores = None
    raw_gender_scores = result.get("gender_scores")
    if isinstance(raw_gender_scores, dict):
        gender_scores = {}
        for k, v in raw_gender_scores.items():
            try:
                gender_scores[str(k)] = float(v) * 100.0
            except Exception:
                continue

    bbox = result.get("region") or result.get("bbox")
    if isinstance(bbox, (list, tuple, np.ndarray)):
        bbox = np.asarray(bbox, dtype=np.float32).reshape(-1).tolist()
    else:
        bbox = None

    return {
        "ok": age is not None,
        "error": None if age is not None else "DeepFace 에서 나이를 얻지 못했습니다.",
        "age": age,
        "gender_label": gender_label,
        "gender_scores": gender_scores,
        "bbox": bbox,
        "confidence": None,
    }


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _merge_gender(uni: Dict[str, Any], df: Dict[str, Any]) -> Optional[Dict[str, float]]:
    scores_list = []
    for src in (uni.get("gender_scores"), df.get("gender_scores")):
        if isinstance(src, dict) and src:
            scores_list.append({str(k): float(v) for k, v in src.items()})

    if scores_list:
        acc: Dict[str, float] = {}
        cnt: Dict[str, int] = {}
        for s in scores_list:
            for k, v in s.items():
                acc[k] = acc.get(k, 0.0) + v
                cnt[k] = cnt.get(k, 0) + 1
        return {k: acc[k] / max(1, cnt[k]) for k in acc.keys()}

    label = uni.get("gender_label") or df.get("gender_label")
    if not label:
        return None

    label_str = str(label).lower()
    if label_str.startswith("m"):
        return {"Man": 100.0, "Woman": 0.0}
    return {"Woman": 100.0, "Man": 0.0}


def analyze_age_ensemble(image_base64: str) -> Dict[str, Any]:
    bgr = _decode_base64_to_bgr(image_base64)
    if bgr is None:
        return {
            "signature": "SOFTTECH_DEEPFACE_UNIFACE_V1",
            "ok": False,
            "error": "이미지 디코딩 실패",
            "age": None,
            "final_age": None,
            "gender": None,
            "ages": {},
            "models": {},
            "model_count": 0,
            "model_errors": {"uniface": "decode_failed", "deepface": "decode_failed"},
        }

    uni = _predict_uniface_from_bgr(bgr)
    df = _predict_deepface_from_bgr(bgr)

    ages: Dict[str, float] = {}
    models: Dict[str, Dict[str, Any]] = {}
    model_errors: Dict[str, Optional[str]] = {}

    if uni.get("ok") and uni.get("age") is not None:
        ages["uniface"] = float(uni["age"])
        models["uniface"] = {
            "age": int(uni["age"]),
            "gender_label": uni.get("gender_label"),
            "gender_scores": uni.get("gender_scores"),
            "bbox": uni.get("bbox"),
            "confidence": uni.get("confidence"),
        }
        model_errors["uniface"] = None
    else:
        model_errors["uniface"] = uni.get("error") or "unknown"

    if df.get("ok") and df.get("age") is not None:
        ages["deepface"] = float(df["age"])
        models["deepface"] = {
            "age": int(df["age"]),
            "gender_label": df.get("gender_label"),
            "gender_scores": df.get("gender_scores"),
            "bbox": df.get("bbox"),
            "confidence": df.get("confidence"),
        }
        model_errors["deepface"] = None
    else:
        model_errors["deepface"] = df.get("error") or "unknown"

    final_age: Optional[float] = None
    if ages:
        vals = list(ages.values())
        final_age = sum(vals) / max(1, len(vals))

    gender_payload = _merge_gender(uni, df)

    global_ok = _safe_float(final_age) is not None

    if model_errors.get("deepface"):
        print(">> DeepFace error:", model_errors["deepface"], flush=True)

    return {
        "signature": "SOFTTECH_DEEPFACE_UNIFACE_V1",
        "ok": global_ok,
        "error": None if global_ok else "모든 모델에서 나이 추론 실패",
        "age": _safe_float(final_age),
        "final_age": _safe_float(final_age),
        "gender": gender_payload,
        "ages": {k: float(v) for k, v in ages.items()},
        "models": models,
        "model_count": len(ages),
        "model_errors": model_errors,
    }

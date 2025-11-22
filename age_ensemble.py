from __future__ import annotations
"""
통합 나이/성별 추정 모듈 (UniFace 단독 버전, 서명 포함).

- UniFace: RetinaFace + AgeGender 조합 (나이 + 성별, ONNX 기반)
- DeepFace 는 아예 사용하지 않는다.
- 응답 JSON 에는 signature 필드를 넣어서
  서버 코드가 확실히 바뀌었는지 확인 가능하게 한다.
"""

from typing import Any, Dict, Optional

import base64
import cv2
import numpy as np

from uniface import RetinaFace, AgeGender


_DETECTOR: Optional[RetinaFace] = None
_AGE_GENDER: Optional[AgeGender] = None


def _get_models() -> tuple[RetinaFace, AgeGender]:
    """RetinaFace / AgeGender 모델을 lazy-init 방식으로 준비한다."""
    global _DETECTOR, _AGE_GENDER

    if _DETECTOR is None:
        _DETECTOR = RetinaFace()

    if _AGE_GENDER is None:
        _AGE_GENDER = AgeGender()

    return _DETECTOR, _AGE_GENDER


def _decode_base64_to_bgr(image_base64: str) -> Optional[np.ndarray]:
    """
    data URL prefix 를 포함한 base64 문자열을 BGR OpenCV 이미지로 변환한다.
    실패 시 None 을 리턴한다.
    """
    try:
        if not image_base64:
            return None

        # data URL prefix 제거
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


def _softmax_2(logits: np.ndarray) -> tuple[float, float]:
    """길이 2 로짓에 대한 softmax → (p0, p1)."""
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
    """
    UniFace (RetinaFace + AgeGender) 로부터
    단일 얼굴에 대한 나이/성별을 추정한다.
    """
    det, ag = _get_models()

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
            "error": "UniFace 에서 얼굴 bbox 를 해석하지 못했습니다.",
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


def _safe_float(x: Any) -> Optional[float]:
    """넘어온 값을 안전하게 float 로 변환한다. 실패 시 None."""
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def analyze_age_ensemble(image_base64: str) -> Dict[str, Any]:
    """
    클라이언트에서 보낸 base64 이미지를 받아
    UniFace 로 나이/성별을 추정한 뒤 JSON 결과로 정리한다.
    """
    bgr = _decode_base64_to_bgr(image_base64)
    if bgr is None:
        return {
            "signature": "SOFTTECH_UNIFACE_ONLY_V3",
            "ok": False,
            "error": "이미지 디코딩 실패",
            "age": None,
            "final_age": None,
            "gender": None,
            "ages": {},
            "models": {},
            "model_count": 0,
        }

    uni = _predict_uniface_from_bgr(bgr)

    ages: Dict[str, float] = {}
    models: Dict[str, Dict[str, Any]] = {}

    if uni.get("ok") and uni.get("age") is not None:
        ages["uniface"] = float(uni["age"])
        models["uniface"] = {
            "age": int(uni["age"]),
            "gender_label": uni.get("gender_label"),
            "gender_scores": uni.get("gender_scores"),
            "bbox": uni.get("bbox"),
            "confidence": uni.get("confidence"),
        }

    final_age: Optional[float] = None
    if ages:
        vals = list(ages.values())
        final_age = sum(vals) / max(1, len(vals))

    gender_payload: Optional[Dict[str, float]] = None
    if uni.get("gender_scores"):
        raw_map = uni["gender_scores"]
        gender_payload = {str(k): float(v) for k, v in raw_map.items()}
    elif uni.get("gender_label"):
        g = str(uni["gender_label"])
        if g.lower().startswith("m"):
            gender_payload = {"Man": 100.0, "Woman": 0.0}
        else:
            gender_payload = {"Woman": 100.0, "Man": 0.0}

    return {
        "signature": "SOFTTECH_UNIFACE_ONLY_V3",
        "ok": _safe_float(final_age) is not None,
        "age": _safe_float(final_age),
        "final_age": _safe_float(final_age),
        "gender": gender_payload,
        "ages": {k: float(v) for k, v in ages.items()},
        "models": models,
        "model_count": len(ages),
    }

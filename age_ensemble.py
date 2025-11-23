from __future__ import annotations
"""
통합 나이/성별 추정 모듈 (DeepFace + UniFace 앙상블 버전, 서명 포함).

- UniFace: RetinaFace + AgeGender 조합 (나이 + 성별, ONNX 기반)
- DeepFace: 공개 라이브러리 deepface 의 age / gender 분석 사용
- 가능한 경우 두 모델의 나이를 평균내서 최종 나이를 만든다.
- 응답 JSON 에는 signature 필드를 넣어서
  서버 코드가 확실히 바뀌었는지 확인 가능하게 한다.
"""

from typing import Any, Dict, Optional, Tuple

import base64
import cv2
import numpy as np

from uniface import RetinaFace, AgeGender

# DeepFace 는 설치 여부가 환경마다 다를 수 있으므로
# 임포트 실패 시에도 UniFace 단독으로 동작하도록 처리한다.
try:  # pragma: no cover - 런타임 환경 보호용
    from deepface import DeepFace  # type: ignore
except Exception:  # pragma: no cover
    DeepFace = None  # type: ignore


_DETECTOR: Optional[RetinaFace] = None
_AGE_GENDER: Optional[AgeGender] = None


def _get_uniface_models() -> Tuple[RetinaFace, AgeGender]:
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


def _softmax_2(logits: np.ndarray) -> Tuple[float, float]:
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


def _predict_deepface_from_bgr(bgr: np.ndarray) -> Dict[str, Any]:
    """
    DeepFace 로부터 단일 얼굴에 대한 나이/성별을 추정한다.
    실패하더라도 전체 앙상블은 UniFace 결과만으로 계속 동작할 수 있게 한다.
    """
    if DeepFace is None:
        return {
            "ok": False,
            "error": "DeepFace 라이브러리가 서버에 설치되어 있지 않습니다.",
            "age": None,
            "gender_label": None,
            "gender_scores": None,
            "bbox": None,
            "confidence": None,
        }

    try:
        # deepface 는 BGR ndarray 를 img_path 인자로 받을 수 있다.
        # enforce_detection=False 로 설정하여 검출 실패시에도 예외를 최소화한다.
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

    # DeepFace 가 gender 확률 맵을 제공하지 않는 버전도 있기 때문에
    # 딕셔너리가 아니면 None 으로 처리한다.
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
    """넘어온 값을 안전하게 float 로 변환한다. 실패 시 None."""
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _merge_gender(uni: Dict[str, Any], df: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    UniFace / DeepFace 의 gender 정보를 합쳐서
    최종 gender 확률 맵을 만든다.
    둘 중 하나만 있으면 그대로 사용하고,
    둘 다 있으면 평균을 낸다.
    """
    scores_list = []
    for src in (uni.get("gender_scores"), df.get("gender_scores")):
        if isinstance(src, dict) and src:
            scores_list.append({str(k): float(v) for k, v in src.items()})

    if scores_list:
        # keys: "Man", "Woman" 중심
        acc: Dict[str, float] = {}
        cnt: Dict[str, int] = {}
        for s in scores_list:
            for k, v in s.items():
                acc[k] = acc.get(k, 0.0) + v
                cnt[k] = cnt.get(k, 0) + 1
        merged = {k: acc[k] / max(1, cnt[k]) for k in acc.keys()}
        return merged

    # 확률 맵이 없으면 label 기반으로만 구성
    label = uni.get("gender_label") or df.get("gender_label")
    if not label:
        return None

    label_str = str(label).lower()
    if label_str.startswith("m"):
        return {"Man": 100.0, "Woman": 0.0}
    return {"Woman": 100.0, "Man": 0.0}


def analyze_age_ensemble(image_base64: str) -> Dict[str, Any]:
    """
    클라이언트에서 보낸 base64 이미지를 받아
    UniFace + DeepFace 로 나이/성별을 추정한 뒤 JSON 결과로 정리한다.
    """
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
        }

    uni = _predict_uniface_from_bgr(bgr)
    df = _predict_deepface_from_bgr(bgr)

    ages: Dict[str, float] = {}
    models: Dict[str, Dict[str, Any]] = {}
    errors = []

    if uni.get("ok") and uni.get("age") is not None:
        ages["uniface"] = float(uni["age"])
        models["uniface"] = {
            "age": int(uni["age"]),
            "gender_label": uni.get("gender_label"),
            "gender_scores": uni.get("gender_scores"),
            "bbox": uni.get("bbox"),
            "confidence": uni.get("confidence"),
        }
    else:
        err = uni.get("error")
        if err:
            errors.append(f"UniFace: {err}")

    if df.get("ok") and df.get("age") is not None:
        ages["deepface"] = float(df["age"])
        models["deepface"] = {
            "age": int(df["age"]),
            "gender_label": df.get("gender_label"),
            "gender_scores": df.get("gender_scores"),
            "bbox": df.get("bbox"),
            "confidence": df.get("confidence"),
        }
    else:
        err = df.get("error")
        if err:
            errors.append(f"DeepFace: {err}")

    final_age: Optional[float] = None
    if ages:
        vals = list(ages.values())
        final_age = sum(vals) / max(1, len(vals))

    gender_payload = _merge_gender(uni, df)

    global_ok = _safe_float(final_age) is not None

    return {
        "signature": "SOFTTECH_DEEPFACE_UNIFACE_V1",
        "ok": global_ok,
        "error": None if global_ok else " / ".join(errors) if errors else "모든 모델에서 나이 추론 실패",
        "age": _safe_float(final_age),
        "final_age": _safe_float(final_age),
        "gender": gender_payload,
        "ages": {k: float(v) for k, v in ages.items()},
        "models": models,
        "model_count": len(ages),
    }

from __future__ import annotations

"""
통합 나이 추정 모듈 (DeepFace + UniFace).

- DeepFace: 기준 모델 (나이 + 성별)
- UniFace: RetinaFace + AgeGender 조합 (나이 + 성별, ONNX 기반)

app.py 에서 predict_age_deepface / analyze_age_ensemble 를 import 해서 사용한다.
JSON 으로 돌려주는 값은 모두 Python 기본 타입(int, float, str)만 사용해서
Flask 의 jsonify 에서 float32 에러가 다시 나오지 않도록 했다.
"""

from typing import Any, Dict, Optional

import base64

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None  # type: ignore

try:
    from deepface import DeepFace  # type: ignore
except Exception:
    DeepFace = None  # type: ignore

try:
    from uniface import RetinaFace, AgeGender  # type: ignore
except Exception:
    RetinaFace = None  # type: ignore
    AgeGender = None  # type: ignore

# UniFace 전역 객체 (지연 초기화)
_UNIFACE_DETECTOR = None
_UNIFACE_AGE_GENDER = None



# ------------------------------------------------
# 공통 유틸
# ------------------------------------------------

def _safe_float(v: Any) -> Optional[float]:
    """numpy.float32 같은 것도 전부 Python float 로 변환."""
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _softmax(x: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
    x = x.astype("float32")
    x = x - x.max()
    exp_x = np.exp(x)
    s = exp_x.sum()
    if s <= 0:
        return np.ones_like(exp_x) / len(exp_x)
    return exp_x / s


def _decode_base64_to_bgr(image_base64: str) -> Optional["np.ndarray"]:  # type: ignore[name-defined]
    """
    data:image/jpeg;base64,.... 형태 또는 순수 base64 문자열을 BGR 이미지로 변환.
    """
    if np is None or cv2 is None:
        return None

    if not image_base64:
        return None

    # dataURL prefix 제거
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(image_base64)
    except Exception:
        return None

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img



# ------------------------------------------------
# UniFace 기반 나이 / 성별
# ------------------------------------------------

def _get_uniface_models():
    """UniFace RetinaFace + AgeGender 모델을 지연 초기화하여 반환."""
    global _UNIFACE_DETECTOR, _UNIFACE_AGE_GENDER
    if RetinaFace is None or AgeGender is None or cv2 is None or np is None:
        return None, None
    if _UNIFACE_DETECTOR is None:
        try:
            _UNIFACE_DETECTOR = RetinaFace()
        except Exception:
            return None, None
    if _UNIFACE_AGE_GENDER is None:
        try:
            _UNIFACE_AGE_GENDER = AgeGender()
        except Exception:
            return None, None
    return _UNIFACE_DETECTOR, _UNIFACE_AGE_GENDER


def _predict_uniface_from_bgr(bgr: "np.ndarray") -> Dict[str, Any]:  # type: ignore[name-defined]
    """UniFace AgeGender 로부터 나이(+성별)를 얻는다."""
    det, ag = _get_uniface_models()
    if det is None or ag is None:
        return {
            "ok": False,
            "error": "UniFace 가 설치되어 있지 않거나 초기화에 실패했습니다.",
            "age": None,
            "gender_label": None,
            "gender_scores": None,
        }
    try:
        faces = det.detect(bgr)
        if not faces:
            return {
                "ok": False,
                "error": "UniFace 에서 얼굴을 찾지 못했습니다.",
                "age": None,
                "gender_label": None,
                "gender_scores": None,
            }
        face0 = faces[0]
        bbox = None
        if isinstance(face0, dict):
            bbox = face0.get("bbox") or face0.get("box") or face0.get("bbox_xyxy")
        elif isinstance(face0, (list, tuple)):
            bbox = face0
        if bbox is None:
            return {
                "ok": False,
                "error": "UniFace bbox 정보가 없습니다.",
                "age": None,
                "gender_label": None,
                "gender_scores": None,
            }

        gender, age = ag.predict(bgr, bbox)  # type: ignore[arg-type]
        age_f = _safe_float(age)
        gender_label = None
        gender_scores = None
        if isinstance(gender, str):
            g = gender.lower()
            if "male" in g or g == "m":
                gender_label = "male"
                gender_scores = {"Man": 100.0, "Woman": 0.0}
            elif "female" in g or g == "f":
                gender_label = "female"
                gender_scores = {"Man": 0.0, "Woman": 100.0}
            else:
                gender_label = gender
        return {
            "ok": age_f is not None,
            "error": None if age_f is not None else "UniFace 나이값 없음",
            "age": age_f,
            "gender_label": gender_label,
            "gender_scores": gender_scores,
        }
    except Exception as e:  # noqa: F841
        return {
            "ok": False,
            "error": f"UniFace 예측 중 오류: {e}",
            "age": None,
            "gender_label": None,
            "gender_scores": None,
        }

# ------------------------------------------------
# DeepFace 기반 나이 / 성별
# ------------------------------------------------

def _predict_deepface_from_bgr(bgr: "np.ndarray") -> Dict[str, Any]:  # type: ignore[name-defined]
    if DeepFace is None:
        return {
            "ok": False,
            "error": "DeepFace 가 설치되어 있지 않습니다.",
            "age": None,
            "gender_label": None,
            "gender_scores": None,
        }

    try:
        # DeepFace 는 RGB 를 선호하므로 변환
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = DeepFace.analyze(
            img_path=rgb,
            actions=["age", "gender"],
            enforce_detection=False,
        )
    except Exception as e:
        return {
            "ok": False,
            "error": f"DeepFace.analyze 실패: {e}",
            "age": None,
            "gender_label": None,
            "gender_scores": None,
        }

    # DeepFace 가 list 를 반환하는 경우 / dict 를 반환하는 경우 모두 처리
    if isinstance(result, list) and result:
        r0 = result[0]
    else:
        r0 = result

    age = _safe_float(r0.get("age")) if isinstance(r0, dict) else None
    gender_label = None
    gender_scores: Optional[Dict[str, float]] = None

    if isinstance(r0, dict):
        # 예: {'Woman': np.float32(4.2), 'Man': np.float32(95.7)}
        g_scores = r0.get("gender")
        if isinstance(g_scores, dict):
            gender_scores = {
                str(k): float(v) for k, v in g_scores.items() if _safe_float(v) is not None
            }
        dom = r0.get("dominant_gender")
        if isinstance(dom, str):
            gender_label = dom

    return {
        "ok": True,
        "age": age,
        "gender_label": gender_label,
        "gender_scores": gender_scores,
    }


def predict_age_deepface(image_base64: str) -> Dict[str, Any]:
    """
    /age 엔드포인트에서 사용하는 헬퍼.

    반환 형식 (예시)
    {
      "age": 31.0,
      "gender": {"Man": 95.7, "Woman": 4.2}
    }
    """
    if np is None or cv2 is None:
        return {
            "ok": False,
            "error": "numpy / opencv 가 없습니다.",
            "age": None,
            "gender": None,
        }

    bgr = _decode_base64_to_bgr(image_base64)
    if bgr is None:
        return {
            "ok": False,
            "error": "이미지 디코딩 실패",
            "age": None,
            "gender": None,
        }

    res = _predict_deepface_from_bgr(bgr)
    if not res.get("ok"):
        return {
            "ok": False,
            "error": res.get("error", "DeepFace 예측 실패"),
            "age": None,
            "gender": None,
        }

    gender_payload = None
    if res.get("gender_scores"):
        gender_payload = {
            str(k): float(v) for k, v in res["gender_scores"].items()
        }
    elif res.get("gender_label"):
        # label 만 있으면 100% 로 표시
        gender_payload = {str(res["gender_label"]): 100.0}

    return {
        "ok": True,
        "age": _safe_float(res.get("age")),
        "gender": gender_payload,
    }


# ------------------------------------------------
# ONNX 모델 세션 준비
# ------------------------------------------------

_GENDERAGE_SESSION: Optional["ort.InferenceSession"] = None  # type: ignore[name-defined]
_AGE_GOOGLENET_SESSION: Optional["ort.InferenceSession"] = None  # type: ignore[name-defined]


def _get_genderage_session() -> Optional["ort.InferenceSession"]:  # type: ignore[name-defined]
    global _GENDERAGE_SESSION
    if ort is None:
        return None
    if _GENDERAGE_SESSION is not None:
        return _GENDERAGE_SESSION
    from pathlib import Path

    model_path = Path(__file__).resolve().parent / "models" / "facial_analysis" / "genderage.onnx"
    if not model_path.exists():
        return None
    try:
        _GENDERAGE_SESSION = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        return _GENDERAGE_SESSION
    except Exception:
        return None


def _get_age_googlenet_session() -> Optional["ort.InferenceSession"]:  # type: ignore[name-defined]
    global _AGE_GOOGLENET_SESSION
    if ort is None:
        return None
    if _AGE_GOOGLENET_SESSION is not None:
        return _AGE_GOOGLENET_SESSION
    from pathlib import Path

    model_path = Path(__file__).resolve().parent / "models" / "age_gender_zoo" / "age_googlenet.onnx"
    if not model_path.exists():
        return None
    try:
        _AGE_GOOGLENET_SESSION = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        return _AGE_GOOGLENET_SESSION
    except Exception:
        return None


# ------------------------------------------------
# genderage.onnx : 나이 + 성별
# ------------------------------------------------

def _predict_genderage_from_bgr(bgr: "np.ndarray") -> Dict[str, Any]:  # type: ignore[name-defined]
    if np is None or cv2 is None or ort is None:
        return {
            "ok": False,
            "error": "numpy / cv2 / onnxruntime 중 하나가 없습니다.",
            "age": None,
            "gender_label": None,
            "gender_scores": None,
        }

    session = _get_genderage_session()
    if session is None:
        return {
            "ok": False,
            "error": "genderage.onnx 세션 초기화 실패",
            "age": None,
            "gender_label": None,
            "gender_scores": None,
        }

    try:
        img = cv2.resize(bgr, (112, 112))
        img = img.astype("float32")

        # BGR -> RGB (모델에 따라 큰 차이는 없지만 일반적인 처리)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # CHW 로 변환
        img = np.transpose(img, (2, 0, 1))
        # 간단한 정규화 (InsightFace 예제에 맞춰서)
        img = (img - 127.5) / 128.0
        inp = img[np.newaxis, :, :, :]

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: inp})
    except Exception as e:
        return {
            "ok": False,
            "error": f"genderage.onnx 추론 실패: {e}",
            "age": None,
            "gender_label": None,
            "gender_scores": None,
        }

    gender_label = None
    gender_scores: Optional[Dict[str, float]] = None
    age_val: Optional[float] = None

    try:
        # 일반적으로 [gender_logits, age_logits] 두 개의 출력이 온다고 가정
        if len(outputs) >= 2:
            g = np.array(outputs[0]).reshape(-1)
            a = np.array(outputs[1]).reshape(-1)

            # gender
            if g.size >= 2:
                probs_g = _softmax(g)
                idx = int(probs_g.argmax())
                gender_label = "Man" if idx == 1 else "Woman"
                gender_scores = {
                    "Woman": float(probs_g[0] * 100.0),
                    "Man": float(probs_g[1] * 100.0),
                }

            # age: 0~N-1 index 의 기대값
            if a.size > 0:
                probs_a = _softmax(a)
                ages = np.arange(len(probs_a), dtype="float32")
                age_val = float((probs_a * ages).sum())
        else:
            # 출력이 하나인 경우, 그냥 분포라고 보고 기대값 계산
            a = np.array(outputs[0]).reshape(-1)
            if a.size > 0:
                probs_a = _softmax(a)
                ages = np.arange(len(probs_a), dtype="float32")
                age_val = float((probs_a * ages).sum())
    except Exception as e:
        return {
            "ok": False,
            "error": f"genderage 결과 파싱 실패: {e}",
            "age": None,
            "gender_label": None,
            "gender_scores": None,
        }

    return {
        "ok": True,
        "age": age_val,
        "gender_label": gender_label,
        "gender_scores": gender_scores,
    }


# ------------------------------------------------
# age_googlenet.onnx : 나이 전용
# ------------------------------------------------

_AGE_BUCKETS = [
    (0, 2),
    (4, 6),
    (8, 12),
    (15, 20),
    (25, 32),
    (38, 43),
    (48, 53),
    (60, 100),
]
_AGE_BUCKET_MID = [(a + b) / 2.0 for (a, b) in _AGE_BUCKETS]


def _predict_age_googlenet_from_bgr(bgr: "np.ndarray") -> Dict[str, Any]:  # type: ignore[name-defined]
    if np is None or cv2 is None or ort is None:
        return {
            "ok": False,
            "error": "numpy / cv2 / onnxruntime 중 하나가 없습니다.",
            "age": None,
        }

    session = _get_age_googlenet_session()
    if session is None:
        return {
            "ok": False,
            "error": "age_googlenet.onnx 세션 초기화 실패",
            "age": None,
        }

    try:
        img = cv2.resize(bgr, (224, 224))
        img = img.astype("float32")

        # CHW
        img = np.transpose(img, (2, 0, 1))
        # 평균값 빼기 (OpenCV 예제 기준 BGR mean)
        mean = np.array([104.0, 117.0, 123.0], dtype="float32").reshape(3, 1, 1)
        img = img - mean
        inp = img[np.newaxis, :, :, :]

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: inp})
    except Exception as e:
        return {
            "ok": False,
            "error": f"age_googlenet.onnx 추론 실패: {e}",
            "age": None,
        }

    try:
        out = np.array(outputs[0]).reshape(-1)
        if out.size == len(_AGE_BUCKET_MID):
            probs = _softmax(out)
            mids = np.array(_AGE_BUCKET_MID, dtype="float32")
            age_val = float((probs * mids).sum())
        else:
            # 크기가 다르면 argmax index 정도만 사용
            idx = int(out.argmax())
            age_val = float(idx)
    except Exception as e:
        return {
            "ok": False,
            "error": f"age_googlenet 결과 파싱 실패: {e}",
            "age": None,
        }

    return {
        "ok": True,
        "age": age_val,
    }


# ------------------------------------------------
# 최종 앙상블
# ------------------------------------------------

def _trimmed_mean(values: list[float], trim_ratio: float = 0.2) -> float:
    """
    간단한 trimmed mean: 양쪽에서 일부 값을 버리고 평균.
    모델이 3개일 때 이상치를 조금 줄이기 위한 용도.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    vs = sorted(values)
    n = len(vs)
    k = int(n * trim_ratio)
    if k >= 1 and n > 2 * k:
        vs = vs[k:-k]
    return float(sum(vs) / len(vs))


def analyze_age_ensemble(image_base64: str) -> Dict[str, Any]:
    """
    /age-ensemble 엔드포인트에서 사용하는 메인 함수.

    반환 예시:
    {
      "age": 31.2,          # 최종 앙상블 나이
      "final_age": 31.2,    # 동일 값 (호환용)
      "gender": { "Man": 95.7, "Woman": 4.3 },  # DeepFace 기준
      "ages": {
        "deepface": 30.8,
        "uniface": 29.5
      },
      "models": {
        "deepface": {...},
        "genderage": {...},
        "age_googlenet": {...}
      },
      "model_count": 3
    }
    """
    if np is None or cv2 is None:
        return {
            "age": None,
            "final_age": None,
            "gender": None,
            "ages": {},
            "models": {},
            "model_count": 0,
            "error": "numpy / opencv 없음",
        }

    bgr = _decode_base64_to_bgr(image_base64)
    if bgr is None:
        return {
            "age": None,
            "final_age": None,
            "gender": None,
            "ages": {},
            "models": {},
            "model_count": 0,
            "error": "이미지 디코딩 실패",
        }

    
    
    # 1) DeepFace
    deep_res = _predict_deepface_from_bgr(bgr)

    # 2) UniFace (AgeGender)
    uniface_res = _predict_uniface_from_bgr(bgr)

    models: Dict[str, Dict[str, Any]] = {
        "deepface": deep_res,
        "uniface": uniface_res,
    }
    ages: Dict[str, float] = {}
    for name, res in models.items():
        age_v = _safe_float(res.get("age"))
        if age_v is not None:
            ages[name] = age_v
    final_age: Optional[float] = None
    if ages:
        final_age = _trimmed_mean(list(ages.values()), trim_ratio=0.2)

    # 성별은 DeepFace 결과를 우선 사용
    gender_payload = None
    if deep_res.get("gender_scores"):
        gender_payload = {
            str(k): float(v) for k, v in deep_res["gender_scores"].items()
        }
    elif deep_res.get("gender_label"):
        gender_payload = {str(deep_res["gender_label"]): 100.0}

    return {
        "age": _safe_float(final_age),
        "final_age": _safe_float(final_age),
        "gender": gender_payload,
        "ages": {k: float(v) for k, v in ages.items()},
        "models": models,
        "model_count": len(ages),
    }
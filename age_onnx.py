from __future__ import annotations

"""
age_onnx.py

- age_googlenet.onnx ONNX 모델을 직접 불러서
  얼굴 나이(대략적인 연령)를 추정하는 모듈.
- DeepFace / UniFace 전혀 사용하지 않는다.
"""

from typing import Any, Dict, Tuple

import base64
import logging
import os

import numpy as np

try:
    import cv2  # type: ignore[import]
except Exception:  # noqa: BLE001
    cv2 = None  # type: ignore[assignment]

try:
    import onnxruntime as ort  # type: ignore[import]
except Exception:  # noqa: BLE001
    ort = None  # type: ignore[assignment]

LOGGER = logging.getLogger("softtech_age_onnx")

_SESSION: "ort.InferenceSession | None" = None  # type: ignore[name-defined]
_INPUT_NAME: str | None = None
_INPUT_SHAPE: Tuple[int, ...] | None = None


def _get_model_path() -> str:
    """
    환경변수 AGE_ONNX_MODEL_PATH 가 있으면 우선 사용하고,
    없으면 기본 경로 models/age_googlenet.onnx 를 사용한다.
    """
    env_path = os.getenv("AGE_ONNX_MODEL_PATH")
    if env_path:
        return env_path
    return "models/age_googlenet.onnx"


def _init_session() -> None:
    """
    ONNX Runtime 세션을 1회 초기화.
    """
    global _SESSION, _INPUT_NAME, _INPUT_SHAPE

    if _SESSION is not None:
        return

    if ort is None:
        raise RuntimeError("onnxruntime 이 설치되어 있지 않습니다.")

    if cv2 is None:
        raise RuntimeError("opencv-python(cv2) 가 설치되어 있지 않습니다.")

    model_path = _get_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX 모델 파일을 찾을 수 없습니다: {model_path}")

    LOGGER.info("Loading ONNX model from %s", model_path)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    _SESSION = sess

    first_input = sess.get_inputs()[0]
    _INPUT_NAME = first_input.name
    # 보통 [1, 3, 224, 224] 형태
    shape = tuple(int(x) if isinstance(x, (int, float)) else -1 for x in first_input.shape)
    _INPUT_SHAPE = shape  # type: ignore[assignment]

    LOGGER.info("ONNX model input name=%s shape=%s", _INPUT_NAME, _INPUT_SHAPE)


def _preprocess_bgr(bgr: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
    """
    age_googlenet.onnx 에 맞게 BGR 이미지를 전처리한다.

    - 입력: BGR, uint8, HxWx3
    - 처리:
        - 입력 크기 → 모델 입력 크기로 resize
        - float32 변환
        - 평균 [104, 117, 123] (BGR) 을 빼고
        - CHW 로 transpose 후 배치 차원 추가
    """
    if _INPUT_SHAPE is None:
        # 기본값 (1, 3, 224, 224)
        h, w = 224, 224
    else:
        # (N, C, H, W)
        shape = _INPUT_SHAPE
        h = int(shape[2]) if len(shape) >= 3 and shape[2] > 0 else 224
        w = int(shape[3]) if len(shape) >= 4 and shape[3] > 0 else 224

    resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype("float32")
    mean = np.array([104.0, 117.0, 123.0], dtype="float32")
    arr -= mean
    # HWC -> CHW
    chw = np.transpose(arr, (2, 0, 1))
    # 배치 차원 추가
    batched = np.expand_dims(chw, axis=0)
    return batched


def _softmax(logits: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
    """
    단순 softmax 구현.
    """
    logits = logits.astype("float32")
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(logits)
    sums = np.sum(exps, axis=-1, keepdims=True)
    return exps / np.maximum(sums, 1e-8)


def predict_age_from_bgr(bgr: "np.ndarray") -> Tuple[float, Dict[str, Any]]:  # type: ignore[name-defined]
    """
    BGR 이미지를 받아서 연령을 추정한다.

    반환:
      (age_years, extra_info)
    """
    _init_session()
    assert _SESSION is not None
    assert _INPUT_NAME is not None

    inp = _preprocess_bgr(bgr)
    outputs = _SESSION.run(None, {_INPUT_NAME: inp})

    if not outputs:
        raise RuntimeError("ONNX 모델 출력이 비어 있습니다.")

    logits = outputs[0]
    # 보통 (1, 101) 형태
    logits = np.array(logits).reshape(1, -1)
    probs = _softmax(logits)[0]

    ages = np.arange(probs.shape[0], dtype="float32")
    age_value = float(np.sum(probs * ages))

    extra = {
        "probs": probs.tolist(),
        "age_labels": ages.tolist(),
    }
    return age_value, extra


def decode_base64_to_bgr(image_base64: str) -> "np.ndarray":  # type: ignore[name-defined]
    """
    base64 → BGR (OpenCV) 이미지로 변환.
    """
    if cv2 is None:
        raise RuntimeError("opencv-python(cv2) 가 설치되어 있지 않습니다.")

    try:
        raw = base64.b64decode(image_base64, validate=True)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"BASE64_DECODE_ERROR: {e}") from e

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("IMAGE_DECODE_ERROR: cv2.imdecode 실패")
    return img


def run_age_from_base64(image_base64: str) -> Dict[str, Any]:
    """
    base64 문자열을 받아 나이를 추정하고
    결과 딕셔너리 형태로 반환한다.
    """
    bgr = decode_base64_to_bgr(image_base64)
    age_value, extra = predict_age_from_bgr(bgr)

    return {
        "ok": True,
        "age": float(age_value),
        "age_raw": float(age_value),
        "extra": extra,
    }

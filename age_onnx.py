from __future__ import annotations

"""
age_onnx.py

ONNX Runtime 기반 나이 추정 모듈.
- 로컬에 있는 age_googlenet.onnx (또는 환경변수로 지정한 경로)를 사용해
  얼굴 나이를 근사적으로 예측한다.
"""

from typing import Any
import os

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None  # type: ignore


_SESSION: "ort.InferenceSession | None" = None
_INPUT_NAME: str | None = None
_INPUT_SHAPE: list[int] | None = None


def _get_model_path() -> str:
    """
    모델 경로를 결정한다.

    우선순위:
      1) 환경변수 AGE_ONNX_MODEL_PATH
      2) 기본값 "models/age_googlenet.onnx"
    """
    env_path = os.getenv("AGE_ONNX_MODEL_PATH")
    if env_path:
        return env_path
    return "models/age_googlenet.onnx"


def _init_session(model_path: str | None = None) -> None:
    """ONNX Runtime 세션을 1회 초기화."""
    global _SESSION, _INPUT_NAME, _INPUT_SHAPE

    if _SESSION is not None:
        return

    if ort is None:
        raise RuntimeError("onnxruntime 이 설치되어 있지 않습니다.")

    if cv2 is None:
        raise RuntimeError("opencv-python(cv2) 가 설치되어 있지 않습니다.")

    path = model_path or _get_model_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"ONNX 모델 파일을 찾을 수 없습니다: {path}")

    sess = ort.InferenceSession(
        path,
        providers=["CPUExecutionProvider"],
    )
    _SESSION = sess
    first_input = sess.get_inputs()[0]
    _INPUT_NAME = first_input.name
    _INPUT_SHAPE = list(first_input.shape)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full_like(x, 1.0 / max(len(x), 1))
    return e / s


def predict_age_onnx(image_bgr: "np.ndarray") -> float:
    """
    BGR 이미지(예: cv2.imread 결과)를 받아서 근사 나이(float)를 반환한다.

    실패 시 예외를 발생시킨다.
    """
    if image_bgr is None or not hasattr(image_bgr, "shape"):
        raise ValueError("유효하지 않은 이미지입니다 (None).")

    _init_session()
    assert _SESSION is not None
    assert _INPUT_NAME is not None

    # 기본 입력 크기는 224x224 로 가정
    h, w = image_bgr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"이미지 크기가 잘못되었습니다: {image_bgr.shape}")

    img = cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_AREA)

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # [0,255] -> [0,1]
    x = img.astype("float32") / 255.0

    # HWC -> CHW, 배치 차원 추가
    x = np.transpose(x, (2, 0, 1))  # (3, 224, 224)
    x = np.expand_dims(x, 0)        # (1, 3, 224, 224)

    # ONNX 추론
    outputs = _SESSION.run(None, {_INPUT_NAME: x})
    if not outputs:
        raise RuntimeError("ONNX 추론 결과가 비어 있습니다.")

    out = outputs[0]
    if out is None:
        raise RuntimeError("ONNX 출력이 None 입니다.")

    arr = np.asarray(out).astype("float32").ravel()
    if arr.size == 0:
        raise RuntimeError("ONNX 출력 배열이 비어 있습니다.")

    # 출력 형태에 따라 처리:
    # - 1차원 1개 값: 직접 나이로 사용
    # - 1차원 N개 값: 0..N-1 의 가중 평균으로 근사 (age 0~N-1 가정)
    if arr.size == 1:
        age_raw = float(arr[0])
    else:
        probs = _softmax(arr)
        ages = np.arange(probs.shape[0], dtype="float32")
        age_raw = float((probs * ages).sum())

    if not np.isfinite(age_raw):
        raise RuntimeError(f"유효하지 않은 나이 값이 나왔습니다: {age_raw!r}")

    return float(age_raw)

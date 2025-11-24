from __future__ import annotations

"""
age_onnx.py

ONNX 기반 나이 추정 모듈.

- Render 서버에서 age_googlenet.onnx 모델을 로컬 ONNX Runtime 으로 실행한다.
- 모델 경로는 환경변수 AGE_ONNX_MODEL_PATH 를 우선 사용하고,
  없으면 기본값 "models/age_googlenet.onnx" 를 사용한다.
"""

from typing import Any, Dict

import os
import base64
import logging

try:
    import numpy as np  # type: ignore
except Exception:  # numpy 미설치 시
    np = None  # type: ignore

try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore


logger = logging.getLogger("age_onnx")

_SESSION: "ort.InferenceSession | None" = None  # type: ignore
_INPUT_NAME: str | None = None
_INPUT_SHAPE: Any = None


def _init_session() -> None:
    """ONNX Runtime 세션을 1회 초기화."""
    global _SESSION, _INPUT_NAME, _INPUT_SHAPE

    if _SESSION is not None:
        return

    if ort is None:
        raise RuntimeError(
            "onnxruntime 이 설치되어 있지 않습니다. venv 에 onnxruntime 을 설치하세요."
        )
    if np is None:
        raise RuntimeError(
            "numpy 가 설치되어 있지 않습니다. venv 에 numpy 를 설치하세요."
        )
    if cv2 is None:
        raise RuntimeError(
            "opencv-python(cv2) 가 설치되어 있지 않습니다. venv 에 opencv-python 을 설치하세요."
        )

    model_path = os.getenv("AGE_ONNX_MODEL_PATH") or "models/age_googlenet.onnx"
    logger.info("Loading ONNX age model from %s", model_path)

    sess = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    _SESSION = sess
    first_input = sess.get_inputs()[0]
    _INPUT_NAME = first_input.name
    _INPUT_SHAPE = first_input.shape


def _decode_image(image_base64: str) -> "np.ndarray":
    """base64 문자열을 BGR 이미지(ndarray)로 디코딩."""
    if np is None or cv2 is None:
        raise RuntimeError("numpy 또는 cv2 가 설치되어 있지 않습니다.")

    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    img_bytes = base64.b64decode(image_base64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩 실패")
    return img


def _prepare_input(img: "np.ndarray") -> "np.ndarray":
    """
    ONNX 입력 텐서를 구성한다.
    - 기본적으로 224x224, NCHW(1,3,H,W) 또는 NHWC(1,H,W,3)를 지원한다.
    """
    if np is None or cv2 is None:
        raise RuntimeError("numpy 또는 cv2 가 설치되어 있지 않습니다.")

    # 기본값
    target_h = 224
    target_w = 224
    channels_first = True

    if _INPUT_SHAPE is not None and isinstance(_INPUT_SHAPE, (list, tuple)):
        shape = list(_INPUT_SHAPE)
        if len(shape) == 4:
            n, c_or_h, h_or_w, last = shape
            # (1,3,H,W) 형식
            if shape[1] == 3:
                _, _, h, w = shape
                target_h, target_w = int(h), int(w)
                channels_first = True
            # (1,H,W,3) 형식
            elif shape[3] == 3:
                _, h, w, _ = shape
                target_h, target_w = int(h), int(w)
                channels_first = False

    resized = cv2.resize(img, (target_w, target_h))
    x = resized.astype("float32") / 255.0

    if channels_first:
        # HWC -> CHW
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)  # (1,3,H,W)
    else:
        # HWC 그대로 두고 배치 차원만 추가
        x = np.expand_dims(x, axis=0)  # (1,H,W,3)

    return x


def predict_age_onnx(image_base64: str) -> Dict[str, Any]:
    """
    ONNX 모델로 나이만 추정하는 함수.
    실패 시 {"ok": False, "error": "..."} 형태로 리턴한다.
    """
    try:
        _init_session()
    except Exception as e:
        return {
            "ok": False,
            "error": f"ONNX 세션 초기화 실패: {e!r}",
        }

    try:
        img = _decode_image(image_base64)
    except Exception as e:
        return {
            "ok": False,
            "error": f"이미지 디코딩 실패: {e!r}",
        }

    try:
        x = _prepare_input(img)
    except Exception as e:
        return {
            "ok": False,
            "error": f"입력 전처리 실패: {e!r}",
        }

    try:
        assert _SESSION is not None and _INPUT_NAME is not None
        outputs = _SESSION.run(None, {_INPUT_NAME: x})
    except Exception as e:
        return {
            "ok": False,
            "error": f"ONNX 추론 실패: {e!r}",
        }

    if not outputs:
        return {
            "ok": False,
            "error": "ONNX 출력이 비어 있습니다.",
        }

    raw = outputs[0]

    try:
        arr = np.array(raw, dtype="float32").ravel()
        if arr.size == 0:
            raise ValueError("빈 출력")

        # 일반적인 age_googlenet 계열은 101차원 로짓(0~100세) 확률로 해석 가능
        if arr.size >= 10:
            indices = np.arange(arr.size, dtype="float32")
            # softmax
            m = float(np.max(arr))
            exp = np.exp(arr - m)
            denom = float(np.sum(exp))
            if denom <= 0.0:
                raise ValueError("softmax 분모가 0 이하입니다.")
            prob = exp / denom
            age_val = float(np.sum(indices * prob))
        else:
            # 차원이 작으면 평균값으로 근사
            age_val = float(arr.mean())
    except Exception as e:
        return {
            "ok": False,
            "error": f"ONNX 출력 후처리 실패: {e!r}",
        }

    return {
        "ok": True,
        "age": age_val,
        "raw": str(raw),
    }

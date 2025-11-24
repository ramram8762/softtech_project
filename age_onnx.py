from __future__ import annotations

"""
age_onnx.py

경량 ONNX 기반 나이 추정 모듈 (age_googlenet.onnx 전용).

- ONNX Runtime 로컬에서 동작
- 얼굴 전체를 224x224 등으로 리사이즈해서 나이를 추정
- DeepFace / UniFace 등에 의존하지 않고, 순수 ONNX 모델만 사용
- 실제 사용하는 age_googlenet.onnx 모델의 입/출력 형식에 맞게
  전처리 / 후처리를 조정할 수 있도록 기본 골격만 제공한다.

환경 변수:
  AGE_ONNX_MODEL_PATH  (선택) 모델 경로를 바꾸고 싶을 때 사용.
                         기본값은 "models/age_googlenet.onnx"
"""

from typing import Any, Dict
import base64
import os

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


_SESSION = None
_INPUT_NAME = None
_INPUT_SHAPE = None  # 예: [1, 3, 224, 224] 또는 [1, 224, 224, 3]


def _get_default_model_path() -> str:
    """기본 ONNX 모델 경로를 반환한다."""
    return os.environ.get("AGE_ONNX_MODEL_PATH", "models/age_googlenet.onnx")


def _init_session(model_path: str | None = None) -> None:
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

    model_path = model_path or _get_default_model_path()

    sess = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    _SESSION = sess
    first_input = sess.get_inputs()[0]
    _INPUT_NAME = first_input.name
    _INPUT_SHAPE = first_input.shape  # 예: [1, 3, 224, 224] 또는 [1, 224, 224, 3]


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
    - age_googlenet.onnx 는 보통 224x224 입력을 사용
    - 입력이 NCHW / NHWC 어떤 형식이든 자동으로 맞추려 시도
    """
    if np is None:
        raise RuntimeError("numpy 가 설치되어 있지 않습니다.")
    if _INPUT_SHAPE is None:
        # 기본값: [1, 3, 224, 224] 형태로 가정
        target_h = 224
        target_w = 224
        channels_first = True
    else:
        shape = list(_INPUT_SHAPE)
        # batch 축(보통 1)은 무시하고, 공간/채널 차원만 본다.
        dims = [d for d in shape if isinstance(d, int) and d > 1]
        if len(dims) >= 3:
            # 예: [3, 224, 224] 또는 [224, 224, 3]
            if dims[0] in (1, 3) and dims[1] > 10 and dims[2] > 10:
                # C, H, W
                channels_first = True
                target_h = dims[1]
                target_w = dims[2]
            elif dims[-1] in (1, 3) and dims[0] > 10 and dims[1] > 10:
                # H, W, C
                channels_first = False
                target_h = dims[0]
                target_w = dims[1]
            else:
                channels_first = True
                target_h = 224
                target_w = 224
        else:
            channels_first = True
            target_h = 224
            target_w = 224

    # 얼굴 전체를 단순 리사이즈해서 사용
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    x = resized.astype("float32") / 255.0

    if channels_first:
        # (H, W, C) -> (C, H, W)
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)  # (1, C, H, W)
    else:
        # (H, W, C)
        x = np.expand_dims(x, axis=0)  # (1, H, W, C)

    return x


def predict_age_onnx(image_base64: str) -> Dict[str, Any]:
    """
    ONNX 모델로 나이만 추정하는 함수.

    반환 형식:
      성공 시: {"ok": True, "age": float, "raw": "<원시 출력 str>"}.
      실패 시: {"ok": False, "error": "..."}.
    """
    try:
        _init_session()
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"ONNX 세션 초기화 실패: {e!r}",
        }

    try:
        img = _decode_image(image_base64)
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"이미지 디코딩 실패: {e!r}",
        }

    try:
        x = _prepare_input(img)
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"ONNX 입력 전처리 실패: {e!r}",
        }

    try:
        assert _SESSION is not None and _INPUT_NAME is not None
        outputs = _SESSION.run(None, {_INPUT_NAME: x})
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"ONNX 추론 실패: {e!r}",
        }

    if np is None:
        return {
            "ok": False,
            "error": "numpy 가 설치되어 있지 않아서 출력 후처리를 할 수 없습니다.",
        }

    if not outputs:
        return {
            "ok": False,
            "error": "ONNX 추론 결과가 비어 있습니다.",
        }

    raw = outputs[0]
    try:
        # 출력 형식에 따라 후처리:
        # - 회귀 스칼라라면 그대로
        # - (1, N) 로지츠라면 argmax 로 0~N-1 나이로 가정
        arr = np.array(raw)
        if arr.ndim == 0:
            age_val = float(arr)
        elif arr.ndim == 1:
            # 예: [age]
            age_val = float(arr.mean())
        elif arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] > 1:
            # 예: (1, N) 분류 로지츠 -> argmax 를 나이로 가정
            age_val = float(np.argmax(arr[0]))
        else:
            # 그 외에는 평균값으로 근사
            age_val = float(arr.mean())
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"ONNX 출력 후처리 실패: {e!r}",
        }

    return {
        "ok": True,
        "age": age_val,
        "raw": str(raw),  # 필요하면 나중에 더 정교하게 가공
    }

from __future__ import annotations

"""
age_ensemble.py

- 앱에서 오는 base64 이미지를 받아서
- ONNX(age_googlenet) 기반으로 근사 나이를 추정하고
- 클라이언트가 기대하는 JSON 구조로 가공해 반환한다.
"""

from typing import Any, Dict
import base64

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from age_onnx import predict_age_onnx


def _decode_base64_image(image_b64: str) -> "np.ndarray":
    if cv2 is None:
        raise RuntimeError("opencv-python(cv2) 가 설치되어 있지 않습니다.")

    if not image_b64 or not isinstance(image_b64, str):
        raise ValueError("image_base64 가 비어있습니다.")

    try:
        raw = base64.b64decode(image_b64)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"base64 디코딩 실패: {e}") from e

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩(cv2.imdecode) 실패")

    return img


def _calibrate_age(age_raw: float) -> float:
    """
    간단한 범위 보정:
    - 0 미만 / NaN / inf → 오류로 간주
    - 15 ~ 60 범위로 클램핑
    """
    if not np.isfinite(age_raw):
        raise ValueError(f"유효하지 않은 나이 값: {age_raw!r}")

    age = float(age_raw)
    if age < 0:
        age = 0.0

    # 최소/최대값 보정 (너가 말한 15~60 구간)
    if age < 15.0:
        age = 15.0
    if age > 60.0:
        age = 60.0

    return age


def analyze_age_ensemble(image_b64: str) -> Dict[str, Any]:
    """
    앱에서 오는 base64 이미지를 받아서
    나이 추정 결과를 JSON(dict) 로 돌려준다.
    """
    try:
        img = _decode_base64_image(image_b64)
        age_raw = predict_age_onnx(img)
        age_final = _calibrate_age(age_raw)

        result: Dict[str, Any] = {
            "ok": True,
            "signature": "SOFTTECH_AGE_GOOGLENET_ONLY_V1",
            "age_raw": float(age_raw),
            "age": float(age_final),
            "final_age": float(age_final),
            # 기존 구조를 최대한 유지: ages / gender / models / model_count
            "ages": {
                "age_googlenet": float(age_raw),
            },
            "gender": None,
            "model_count": 1,
            "models": {
                "age_googlenet": {
                    "age": float(age_raw),
                    "confidence": 1.0,
                }
            },
        }
        return result
    except Exception as e:
        # 여기서 발생한 에러는 app.py 에서 HTTP 500 으로 매핑된다.
        return {
            "ok": False,
            "error": "AGE_INFERENCE_ERROR",
            "message": f"{e.__class__.__name__}: {e}",
        }

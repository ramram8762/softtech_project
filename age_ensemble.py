from __future__ import annotations
"""
age_ensemble.py

DeepFace / UniFace 를 모두 제거하고,
age_googlenet.onnx 하나만 사용하는 경량 나이 추정 래퍼.

- 내부에서는 age_onnx.predict_age_onnx() 를 호출한다.
- 클라이언트(앱) 입장에서는 예전 UniFace 서버와 비슷한 JSON 구조를 유지한다.
"""

from typing import Any, Dict, Optional

import math
import logging

from age_onnx import predict_age_onnx

logger = logging.getLogger("age_ensemble")


def _safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
        if math.isfinite(f):
            return f
        return None
    except Exception:
        return None


def _tune_visible_age(raw_age: float) -> float:
    """
    겉보이는 나이를 15~60 세 구간으로 제한하고,
    60 세 부근에서는 최대 약 15 년까지 감산하는 간단한 튜닝.

    - 15세 이하는 15세로 올림
    - 60세 이상은 60세로 잘라낸 뒤,
      15~60 구간에서 선형적으로 최대 15년까지 감산
      (15세 -> 15세, 60세 -> 45세 근사)
    """
    age = max(15.0, min(60.0, float(raw_age)))
    # 15 -> 0, 60 -> 1 로 정규화
    t = (age - 15.0) / 45.0
    t = max(0.0, min(1.0, t))
    reduction = 15.0 * t
    tuned = age - reduction
    return tuned


def analyze_age_ensemble(image_base64: str) -> Dict[str, Any]:
    """
    클라이언트에서 보낸 base64 이미지를 받아
    age_googlenet.onnx 로 근사 나이를 추정한 뒤 JSON 결과로 정리한다.
    """
    signature = "SOFTTECH_AGE_GOOGLENET_ONLY_V1"

    if not image_base64:
        return {
            "signature": signature,
            "ok": False,
            "error": "EMPTY_IMAGE",
            "age": None,
            "final_age": None,
            "gender": None,
            "ages": {},
            "models": {},
            "model_count": 0,
        }

    core = predict_age_onnx(image_base64)
    if not core or not core.get("ok"):
        logger.warning("predict_age_onnx failed: %s", core)
        return {
            "signature": signature,
            "ok": False,
            "error": core.get("error", "AGE_ONNX_FAILED") if isinstance(core, dict) else "AGE_ONNX_FAILED",
            "age": None,
            "final_age": None,
            "gender": None,
            "ages": {},
            "models": {},
            "model_count": 0,
        }

    raw_age = _safe_float(core.get("age"))
    if raw_age is None:
        logger.warning("predict_age_onnx returned invalid age: %s", core)
        return {
            "signature": signature,
            "ok": False,
            "error": "INVALID_AGE_VALUE",
            "age": None,
            "final_age": None,
            "gender": None,
            "ages": {},
            "models": {},
            "model_count": 0,
        }

    tuned_age = _tune_visible_age(raw_age)

    ages = {"age_googlenet": float(raw_age)}
    models = {
        "age_googlenet": {
            "age": float(raw_age),
            "provider": "onnxruntime_cpu",
            "weight": 1.0,
        }
    }

    # 현재는 age_googlenet 에서 성별 정보를 제공하지 않으므로 None
    gender_payload = None

    return {
        "signature": signature,
        "ok": True,
        "age": float(tuned_age),
        "final_age": float(tuned_age),
        "gender": gender_payload,
        "ages": ages,
        "models": models,
        "model_count": 1,
    }

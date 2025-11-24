from __future__ import annotations

"""
age_ensemble.py

- DeepFace / UniFace 완전히 제거.
- age_googlenet.onnx 하나만 사용해서 나이를 추정한다.
- React Native 앱에서 기대하는 JSON 구조에 맞춰 응답을 조립한다.
"""

from typing import Any, Dict

import logging

from age_onnx import run_age_from_base64

LOGGER = logging.getLogger("softtech_age_ensemble")

SIGNATURE = "SOFTTECH_AGE_GOOGLENET_ONLY_V1"


def analyze_age_ensemble(image_base64: str) -> Dict[str, Any]:
    """
    React Native 쪽에서 사용하는 통합 진입점.

    입력:
      - image_base64: 사진 JPEG 을 base64 로 인코딩한 문자열

    출력 예시:
      {
        "ok": true,
        "signature": "SOFTTECH_AGE_GOOGLENET_ONLY_V1",
        "age_raw": 31.2,
        "age": 31.2,
        "final_age": 31.2,
        "ages": { "age_googlenet": 31.2 },
        "models": {
          "age_googlenet": { "age": 31.2, "weight": 1.0 }
        },
        "model_count": 1,
        "gender": null
      }
    """
    try:
        base = run_age_from_base64(image_base64)
        if not base.get("ok"):
            # 이 경우는 거의 없지만, 방어적으로 처리
            return {
                "ok": False,
                "error": base.get("error", "AGE_MODEL_FAILED"),
                "message": base.get("message", ""),
                "signature": SIGNATURE,
            }

        age_value = float(base.get("age", 0.0))
        # 여기서는 특별한 튜닝 없이 그대로 전달.
        age = age_value
        final_age = age_value

        result: Dict[str, Any] = {
            "ok": True,
            "signature": SIGNATURE,
            "age_raw": age_value,
            "age": age,
            "final_age": final_age,
            "ages": {
                "age_googlenet": age_value,
            },
            "models": {
                "age_googlenet": {
                    "age": age_value,
                    "weight": 1.0,
                }
            },
            "model_count": 1,
            # 성별 정보는 더 이상 제공하지 않음.
            "gender": None,
        }
        return result

    except Exception as e:  # noqa: BLE001
        LOGGER.exception("analyze_age_ensemble 실패: %s", e)
        return {
            "ok": False,
            "error": "EXCEPTION",
            "message": f"{e.__class__.__name__}: {e}",
            "signature": SIGNATURE,
        }

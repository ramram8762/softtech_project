from __future__ import annotations

"""
SoftTech 통합 나이 분석 모듈 (age_googlenet 전용 버전)

- DeepFace / UniFace 완전 제거
- age_googlenet.onnx(ONNX Runtime) 결과만 사용
- 모바일 앱에서는 이 모듈의 analyze_age_ensemble() 만 호출하면 됨
"""

from typing import Any, Dict

from age_onnx import predict_age_onnx


SIGNATURE = "SOFTTECH_AGE_GOOGLENET_ONLY_V1"


def _safe_int(x: Any) -> int | None:
    try:
        v = float(x)
        if v != v:  # NaN 체크
            return None
        return int(round(v))
    except Exception:
        return None


def analyze_age_ensemble(image_base64: str) -> Dict[str, Any]:
    """
    모바일 앱에서 사용하는 진입 함수.

    입력:
        - image_base64: data URL prefix 포함/미포함 상관 없이 base64 문자열

    출력(JSON dict 예시):
        {
            "ok": true,
            "signature": "SOFTTECH_AGE_GOOGLENET_ONLY_V1",
            "age": 32,
            "final_age": 32,
            "age_raw": 31.7,
            "ages": {
                "age_googlenet": 32
            },
            "models": {
                "age_googlenet": {
                    "age_raw": 31.7
                }
            },
            "model_count": 1,
            "gender": {
                "Man": 50.0,
                "Woman": 50.0
            }
        }
    """
    # ONNX 쪽에 위임
    onnx_result = predict_age_onnx(image_base64)

    if not isinstance(onnx_result, dict) or not onnx_result.get("ok"):
        # ONNX 쪽에서 에러 난 경우, 그대로 ok:false 로 전달
        return {
            "ok": False,
            "signature": SIGNATURE,
            "error": onnx_result.get("error") if isinstance(onnx_result, dict) else "ONNX_ERROR",
        }

    age_raw = onnx_result.get("age_raw")
    age_int = _safe_int(age_raw)

    if age_int is None:
        return {
            "ok": False,
            "signature": SIGNATURE,
            "error": "AGE_PARSE_ERROR",
        }

    # 앱 호환을 위해 예전 구조 유지
    ages = {
        "age_googlenet": age_int,
    }
    models = {
        "age_googlenet": {
            "age_raw": float(age_raw),
        }
    }

    # 성별은 이 버전에서 추정하지 않으므로, 중립값으로 고정
    gender_payload = {
        "Man": 50.0,
        "Woman": 50.0,
    }

    return {
        "ok": True,
        "signature": SIGNATURE,
        "age": age_int,
        "final_age": age_int,
        "age_raw": float(age_raw),
        "ages": ages,
        "models": models,
        "model_count": len(ages),
        "gender": gender_payload,
    }

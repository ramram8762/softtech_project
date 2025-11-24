from __future__ import annotations

"""
SoftTech 통합 나이 분석 모듈 (연령대 전용 버전)

- DeepFace / UniFace / genderage 완전 제거
- age_googlenet.onnx 의 "연령대 그룹" 정보만 래핑
- 성별 정보 완전 제거
"""

from typing import Any, Dict

from age_onnx import predict_age_onnx


SIGNATURE = "SOFTTECH_AGE_GOOGLENET_GROUPS_V1"


def analyze_age_ensemble(image_base64: str) -> Dict[str, Any]:
    """
    모바일 앱에서 사용하는 진입 함수.

    입력:
        - image_base64: data URL prefix 포함/미포함 상관 없이 base64 문자열

    출력(JSON dict 예시):
        {
            "ok": true,
            "signature": "SOFTTECH_AGE_GOOGLENET_GROUPS_V1",
            "provider": "age_googlenet",
            "num_groups": 8,
            "group_index": 3,
            "group_probs": [...],
            "group_logits": [...]
        }

    ※ 나이(정수/실수), 성별, 우리쪽 보정 수식 전부 포함하지 않음.
    """
    base = predict_age_onnx(image_base64)

    if not isinstance(base, dict) or not base.get("ok"):
        return {
            "ok": False,
            "signature": SIGNATURE,
            "error": base.get("error") if isinstance(base, dict) else "ONNX_ERROR",
        }

    return {
        "ok": True,
        "signature": SIGNATURE,
        "provider": base.get("provider", "age_googlenet"),
        "num_groups": base.get("num_groups"),
        "group_index": base.get("group_index"),
        "group_probs": base.get("group_probs"),
        "group_logits": base.get("group_logits"),
    }

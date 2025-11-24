from __future__ import annotations

"""
SoftTech 통합 나이 분석 모듈 (연령대 전용 + 어린 구간 측정불가)

- DeepFace / UniFace / genderage 전부 사용 안 함
- age_googlenet.onnx 의 "연령대 그룹" 정보만 사용
- 성별 정보 없음
- 0~2번 그룹(아주 어린 연령대)은 "측정불가"로만 표시
"""

from typing import Any, Dict

from age_onnx import predict_age_onnx


SIGNATURE = "SOFTTECH_AGE_GOOGLENET_GROUPS_V2"

# 많이 쓰이는 age_googlenet 8개 연령대 정의
AGE_GROUP_LABELS = [
    "0-2",
    "4-6",
    "8-12",
    "15-20",
    "25-32",
    "38-43",
    "48-53",
    "60+",
]


def analyze_age_ensemble(image_base64: str) -> Dict[str, Any]:
    """
    모바일 앱에서 사용하는 진입 함수.

    입력:
        - image_base64: data URL prefix 포함/미포함 상관 없이 base64 문자열

    출력(JSON dict 예시):
        {
            "ok": true,
            "signature": "SOFTTECH_AGE_GOOGLENET_GROUPS_V2",
            "provider": "age_googlenet",
            "num_groups": 8,
            "group_index": 4,
            "group_label": "25-32",
            "group_probs": [...],
            "group_logits": [...],
            "measurable": true   # 0~2번 구간이면 false
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

    num_groups = int(base.get("num_groups", 0))
    group_index_raw = base.get("group_index", -1)

    try:
        group_index = int(group_index_raw)
    except Exception:
        group_index = -1

    # 라벨 추출 (안전하게)
    group_label = None
    if 0 <= group_index < len(AGE_GROUP_LABELS):
        group_label = AGE_GROUP_LABELS[group_index]

    # 0,1,2 그룹(아주 어린 연령대)은 "측정불가"
    # 그 외(3~7)는 측정 가능
    measurable = bool(group_index >= 3)

    return {
        "ok": True,
        "signature": SIGNATURE,
        "provider": base.get("provider", "age_googlenet"),
        "num_groups": num_groups,
        "group_index": group_index,
        "group_label": group_label,
        "group_probs": base.get("group_probs"),
        "group_logits": base.get("group_logits"),
        "measurable": measurable,
    }

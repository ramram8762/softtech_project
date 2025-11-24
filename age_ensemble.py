from __future__ import annotations
"""
통합 나이 추정 모듈 (age_googlenet 단독 버전).

- DeepFace / UniFace 는 완전히 제거.
- age_googlenet.onnx (ONNX Runtime) 결과만 사용해서 "근사치 나이"를 얻는다.
- 응답 JSON 은 기존 프런트엔드와 최대한 호환되도록
  `age`, `final_age`, `ages`, `models`, `signature` 필드를 유지한다.
- 15~60세 범위 내에서만 보이는 나이를 사용하고,
  60세 근처에서는 최대 15살까지 완만하게 감산하는 튜닝을 적용한다.
  (이 값은 이후 앱의 16가지 피부 분석 지표로 한 번 더 보정해서
   최종 "피부 나이"를 만들 수 있다.)
"""

from typing import Any, Dict, Optional

from age_onnx import predict_age_onnx


def _safe_float(x: Any) -> Optional[float]:
    """넘어온 값을 안전하게 float 로 변환한다. 실패 시 None."""
    try:
        if x is None:
            return None
        return float(x)
    except Exception:  # noqa: BLE001
        return None


def _calibrate_visible_age_15_60(raw: Optional[float]) -> Optional[float]:
    """
    15~60 범위 내에서만 '겉보이는 피부 나이'를 보여 주기 위한 튜닝 함수.

    - 입력 나이가 15 미만이면 15로 올려 준다.
    - 60 초과면 60으로 잘라 준다.
    - 15~60 구간에서 위로 갈수록 최대 15살까지 조금씩 감소시킨다.
      (예: 60 → 약 45 부근)
    """
    raw_f = _safe_float(raw)
    if raw_f is None:
        return None

    # 기본 범위 클리핑
    if raw_f < 15.0:
        raw_f = 15.0
    if raw_f > 60.0:
        raw_f = 60.0

    # 15~60 구간을 0~1 로 정규화
    t = (raw_f - 15.0) / 45.0  # 0~1
    # t 가 커질수록 더 많이 빼 주되, 60 에서 약 15년 정도 빠지도록
    subtract = 15.0 * (t ** 1.2)  # 지수 1.2 는 완만한 곡선
    calibrated = raw_f - subtract

    # 안전 범위 보정
    if calibrated < 15.0:
        calibrated = 15.0
    if calibrated > 60.0:
        calibrated = 60.0

    return calibrated


def analyze_age_ensemble(image_base64: str) -> Dict[str, Any]:
    """
    클라이언트에서 보낸 base64 이미지를 받아
    age_googlenet.onnx 를 통해 나이를 추정하고,
    SoftTech 앱에서 사용하는 JSON 포맷으로 감싸서 리턴한다.
    """
    signature = "SOFTTECH_AGE_GOOGLENET_ONLY_V1"

    base = predict_age_onnx(image_base64)
    if not base.get("ok"):
        # ONNX 단계에서 실패한 경우
        return {
            "signature": signature,
            "ok": False,
            "error": base.get("error") or "AGE_ONNX_ERROR",
            "age": None,
            "age_raw": None,
            "final_age": None,
            "gender": None,
            "ages": {},
            "models": {},
            "model_count": 0,
        }

    raw_age = base.get("age")
    age_raw = _safe_float(raw_age)
    visible_age = _calibrate_visible_age_15_60(age_raw)

    if visible_age is None:
        return {
            "signature": signature,
            "ok": False,
            "error": "AGE_CALIBRATION_FAILED",
            "age": None,
            "age_raw": age_raw,
            "final_age": None,
            "gender": None,
            "ages": {},
            "models": {},
            "model_count": 0,
        }

    ages: Dict[str, float] = {}
    models: Dict[str, Dict[str, Any]] = {}

    if age_raw is not None:
        ages["age_googlenet"] = float(age_raw)
        models["age_googlenet"] = {
            "age": float(age_raw),
        }

    return {
        "signature": signature,
        "ok": True,
        "error": None,
        # age_raw: ONNX 모델이 그대로 뱉은 값 (근사치 얼굴 나이)
        "age_raw": age_raw,
        # age / final_age: 15~60 범위 튜닝이 적용된 값
        "age": visible_age,
        "final_age": visible_age,
        # 이제 성별은 사용하지 않으므로 None 으로 둔다.
        "gender": None,
        "ages": ages,
        "models": models,
        "model_count": len(ages),
    }

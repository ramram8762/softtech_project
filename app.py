
from __future__ import annotations

from typing import Any, Dict

from flask import Flask, request, jsonify

# 같은 폴더에 있는 age_ensemble.py 에서 통합 로직을 가져온다.
from age_ensemble import analyze_age_ensemble, predict_age_deepface

app = Flask(__name__)


@app.get("/health")
def health() -> Any:
    """서버 헬스 체크."""
    return jsonify({"ok": True, "status": "alive"})


@app.post("/age")
def age_deepface() -> Any:
    """
    DeepFace 단일 나이 예측 (기존 호환용).

    요청 JSON:
    {
      "image_base64": "data:image/jpeg;base64,..."
    }

    응답(JSON, 성공 시 예시):
    {
      "ok": true,
      "age": 31.0,
      "gender_label": "Man",
      "gender_scores": { "Man": 95.7, "Woman": 4.3 },
      "raw": { ... DeepFace 원본 결과 ... }
    }
    """
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    image_b64 = data.get("image_base64")
    if not image_b64:
        return (
            jsonify({"ok": False, "error": "image_base64 필드가 필요합니다."}),
            400,
        )

    res = predict_age_deepface(image_b64)
    status = 200 if res.get("ok") else 500
    return jsonify(res), status


@app.post("/age-ensemble")
def age_ensemble_api() -> Any:
    """
    DeepFace + age_googlenet (ONNX Model Zoo) 를 함께 돌리는 엔드포인트.

    요청 JSON:
    {
      "image_base64": "data:image/jpeg;base64,..."
    }

    응답(JSON, 성공 시 예시):
    {
      "ok": true,
      "age": 33.2,                 # 앙상블(합산) 나이 → 앱에서 '피부 분석 나이'에 사용
      "gender": { "Man": 95.7, ... },  # DeepFace gender 결과
      "models": {
        "deepface":    { "ok": true, "age": 31.0, ... },
        "age_googlenet": { "ok": true, "age": 35.0, ... }
      },
      "ages": {
        "deepface": 31.0,
        "age_googlenet": 35.0
      },
      "model_count": 2
    }
    """
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    image_b64 = data.get("image_base64")
    if not image_b64:
        return (
            jsonify({"ok": False, "error": "image_base64 필드가 필요합니다."}),
            400,
        )

    result = analyze_age_ensemble(image_b64)
    return jsonify({"ok": True, **result}), 200


if __name__ == "__main__":
    # 개발용: 0.0.0.0:8000
    app.run(host="0.0.0.0", port=8000, debug=False)

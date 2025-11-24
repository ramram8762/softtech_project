from __future__ import annotations

"""
app.py

SoftTech 얼굴 나이 추정용 Flask 서버.

- GET  /            : 헬스 체크
- POST /age-ensemble: base64 이미지를 받아 나이 추정 결과 반환
"""

import os
from typing import Any, Dict

from flask import Flask, request, jsonify

from age_ensemble import analyze_age_ensemble

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index() -> Any:
    """헬스 체크 엔드포인트."""
    return jsonify(
        {
            "ok": True,
            "service": "SoftTech AgeGoogLeNet Age API",
            "version": "4-0-age_googlenet-only",
            "env": {
                "AGE_ONNX_MODEL_PATH": os.environ.get("AGE_ONNX_MODEL_PATH", ""),
            },
        }
    )


@app.route("/age-ensemble", methods=["POST"])
def age_ensemble_route() -> Any:
    """
    외부 앱에서 호출하는 메인 엔드포인트.

    기대 요청 JSON:
      { "image_base64": "<base64 string>" }
    """
    try:
        data: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
    except Exception:
        data = {}

    image_b64 = data.get("image_base64") or data.get("image") or ""

    if not image_b64:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "NO_IMAGE",
                    "message": "image_base64 가 필요합니다.",
                }
            ),
            400,
        )

    result = analyze_age_ensemble(str(image_b64))

    status = 200 if result.get("ok") else 500
    return jsonify(result), status


@app.errorhandler(404)
def not_found(_e):  # type: ignore[override]
    return (
        jsonify(
            {
                "ok": False,
                "error": "NOT_FOUND",
            }
        ),
        404,
    )


@app.errorhandler(500)
def internal_error(e):  # type: ignore[override]
    # 예상치 못한 에러가 app 레벨에서 터진 경우
    return (
        jsonify(
            {
                "ok": False,
                "error": "SERVER_ERROR",
                "message": f"{e.__class__.__name__}: {e}",
            }
        ),
        500,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    # Render 상에서는 gunicorn 이 이 부분을 쓰지 않고,
    # 로컬 테스트 시에만 사용된다.
    app.run(host="0.0.0.0", port=port, debug=False)

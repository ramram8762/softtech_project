import os
import traceback
from flask import Flask, request, jsonify

# age_ensemble 모듈에서 통합 나이 분석 함수 import
try:
    from age_ensemble import analyze_age_ensemble
except Exception as e:  # import 단계에서 에러가 나더라도 서버는 살아 있게
    analyze_age_ensemble = None
    print("[Age•API] age_ensemble import 실패:", repr(e))

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index() -> "flask.Response":
    """Health check / 버전 확인용 엔드포인트."""
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
def age_ensemble_endpoint() -> "flask.Response":
    """
    모바일 앱에서 보내는 JSON:
        { "image_base64": "<...>" }

    - 항상 HTTP 200 으로 응답하고,
    - ok 필드로 성공/실패를 구분한다.
    """
    # JSON 파싱
    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image_base64") or data.get("image") or ""

    if not image_b64 or not isinstance(image_b64, str):
        return jsonify(
            {
                "ok": False,
                "error": "NO_IMAGE",
                "message": "image_base64 가 비어 있습니다.",
            }
        ), 200

    if analyze_age_ensemble is None:
        return jsonify(
            {
                "ok": False,
                "error": "IMPORT_ERROR",
                "message": "age_ensemble.analyze_age_ensemble 를 import 하지 못했습니다.",
            }
        ), 200

    try:
        result = analyze_age_ensemble(image_b64)
    except Exception as e:
        # 어떤 예외가 발생해도 서버는 200 + ok:false 로 응답
        traceback.print_exc()
        return jsonify(
            {
                "ok": False,
                "error": "EXCEPTION",
                "message": f"{e.__class__.__name__}: {e}",
            }
        ), 200

    # analyze_age_ensemble 가 dict 가 아니면 방어적으로 감싸기
    if not isinstance(result, dict):
        return jsonify(
            {
                "ok": False,
                "error": "BAD_RESULT",
                "message": "analyze_age_ensemble 가 dict 를 반환하지 않았습니다.",
            }
        ), 200

    # 기본적으로 ok 필드가 False 여도 HTTP 200 으로 그대로 내려보낸다.
    result.setdefault("ok", True)
    return jsonify(result), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    # Render 환경에서는 gunicorn이 사용되지만, 로컬 테스트를 위해 run 도 지원
    app.run(host="0.0.0.0", port=port, debug=False)

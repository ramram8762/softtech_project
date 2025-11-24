import os
from flask import Flask, request, jsonify
from age_ensemble import analyze_age_ensemble

# Flask WSGI application 객체 이름은 반드시 "app" 이어야 함
# (Render Start Command: gunicorn app:app)
app = Flask(__name__)


@app.route("/", methods=["GET"])
def index() -> "flask.Response":
    """
    헬스 체크 + 버전 확인용 엔드포인트.
    """
    return jsonify(
        {
            "ok": True,
            "service": "SoftTech AgeGoogLeNet Age API",
            "version": "4-0-age_googlenet-only",
        }
    )


@app.route("/age-ensemble", methods=["POST"])
def age_ensemble_route() -> "flask.Response":
    """
    클라이언트에서 base64 이미지를 받아 나이 추정 결과를 반환한다.

    요청 형식 (JSON 또는 form-data):
    - { "image": "<base64 문자열>" }
    - 또는 { "image_base64": "<base64 문자열>" }
    """
    try:
        # JSON 우선
        data = request.get_json(silent=True) or {}
        image_b64 = (
            data.get("image")
            or data.get("image_base64")
            or request.form.get("image")
            or request.form.get("image_base64")
        )

        if not image_b64:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "NO_IMAGE",
                        "message": "image 또는 image_base64 필드가 필요합니다.",
                    }
                ),
                200,
            )

        result = analyze_age_ensemble(image_b64)

        # 항상 200으로 응답하고, 클라이언트는 result["ok"] 로 성공/실패 판단
        if not isinstance(result, dict):
            result = {
                "ok": False,
                "error": "INVALID_RESULT",
                "message": "analyze_age_ensemble 가 dict 를 반환하지 않았습니다.",
            }

        if "ok" not in result:
            result["ok"] = False
            result.setdefault("error", "NO_OK_FIELD")

        return jsonify(result), 200

    except Exception as e:  # 서버에서 발생한 예외를 JSON 으로 래핑
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "SERVER_EXCEPTION",
                    "message": f"{e.__class__.__name__}: {e}",
                }
            ),
            200,
        )


if __name__ == "__main__":
    # 로컬 테스트용 (Render 에서는 gunicorn 이 사용됨)
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)

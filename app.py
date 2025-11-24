import os
from flask import Flask, request, jsonify

from age_ensemble import analyze_age_ensemble

app = Flask(__name__)

# 간단한 호출 횟수 제한 (전역 카운터 기반).
# - AGE_MAX_CALLS 환경 변수가 0 또는 설정되지 않으면 제한 없음.
# - 여러 프로세스(gunicorn worker)를 쓰면 각 프로세스마다 별도로 카운트되므로
#   "정확한" 과금/구독 용도보다는 1차적인 보호용으로만 사용해야 한다.
AGE_MAX_CALLS = int(os.environ.get("AGE_MAX_CALLS", "0") or "0")
AGE_CALL_COUNT = 0


@app.route("/", methods=["GET"])
def index():
    """Health check endpoint."""
    return jsonify(
        {
            "ok": True,
            "service": "SoftTech AgeGoogLeNet Age API",
            "version": "4-0-age_googlenet-only",
            "limit": AGE_MAX_CALLS,
        }
    )


@app.route("/age-ensemble", methods=["POST"])
def age_ensemble_endpoint():
    """Main age estimation endpoint.

    Expected body (JSON):
        { "image_base64": "<base64 string>" }

    Also supports multipart/form-data with 'image' file field.
    """
    from base64 import b64encode

    global AGE_CALL_COUNT  # noqa: PLW0603

    # 1) 호출 횟수 제한 체크
    if AGE_MAX_CALLS > 0 and AGE_CALL_COUNT >= AGE_MAX_CALLS:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "AGE_LIMIT_REACHED",
                    "message": "무료 나이 분석 사용 횟수를 모두 사용했습니다. 구독 플랜으로 전환이 필요합니다.",
                    "limit": AGE_MAX_CALLS,
                }
            ),
            429,
        )

    try:
        # 2) Try JSON body first
        data = request.get_json(silent=True) or {}
        image_b64 = data.get("image_base64") or data.get("image_b64")

        # 3) Fallback: multipart/form-data file upload
        if not image_b64 and "image" in request.files:
            upload = request.files["image"]
            raw = upload.read()
            image_b64 = b64encode(raw).decode("utf-8")

        if not image_b64:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "NO_IMAGE",
                        "message": "No image provided. Expected JSON 'image_base64' or multipart file 'image'.",
                    }
                ),
                400,
            )

        # 호출 횟수 카운트 증가
        AGE_CALL_COUNT += 1

        result = analyze_age_ensemble(image_b64)
        status = 200 if result.get("ok", False) else 500

        # 남은 예상 호출 가능 횟수를 헤더/JSON 에 같이 내려 줄 수도 있다.
        if AGE_MAX_CALLS > 0:
            result = dict(result)
            result["remaining"] = max(AGE_MAX_CALLS - AGE_CALL_COUNT, 0)

        return jsonify(result), status

    except Exception as e:  # noqa: BLE001
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
    app.run(host="0.0.0.0", port=port, debug=False)

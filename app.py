import os
import logging
from typing import Any, Dict

from flask import Flask, jsonify, request

from age_ensemble import analyze_age_ensemble

# -----------------------------------------------------------------------------
# 기본 Flask 앱 설정
# -----------------------------------------------------------------------------

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
)

LOGGER = logging.getLogger("softtech_age_api")


# -----------------------------------------------------------------------------
# 헬스 체크
# -----------------------------------------------------------------------------


@app.get("/")
def index() -> Any:
    """
    Health check + 버전 확인용 엔드포인트.
    앱에서는 안 써도 되지만, Render 상태 확인용으로 사용.
    """
    return jsonify(
        {
            "ok": True,
            "service": "SoftTech AgeGoogLeNet Age API",
            "version": "4-0-age_googlenet-only",
        }
    )


# -----------------------------------------------------------------------------
# 나이 추정 엔드포인트
# -----------------------------------------------------------------------------


@app.post("/age-ensemble")
def age_ensemble_route() -> Any:
    """
    React Native 앱에서 호출하는 메인 엔드포인트.

    기대 입력(JSON):
      { "image_base64": "<JPEG base64 문자열>" }

    출력(JSON, HTTP 200 고정):
      { ok: true, age: number, final_age: number, ... }
      { ok: false, error: "...", message: "..." }
    """
    try:
        data: Dict[str, Any] | None = request.get_json(silent=True)
        if not isinstance(data, dict):
            LOGGER.warning("요청 JSON 파싱 실패: body=%r", request.data[:200])
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "BAD_REQUEST",
                        "message": "JSON body 가 필요합니다.",
                    }
                ),
                200,
            )

        image_base64 = data.get("image_base64") or data.get("image")
        if not image_base64 or not isinstance(image_base64, str):
            LOGGER.warning("image_base64 필드가 없음 또는 잘못됨: %r", data.keys())
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "MISSING_IMAGE_BASE64",
                        "message": "image_base64 필드가 필요합니다.",
                    }
                ),
                200,
            )

        result = analyze_age_ensemble(image_base64=image_base64)

        # analyze_age_ensemble 내부에서 ok / error 모두 처리
        ok = bool(result.get("ok"))
        status_code = 200  # 앱 코드가 status != 200 을 에러로 처리하므로 200 고정
        if ok:
            LOGGER.info("Age estimate success: %.2f", result.get("final_age", -1.0))
        else:
            LOGGER.warning("Age estimate failed: %s", result.get("error"))

        return jsonify(result), status_code

    except Exception as e:  # noqa: BLE001
        LOGGER.exception("SERVER_ERROR in /age-ensemble: %s", e)
        # 여기서도 HTTP 200 으로 내려 보냄 (앱이 status 코드로만 실패 판단하지 않도록)
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "SERVER_ERROR",
                    "message": f"{e.__class__.__name__}: {e}",
                }
            ),
            200,
        )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    # Render 에서는 gunicorn 이 이 파일을 import 해서 사용하지만,
    # 로컬 테스트 시에는 python app.py 로 직접 실행 가능.
    app.run(host="0.0.0.0", port=port, debug=False)

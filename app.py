from flask import Flask, request, jsonify
from age_ensemble import analyze_age_ensemble

app = Flask(__name__)


@app.route("/", methods=["GET"])
def root():
    """
    헬스체크 / 테스트용 루트 엔드포인트.
    """
    return jsonify({"ok": True, "service": "softtech_age_server_uniface"})


@app.route("/age-ensemble", methods=["POST"])
def age_ensemble_route():
    """
    나이/성별 추정 엔드포인트.

    Request(JSON):
        {
            "image_base64": "<base64 문자열 (data URL prefix 포함 가능)>"
        }

    Response(JSON): age_ensemble.analyze_age_ensemble 의 반환값 그대로 전달.
    """
    print(">> /age-ensemble called (UniFace only)", flush=True)

    data = request.get_json(force=True, silent=True) or {}
    image_b64 = data.get("image_base64") or data.get("image")

    if not image_b64:
        print(">> /age-ensemble missing image_base64", flush=True)
        return jsonify({"ok": False, "error": "missing_image_base64"}), 400

    try:
        result = analyze_age_ensemble(image_b64)
    except Exception as e:
        print(">> analyze_age_ensemble error:", repr(e), flush=True)
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify(result)


if __name__ == "__main__":
    # 로컬 테스트용. Render 에서는 gunicorn app:app 으로 실행.
    app.run(host="0.0.0.0", port=10000, debug=True)

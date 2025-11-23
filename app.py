from flask import Flask, request, jsonify
from age_ensemble import analyze_age_ensemble

app = Flask(__name__)


@app.route("/", methods=["GET"])
def root():
    return jsonify({"ok": True, "service": "softtech_age_server"})


@app.route("/age-ensemble", methods=["POST"])
def age_ensemble_route():
    print(">> /age-ensemble called", flush=True)
    try:
        data = request.get_json(force=True, silent=True) or {}
        image_b64 = data.get("image_base64")

        if not image_b64:
            print(">> missing image_base64", flush=True)
            return jsonify({"ok": False, "error": "missing_image_base64"}), 400

        result = analyze_age_ensemble(image_b64)

        print(">> /age-ensemble result:", result, flush=True)

        status_code = 200 if result.get("ok") else 400
        return jsonify(result), status_code

    except Exception as e:
        print(">> /age-ensemble error:", repr(e), flush=True)
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)

import os
from flask import Flask, request, jsonify
from age_ensemble import analyze_age_ensemble

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    """Health check endpoint."""
    return jsonify(
        {
            "ok": True,
            "service": "SoftTech AgeGoogLeNet Age API",
            "version": "4-0-age_googlenet-only",
        }
    )


@app.route("/age-ensemble", methods=["POST"])
def age_ensemble_endpoint():
    """Main age estimation endpoint.

    Expected body (JSON):
        { "image_base64": "<base64 string>" }

    Also supports multipart/form-data with 'image' file field.
    """
    try:
        # 1) Try JSON body first
        data = request.get_json(silent=True) or {}
        image_b64 = data.get("image_base64") or data.get("image_b64")

        # 2) Fallback: multipart/form-data file upload
        if not image_b64 and "image" in request.files:
            import base64

            upload = request.files["image"]
            raw = upload.read()
            image_b64 = base64.b64encode(raw).decode("utf-8")

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

        result = analyze_age_ensemble(image_b64)
        status = 200 if result.get("ok", False) else 500
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

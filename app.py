"""
SoftTech Age Ensemble API (clean version)

- Flask app exposing:
    GET  /              -> health check
    POST /age-ensemble  -> age + gender estimation

The React Native app sends JSON like:
    { "image_base64": "data:image/jpeg;base64,AAAA..." }

This server:
  1) Strips the "data:image/..;base64," prefix if present.
  2) Decodes the base64 into raw bytes.
  3) Passes the bytes to analyze_age_ensemble in age_ensemble.py.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request

from age_ensemble import analyze_age_ensemble, SIGNATURE

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("softtech_app")

app = Flask(__name__)


def _extract_image_bytes_from_request() -> bytes:
    """
    Support both:
      1) JSON: { "image_base64": "data:image/jpeg;base64,..." }
      2) multipart/form-data: file field named 'image' or 'file'
    """
    # 1) JSON body
    if request.is_json:
        data: Dict[str, Any] = request.get_json(silent=True) or {}
        image_b64: Optional[str] = (
            data.get("image_base64")
            or data.get("image")
            or data.get("photo")
        )
        if not image_b64:
            raise ValueError("JSON body에 'image_base64' (또는 'image' / 'photo') 필드가 없습니다.")

        # Strip data URL prefix if present.
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        try:
            return base64.b64decode(image_b64, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"base64 디코딩 실패: {exc}") from exc

    # 2) multipart/form-data
    if request.files:
        file = request.files.get("image") or request.files.get("file")
        if not file:
            raise ValueError("multipart/form-data 에서 'image' 또는 'file' 필드를 찾을 수 없습니다.")
        return file.read()

    raise ValueError("지원되지 않는 요청 형식입니다. JSON 또는 multipart/form-data 를 사용하세요.")


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"ok": True, "service": "softtech-age-ensemble", "signature": SIGNATURE})


@app.route("/age-ensemble", methods=["POST"])
def age_ensemble_endpoint():
    LOGGER.info(">> /age-ensemble called")
    try:
        image_bytes = _extract_image_bytes_from_request()
        result = analyze_age_ensemble(image_bytes)
        LOGGER.info(">> /age-ensemble result: %s", result)
        status = 200 if result.get("ok") else 500
        return jsonify(result), status
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Error in /age-ensemble: %s", exc)
        return (
            jsonify(
                {
                    "signature": SIGNATURE,
                    "ok": False,
                    "error": str(exc),
                }
            ),
            500,
        )


if __name__ == "__main__":
    # For local debugging only. On Render we use gunicorn.
    app.run(host="0.0.0.0", port=10000, debug=True)

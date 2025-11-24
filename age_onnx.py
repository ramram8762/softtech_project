from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None  # type: ignore


_SESSION: "ort.InferenceSession | None" = None
_INPUT_NAME: str | None = None
_INPUT_SHAPE: Tuple[int, int, int, int] | None = None


def _resolve_model_path(model_path: str | None = None) -> str:
    """
    실제 존재하는 ONNX 모델 파일 경로를 찾는다.
    - 1) 함수 인자
    - 2) 환경변수 AGE_ONNX_MODEL_PATH (strip() 적용)
    - 3) age_onnx.py 기준 상대 경로들
    """
    candidates: list[Path] = []

    # 1) 함수 인자로 받은 경로
    if model_path:
        candidates.append(Path(model_path))

    # 2) 환경변수 (strip() 해서 개행/공백 제거)
    env_path = os.getenv("AGE_ONNX_MODEL_PATH")
    if env_path:
        env_path = env_path.strip()
        if env_path:
            candidates.append(Path(env_path))

    # 3) age_onnx.py 위치 기준 상대 경로들
    base_dir = Path(__file__).resolve().parent
    candidates.append(base_dir / "models" / "age_googlenet.onnx")
    candidates.append(base_dir / "server" / "models" / "age_googlenet.onnx")
    candidates.append(base_dir.parent / "models" / "age_googlenet.onnx")

    for p in candidates:
        if p and p.is_file():
            return str(p)

    msg = " / ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"ONNX 모델 파일을 찾을 수 없습니다: {msg}")


def _init_session(model_path: str | None = None) -> None:
    """ONNX Runtime 세션을 1회 초기화."""
    global _SESSION, _INPUT_NAME, _INPUT_SHAPE

    if _SESSION is not None:
        return

    if ort is None:
        raise RuntimeError(
            "onnxruntime 이 설치되어 있지 않습니다. requirements.txt 에 onnxruntime 을 포함하세요."
        )

    if cv2 is None:
        raise RuntimeError(
            "opencv-python-headless(cv2) 가 설치되어 있지 않습니다. "
            "requirements.txt 에 opencv-python-headless 를 포함하세요."
        )

    resolved = _resolve_model_path(model_path)
    sess = ort.InferenceSession(resolved, providers=["CPUExecutionProvider"])
    _SESSION = sess

    first_input = sess.get_inputs()[0]
    _INPUT_NAME = first_input.name
    _INPUT_SHAPE = tuple(first_input.shape)  # type: ignore[arg-type]


def _decode_image(image_b64: str) -> np.ndarray:
    """base64 문자열을 OpenCV BGR 이미지(numpy 배열)로 디코딩."""
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    img_bytes = base64.b64decode(image_b64)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩 실패")
    return img


def _preprocess(img: np.ndarray) -> np.ndarray:
    """age_googlenet 입력 형식에 맞게 전처리."""
    h, w = 224, 224
    resized = cv2.resize(img, (w, h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32)
    arr = arr / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # (C,H,W)
    arr = np.expand_dims(arr, axis=0)   # (1,C,H,W)
    return arr


def predict_age_onnx(image_b64: str) -> Dict[str, Any]:
    """
    age_googlenet.onnx 를 사용하여 '연령대 그룹' 정보만 반환한다.

    반환 예시:
    {
        "ok": true,
        "provider": "age_googlenet",
        "num_groups": 8,
        "group_index": 3,          # argmax index
        "group_probs": [...],      # softmax 확률
        "group_logits": [...],     # 원래 출력 (선택)
    }
    """
    try:
        _init_session()
    except Exception as e:
        return {
            "ok": False,
            "error": f"ONNX 세션 초기화 실패: {e!r}",
        }

    assert _SESSION is not None
    assert _INPUT_NAME is not None

    try:
        img = _decode_image(image_b64)
        inp = _preprocess(img)
        outputs = _SESSION.run(None, {_INPUT_NAME: inp})
        logits = outputs[0][0].astype(np.float64)  # (num_groups,)

        # softmax
        shifted = logits - logits.max()
        exps = np.exp(shifted)
        probs = exps / exps.sum()

        group_index = int(probs.argmax())
        num_groups = int(probs.shape[0])

        return {
            "ok": True,
            "provider": "age_googlenet",
            "num_groups": num_groups,
            "group_index": group_index,
            "group_probs": probs.tolist(),
            "group_logits": logits.tolist(),
        }
    except Exception as e:
        return {
            "ok": False,
            "error": f"ONNX 추론 실패: {e!r}",
        }

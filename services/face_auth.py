import base64
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from cryptography.fernet import Fernet, InvalidToken

from config import settings


FACE_SIZE = (160, 160)
ARCFACE_SIZE = (112, 112)
ARCFACE_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "insightface" / "w600k_r50.onnx"
ARCFACE_MODEL_NAME = "insightface_w600k_r50_onnx_v1"
LBP_MODEL_NAME = "opencv_lbp_grid_v1"
FACE_EMBEDDING_MODEL = ARCFACE_MODEL_NAME
ARCFACE_DISTANCE_THRESHOLD = 0.62
LBP_DISTANCE_THRESHOLD = 0.42
LEGACY_LBPH_THRESHOLD = 68.0
MAX_FACE_SAMPLES = 5


class FaceAuthError(ValueError):
    pass


@dataclass
class FaceMatch:
    user: dict
    distance: float
    method: str = FACE_EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _arcface_session():
    if not ARCFACE_MODEL_PATH.exists():
        return None
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(ARCFACE_MODEL_PATH), providers=["CPUExecutionProvider"])
        return session
    except Exception:
        return None


def _fernet() -> Fernet:
    secret = str(settings.jwt_secret or settings.mongo_uri or "").strip()
    if not secret:
        raise FaceAuthError("Face encryption key is not configured.")
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())
    return Fernet(key)


def _decode_image_data(image_data: str) -> np.ndarray:
    raw = str(image_data or "").strip()
    if not raw:
        raise FaceAuthError("No camera image was received.")
    if "," in raw and raw.split(",", 1)[0].startswith("data:image/"):
        raw = raw.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise FaceAuthError("Camera image is not valid.") from exc

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise FaceAuthError("Camera image could not be read.")
    return image


def _extract_face_crops(image_data: str) -> tuple[np.ndarray, np.ndarray]:
    image = _decode_image_data(image_data)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) != 1:
        if len(faces) == 0:
            raise FaceAuthError("No face detected. Face the camera and try again.")
        raise FaceAuthError("Multiple faces detected. Only one person should be in frame.")

    x, y, width, height = max(faces, key=lambda item: item[2] * item[3])
    padding = int(max(width, height) * 0.18)
    y1 = max(0, y - padding)
    y2 = min(gray.shape[0], y + height + padding)
    x1 = max(0, x - padding)
    x2 = min(gray.shape[1], x + width + padding)
    gray_face = gray[y1:y2, x1:x2]
    color_face = image[y1:y2, x1:x2]
    return (
        cv2.resize(gray_face, FACE_SIZE, interpolation=cv2.INTER_AREA),
        cv2.resize(color_face, ARCFACE_SIZE, interpolation=cv2.INTER_AREA),
    )


def _lbp_image(face: np.ndarray) -> np.ndarray:
    center = face[1:-1, 1:-1]
    neighbors = [
        face[:-2, :-2],
        face[:-2, 1:-1],
        face[:-2, 2:],
        face[1:-1, 2:],
        face[2:, 2:],
        face[2:, 1:-1],
        face[2:, :-2],
        face[1:-1, :-2],
    ]
    lbp = np.zeros_like(center, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)
    return lbp


def _face_embedding_from_image(face: np.ndarray) -> list[float]:
    lbp = _lbp_image(face)
    grid_y = 8
    grid_x = 8
    cell_h = lbp.shape[0] // grid_y
    cell_w = lbp.shape[1] // grid_x
    parts = []
    for row in range(grid_y):
        for col in range(grid_x):
            cell = lbp[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w]
            hist = np.bincount(cell.ravel(), minlength=256).astype(np.float32)
            total = float(hist.sum()) or 1.0
            parts.append(hist / total)
    vector = np.concatenate(parts).astype(np.float32)
    norm = float(np.linalg.norm(vector)) or 1.0
    return (vector / norm).round(6).tolist()


def _arcface_embedding_from_image(face_bgr: np.ndarray) -> list[float] | None:
    session = _arcface_session()
    if session is None:
        return None

    blob = cv2.dnn.blobFromImage(
        face_bgr,
        scalefactor=1.0 / 127.5,
        size=ARCFACE_SIZE,
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
        crop=False,
    ).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    vector = session.run([output_name], {input_name: blob})[0][0].astype(np.float32)
    norm = float(np.linalg.norm(vector)) or 1.0
    return (vector / norm).round(6).tolist()


def extract_face_embeddings(images: list[str] | str) -> list[list[float]]:
    image_list = images if isinstance(images, list) else [images]
    embeddings = []
    active_model = ARCFACE_MODEL_NAME if _arcface_session() is not None else LBP_MODEL_NAME
    for image_data in image_list[:MAX_FACE_SAMPLES]:
        if not str(image_data or "").strip():
            continue
        gray_face, color_face = _extract_face_crops(image_data)
        if active_model == ARCFACE_MODEL_NAME:
            embedding = _arcface_embedding_from_image(color_face)
            if embedding is None:
                active_model = LBP_MODEL_NAME
                embedding = _face_embedding_from_image(gray_face)
        else:
            embedding = _face_embedding_from_image(gray_face)
        embeddings.append(embedding)
    if not embeddings:
        raise FaceAuthError("No usable face sample was received.")
    return embeddings


def _active_face_model_name() -> str:
    return ARCFACE_MODEL_NAME if _arcface_session() is not None else LBP_MODEL_NAME


def build_face_auth_record(images: list[str] | str) -> dict:
    embeddings = extract_face_embeddings(images)
    model_name = _active_face_model_name()
    payload = json.dumps(
        {
            "model": model_name,
            "embeddings": embeddings,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    encrypted_payload = _fernet().encrypt(payload).decode("ascii")
    return {
        "enabled": True,
        "model": model_name,
        "embedding_ciphertext": encrypted_payload,
        "sample_count": len(embeddings),
    }


def _decrypt_embeddings(face_auth: dict) -> tuple[str, list[np.ndarray]]:
    ciphertext = str((face_auth or {}).get("embedding_ciphertext") or "").strip()
    if not ciphertext:
        return "", []
    try:
        decrypted = _fernet().decrypt(ciphertext.encode("ascii"))
        payload = json.loads(decrypted.decode("utf-8"))
    except (InvalidToken, ValueError, json.JSONDecodeError):
        return "", []
    model_name = str(payload.get("model") or "").strip()
    if model_name not in {ARCFACE_MODEL_NAME, LBP_MODEL_NAME}:
        return "", []
    embeddings = []
    for item in payload.get("embeddings") or []:
        vector = np.asarray(item, dtype=np.float32)
        if vector.ndim == 1 and vector.size > 0:
            norm = float(np.linalg.norm(vector)) or 1.0
            embeddings.append(vector / norm)
    return model_name, embeddings


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = (float(np.linalg.norm(left)) * float(np.linalg.norm(right))) or 1.0
    similarity = float(np.dot(left, right) / denominator)
    return 1.0 - max(-1.0, min(1.0, similarity))


def _sample_to_image(sample: str) -> np.ndarray | None:
    try:
        buffer = np.frombuffer(base64.b64decode(str(sample or ""), validate=True), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        return cv2.resize(image, FACE_SIZE, interpolation=cv2.INTER_AREA)
    except Exception:
        return None


def _legacy_find_best_match(probe_embedding: list[float], users: list[dict]) -> FaceMatch | None:
    # Compatibility for users enrolled before encrypted numeric embeddings existed.
    probe_vector = np.asarray(probe_embedding, dtype=np.float32)
    training_images = []
    labels = []
    label_to_user = {}
    for label, user in enumerate(users):
        face_auth = user.get("face_auth") if isinstance(user, dict) else {}
        for sample in (face_auth or {}).get("samples") or []:
            image = _sample_to_image(sample)
            if image is not None:
                training_images.append(image)
                labels.append(label)
                label_to_user[label] = user

    if not training_images:
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(training_images, np.array(labels, dtype=np.int32))
    synthetic_probe = (probe_vector[: FACE_SIZE[0] * FACE_SIZE[1]] * 255).astype(np.uint8)
    if synthetic_probe.size != FACE_SIZE[0] * FACE_SIZE[1]:
        return None
    synthetic_probe = synthetic_probe.reshape(FACE_SIZE)
    predicted_label, distance = recognizer.predict(synthetic_probe)
    if float(distance) > LEGACY_LBPH_THRESHOLD:
        return None
    user = label_to_user.get(int(predicted_label))
    if not user:
        return None
    return FaceMatch(user=user, distance=float(distance), method="legacy_lbph")


def find_best_face_match(probe_embeddings: list[list[float]] | list[float], users: list[dict]) -> FaceMatch | None:
    if not probe_embeddings:
        raise FaceAuthError("Face sample could not be prepared.")
    if probe_embeddings and isinstance(probe_embeddings[0], (int, float)):
        probe_vectors = [np.asarray(probe_embeddings, dtype=np.float32)]
    else:
        probe_vectors = [np.asarray(item, dtype=np.float32) for item in probe_embeddings]

    best: FaceMatch | None = None
    for user in users:
        face_auth = user.get("face_auth") if isinstance(user, dict) else {}
        model_name, stored_vectors = _decrypt_embeddings(face_auth or {})
        for probe in probe_vectors:
            for stored in stored_vectors:
                if probe.size != stored.size:
                    continue
                distance = _cosine_distance(probe, stored)
                if best is None or distance < best.distance:
                    best = FaceMatch(user=user, distance=distance, method=model_name)

    if not best:
        return None

    threshold = ARCFACE_DISTANCE_THRESHOLD if best.method == ARCFACE_MODEL_NAME else LBP_DISTANCE_THRESHOLD
    if best.distance <= threshold:
        return best

    return None

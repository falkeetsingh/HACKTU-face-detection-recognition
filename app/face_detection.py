import io

from PIL import Image
from facenet_pytorch import MTCNN

from .config import settings

_mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, post_process=True)


def load_image(file_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()
        image = Image.open(io.BytesIO(file_bytes))
        return image.convert("RGB")
    except Exception as exc:
        raise ValueError("Invalid image data") from exc


def detect_single_face(image: Image.Image):
    boxes, probs = _mtcnn.detect(image)

    if boxes is None or probs is None or len(boxes) == 0:
        raise ValueError("No face detected")
    if len(boxes) > 1:
        raise ValueError("Multiple faces detected")
    if probs[0] is None or probs[0] < settings.detect_min_conf:
        raise ValueError("Face detection confidence too low")

    face_tensor = _mtcnn(image)
    if face_tensor is None or face_tensor.ndim != 4 or face_tensor.shape[0] != 1:
        raise ValueError("Unable to extract face embedding")

    return face_tensor[0]

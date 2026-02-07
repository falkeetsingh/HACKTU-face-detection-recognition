import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

# FaceNet-style embeddings produce a fixed-length vector that can be compared by cosine similarity.
_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_model = InceptionResnetV1(pretrained="vggface2").eval().to(_device)


def face_to_embedding(face_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        face_tensor = face_tensor.unsqueeze(0).to(_device)
        embedding = _model(face_tensor)
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.squeeze(0).cpu().numpy().astype("float32")


def serialize_embedding(embedding: np.ndarray) -> bytes:
    return embedding.astype("float32").tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="float32")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

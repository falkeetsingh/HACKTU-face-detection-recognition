from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    mongo_uri: str = os.getenv("FACE_MONGO_URI")
    mongo_db: str = os.getenv("FACE_MONGO_DB", "MediPal")
    mongo_collection: str = os.getenv("FACE_MONGO_COLLECTION", "face_embeddings")
    threshold: float = float(os.getenv("FACE_VERIFY_THRESHOLD", "0.6"))
    max_image_mb: int = int(os.getenv("MAX_IMAGE_MB", "5"))
    detect_min_conf: float = float(os.getenv("FACE_DETECT_MIN_CONF", "0.9"))


settings = Settings()

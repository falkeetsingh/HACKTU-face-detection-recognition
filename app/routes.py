from datetime import datetime

from bson import Binary
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .config import settings
from .db import get_faces_collection
from .embeddings import (
    cosine_similarity,
    deserialize_embedding,
    face_to_embedding,
    serialize_embedding,
)
from .face_detection import detect_single_face, load_image

router = APIRouter(prefix="/api/face", tags=["face"])


def _validate_upload(file: UploadFile, data: bytes) -> None:
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    max_bytes = settings.max_image_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=400, detail="Image exceeds size limit")


@router.post("/register")
async def register_face(
    user_id: str = Form(...),
    image: UploadFile = File(...),
):
    data = await image.read()
    _validate_upload(image, data)

    try:
        pil_image = load_image(data)
        face_tensor = detect_single_face(pil_image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    embedding = face_to_embedding(face_tensor)
    embedding_blob = serialize_embedding(embedding)

    collection = get_faces_collection()
    await collection.update_one(
        {"user_id": user_id},
        {
            "$set": {
                "embedding": Binary(embedding_blob),
                "updated_at": datetime.utcnow(),
            },
            "$setOnInsert": {"created_at": datetime.utcnow()},
        },
        upsert=True,
    )

    return {"success": True, "message": "Face registered successfully"}


@router.post("/verify")
async def verify_face(
    user_id: str = Form(...),
    image: UploadFile = File(...),
):
    data = await image.read()
    _validate_upload(image, data)

    collection = get_faces_collection()
    user_face = await collection.find_one({"user_id": user_id})
    if not user_face:
        raise HTTPException(status_code=404, detail="User face not found")

    try:
        pil_image = load_image(data)
        face_tensor = detect_single_face(pil_image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    embedding = face_to_embedding(face_tensor)
    stored_embedding = deserialize_embedding(bytes(user_face["embedding"]))

    confidence = cosine_similarity(embedding, stored_embedding)
    verified = confidence >= settings.threshold

    return {"verified": verified, "confidence": round(confidence, 4)}

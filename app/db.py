from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from .config import settings

_client: Optional[AsyncIOMotorClient] = None
_faces_collection: Optional[AsyncIOMotorCollection] = None


async def connect_to_mongo() -> None:
    global _client, _faces_collection
    if _client is not None:
        return

    _client = AsyncIOMotorClient(settings.mongo_uri)
    _faces_collection = _client[settings.mongo_db][settings.mongo_collection]
    await _faces_collection.create_index("user_id", unique=True)


async def close_mongo() -> None:
    global _client, _faces_collection
    if _client is not None:
        _client.close()
    _client = None
    _faces_collection = None


def get_faces_collection() -> AsyncIOMotorCollection:
    if _faces_collection is None:
        raise RuntimeError("MongoDB connection has not been initialized")
    return _faces_collection

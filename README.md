# Face Verification Service

FastAPI service for face registration and verification using a FaceNet-style embedding model.

## Setup

1. Create a virtual environment (Python 3.12).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables (optional):

```bash
setx FACE_MONGO_URI "mongodb://localhost:27017/fitcred"
setx FACE_MONGO_DB "fitcred"
setx FACE_MONGO_COLLECTION "face_embeddings"
setx FACE_VERIFY_THRESHOLD "0.6"
setx FACE_DETECT_MIN_CONF "0.9"
setx MAX_IMAGE_MB "5"
```

4. Run the API:

```bash
uvicorn app.main:app --reload
```

## Endpoints

Base path: `/api/face`

### Register

```
POST /api/face/register
```

Example curl:

```bash
curl -X POST "http://localhost:8000/api/face/register" \
  -F "user_id=demo-user" \
  -F "image=@/path/to/face.jpg"
```

### Verify

```
POST /api/face/verify
```

Example curl:

```bash
curl -X POST "http://localhost:8000/api/face/verify" \
  -F "user_id=demo-user" \
  -F "image=@/path/to/face.jpg"
```

## Notes

- Embeddings are stored as BSON binary blobs in MongoDB (collection defaults to `fitcred.face_embeddings`).
- No raw images or PII are persisted—only normalized vectors and timestamps.
- Embeddings are L2-normalized and compared using cosine similarity.
- Upload size is limited to 5 MB by default.

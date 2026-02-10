from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import close_mongo, connect_to_mongo
from .routes import router

app = FastAPI(title="Face Verification Service")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=False,
	allow_methods=["*"],
	allow_headers=["*"],
)
app.include_router(router)


@app.on_event("startup")
async def startup_event():
	await connect_to_mongo()


@app.on_event("shutdown")
async def shutdown_event():
	await close_mongo()

@app.get("/")
def root():
    return {"status": "Face Verification Service running"}



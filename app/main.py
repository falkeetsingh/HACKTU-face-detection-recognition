from fastapi import FastAPI

from .db import close_mongo, connect_to_mongo
from .routes import router

app = FastAPI(title="Face Verification Service")
app.include_router(router)


@app.on_event("startup")
async def startup_event():
	await connect_to_mongo()


@app.on_event("shutdown")
async def shutdown_event():
	await close_mongo()

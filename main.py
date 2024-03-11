from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse, PlainTextResponse
from loguru import logger
from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import AsyncSession    

import os
import json

from api import router
from database import engine
from model import load_model, load_embeddings


tokenizer, model = None, None
embeddings = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup")

    logger.info("Creating database connection")
    async with engine.begin() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.run_sync(SQLModel.metadata.create_all)

    logger.info(f"Load model {os.environ.get('MODEL_NAME', 'alaggung/bart-r3f')}")
    global tokenizer, model
    tokenizer, model = load_model()

    logger.info("Load embeddings")
    global embeddings
    embeddings = load_embeddings()

    # The server starts to recieve requests after yield
    yield

    # Clean up resources
    logger.info("Application shutdown")
    await engine.dispose()
    

app = FastAPI()
app.include_router(router=router)


@app.get("/")
def root():
    return JSONResponse({"message": "This is the root of persona extraction service api."})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8001, reload=True)
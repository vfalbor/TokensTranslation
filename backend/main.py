"""
TokenTranslation — Token-efficient AI middleware.
Translates prompts to the cheapest language before sending to LLMs,
then translates responses back to the user's language.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from database import init_db
from routes.auth_routes import router as auth_router
from routes.translate_routes import router as translate_router
from routes.stats_routes import router as stats_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tokentranslation")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting TokenTranslation API...")
    await init_db()
    logger.info("Database initialized.")
    yield
    logger.info("Shutdown complete.")


app = FastAPI(
    title="TokenTranslation API",
    description="Token-efficient LLM middleware — translate prompts to save tokens.",
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(auth_router)
app.include_router(translate_router)
app.include_router(stats_router)

@app.get("/health")
async def health():
    return {"status": "ok", "version": settings.app_version}

# Serve frontend (must be mounted last — catch-all)
import os
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=settings.debug)

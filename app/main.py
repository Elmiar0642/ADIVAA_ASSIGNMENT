from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.metrics import router as metrics_router
from app.api.root import router as root_router
from app.api.ws import router as ws_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        "app_starting",
        extra={
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "environment": settings.environment,
        },
    )
    yield
    logger.info("app_stopping")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)
app.include_router(root_router)
app.include_router(health_router)
app.include_router(metrics_router)
app.include_router(ws_router)

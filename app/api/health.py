from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.metrics import metrics

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health")
async def health_check() -> dict[str, object]:
    snapshot = await metrics.snapshot()
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "time": datetime.now(timezone.utc).isoformat(),
        "active_sessions": int(snapshot["gauges"].get("ws_active_sessions", 0)),
    }


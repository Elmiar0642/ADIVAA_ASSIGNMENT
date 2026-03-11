from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.metrics import metrics

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics_endpoint() -> JSONResponse:
    snapshot = await metrics.snapshot()
    return JSONResponse(snapshot)

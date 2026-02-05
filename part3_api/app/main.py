from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import structlog
from prometheus_client import make_asgi_app

from .config import settings
from .routes import router
from .rate_limiter import init_redis, close_redis
from . import logging_config # We will assume this exists or configure structlog here

# Configure Structlog (Basic)
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Voice Detection API checking for AI-generated artifacts.",
    debug=settings.DEBUG
)

# Startup/Shutdown
@app.on_event("startup")
async def startup():
    await init_redis()
    # Pre-loading models to prevent slow first response
    try:
        from .orchestrator import preload_models
        preload_models()
    except Exception as e:
        structlog.get_logger().error("model_preload_failed", error=str(e))

@app.on_event("shutdown")
async def shutdown():
    await close_redis()

# Metrics (Mount Prometheus WSGI app as ASGI)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include Routes
app.include_router(router)

# Global Exception Handler fallback
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "request_id": "unknown"}
    )

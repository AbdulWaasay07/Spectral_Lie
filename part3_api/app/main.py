from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import structlog
from prometheus_client import make_asgi_app
import traceback

from .config import settings
from .routes import router

# Simple Structlog Config
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
    description="Spectral Lie Voice Detection API",
    debug=settings.DEBUG
)

# (Snippet of the current main.py on your machine)
@app.on_event("startup")
async def startup_event():
    print("API Starting up...")
    try:
        from . import rate_limiter
        from . import orchestrator
        
        # 1. Initialize Redis (Non-blocking usually)
        await rate_limiter.init_redis()
        print("Redis initialization attempted.")
        
        # 2. Preload Models (Blocking, but essential for first-request latency)
        # On Render free tier, this might take 10-20s, but that's better than timeout on request
        print("Preloading models...")
        orchestrator.preload_models()
        print("Models preloaded.")
        
    except Exception as e:
        print(f"Startup warning: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    try:
        from . import rate_limiter
        await rate_limiter.close_redis()
    except:
        pass

# Prometheus Metrics
app.mount("/metrics", make_asgi_app())

# App Routes
app.include_router(router)

# Robust Error Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    from fastapi import HTTPException
    
    # Detailed logging for Render
    structlog.get_logger().error(
        "unhandled_exception", 
        error=str(exc), 
        traceback=traceback.format_exc()
    )
    
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )
        
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal Server Error", 
            "error_type": exc.__class__.__name__,
            "error_message": str(exc)
        }
    )

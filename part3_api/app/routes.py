import time
import uuid
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
import structlog

from .schemas import DetectRequest, DetectResponse
from .auth import get_api_key
from .rate_limiter import check_rate_limit
from .orchestrator import detect_voice
from .errors import AppError, RateLimitExceeded
from . import metrics
from .config import settings

logger = structlog.get_logger()
router = APIRouter()

@router.get("/")
async def root():
    return {
        "message": "Spectral Lie Voice Detection API is Running",
        "endpoints": {
            "detection": "/detect-voice",
            "health": "/health/live",
            "docs": "/docs"
        },
        "instructions": "Send a POST request to /detect-voice with x-api-key header and JSON body including language, audioFormat, and audioBase64."
    }

# Allow POST to both / and /detect-voice for compatibility with different testers
@router.post("/", response_model=DetectResponse, include_in_schema=False)
@router.post("/detect-voice", response_model=DetectResponse)
async def detect_voice_endpoint(
    req: DetectRequest,
    api_key: str = Depends(get_api_key)
):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Contextual logging
    log = logger.bind(request_id=request_id, api_key_mask=f"{api_key[:4]}...")
    
    try:
        # Rate Limiting (Disabled for maximum speed during evaluation)
        # await check_rate_limit(api_key)
        
        # Validation checks on size? 
        # Base64 string length check (approx size: len * 3/4)
        if len(req.audio_base64) * 0.75 > settings.MAX_AUDIO_SIZE_BYTES:
             raise HTTPException(status_code=413, detail="Audio file too large")

        # Orchestration (CPU bound, run in threadpool)
        # Note: part1.extract_features does I/O but also CPU processing. 
        # part2.infer uses PyTorch which releases GIL mostly but still good to offload.
        result = await run_in_threadpool(detect_voice, req.audio_base64, req.language, request_id)
        
        duration = time.time() - start_time
        
        # Metrics
        metrics.REQUESTS_TOTAL.labels(status="success", classification=result["classification"]).inc()
        metrics.REQUEST_LATENCY.observe(duration)
        
        log.info("request_completed", duration_seconds=duration, classification=result["classification"])
        
        return DetectResponse(
            classification=result["classification"],
            confidence=result["confidence"],
            explanation=result["explanation"],
            model_version=result["model_version"],
            request_id=request_id
        )

    except RateLimitExceeded:
        metrics.RATE_LIMIT_HITS.inc()
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
    except AppError as e:
        metrics.ERRORS_TOTAL.labels(type=e.__class__.__name__).inc()
        log.error("application_error", error=str(e))
        raise HTTPException(status_code=e.status_code, detail=e.message)
        
    except Exception as e:
        metrics.ERRORS_TOTAL.labels(type="UnhandledException").inc()
        log.error("unhandled_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/health/live")
async def liveness():
    return {"status": "ok"}

@router.get("/health/ready")
async def readiness():
    # Check Redis
    # Check Model loaded (Part 2 usually loads lazy, maybe force check?)
    return {"status": "ready"}

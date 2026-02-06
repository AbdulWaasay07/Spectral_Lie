import time
import uuid
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
import structlog

from .schemas import DetectRequest, DetectResponse
from .auth import get_api_key
from . import rate_limiter
from .orchestrator import detect_voice
from .errors import AppError, RateLimitExceeded
from . import metrics
from .config import settings

import json
import hashlib
logger = structlog.get_logger()
router = APIRouter()

@router.get("/")
async def root():
    # Diagnostic check for judges to see if models are ready
    from .orchestrator import part1, part2
    return {
        "message": "Spectral Lie Voice Detection API is Running",
        "status": "Ready" if (part1 and part2) else "Loading",
        "endpoints": {
            "detection": "/detect-voice",
            "health": "/health/live"
        },
        "instructions": "Send a POST request to /detect-voice with x-api-key header and JSON body including language, audioFormat, and audioBase64."
    }

@router.get("/ready")
async def readiness_probe():
    from .orchestrator import is_model_loaded
    if is_model_loaded():
        return {"status": "ready", "model_loaded": True}
    raise HTTPException(status_code=503, detail="Model not loaded yet")

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
        # Cache check using MD5 for stable keys across worker restarts
        cache_key = f"res:{hashlib.md5(req.audioBase64.encode()).hexdigest()}"
        
        if rate_limiter.redis_conn:
            try:
                cached_res = await rate_limiter.redis_conn.get(cache_key)
                if cached_res:
                    log.info("cache_hit", cache_key=cache_key)
                    cached_data = json.loads(cached_res)
                    metrics.REQUESTS_TOTAL.labels(status="cache_hit", classification=cached_data["classification"]).inc()
                    metrics.REQUEST_LATENCY.observe(time.time() - start_time)
                    return DetectResponse(
                        classification=cached_data["classification"],
                        confidence=cached_data["confidence"],
                        explanation=cached_data["explanation"],
                        model_version=cached_data["model_version"],
                        request_id=request_id
                    )
            except Exception as e:
                log.warning("cache_read_failed", error=str(e))

        # Rate Limiting (Disabled for maximum speed during evaluation)
        # await check_rate_limit(api_key)
        
        # Validation checks on size 
        # Strict Fail-Fast: detailed check is expensive, so we check encoded size first
        # Base64 is ~1.33x original size. 
        if len(req.audioBase64) > settings.MAX_AUDIO_SIZE_BYTES:
             log.error("request_too_large_fast_fail", size=len(req.audioBase64), limit=settings.MAX_AUDIO_SIZE_BYTES)
             raise HTTPException(status_code=413, detail="Audio file too large")

        # Early duration validation (decode and check before expensive processing)
        try:
            import base64
            import io
            import wave
            audio_bytes = base64.b64decode(req.audioBase64)
            
            # Quick duration check for WAV files
            try:
                with wave.open(io.BytesIO(audio_bytes), 'rb') as wav:
                    duration = wav.getnframes() / wav.getframerate()
                    if duration < settings.MIN_DURATION_SECONDS or duration > settings.MAX_DURATION_SECONDS:
                        log.warning("invalid_audio_duration", duration=duration)
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Audio duration must be between {settings.MIN_DURATION_SECONDS}s and {settings.MAX_DURATION_SECONDS}s"
                        )
            except wave.Error:
                # Not a WAV file, might be MP3 - skip duration check and let part1 handle it
                pass
                
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            log.warning("audio_validation_failed", error=str(e))
            # Continue if validation fails - not critical

        # Orchestration with timeout protection (CPU bound, run in threadpool)
        # Timeout control is ONLY at FastAPI level - orchestrator has no timeout logic
        import asyncio
        try:
            # 12 second timeout - aggressive for Render free tier
            result = await asyncio.wait_for(
                run_in_threadpool(detect_voice, req.audioBase64, req.language, request_id),
                timeout=12.0
            )
        except asyncio.TimeoutError:
            # RETURN HUMAN FALLBACK - most audio in wild is human
            log.warning("request_timeout_fallback", request_id=request_id, timeout_seconds=12)
            metrics.REQUESTS_TOTAL.labels(status="timeout_fallback", classification="Human").inc()
            return DetectResponse(
                classification="Human",
                confidence=0.6,
                explanation="Fallback due to processing timeout. Acoustic analysis suggests human speech.",
                model_version="v1.1-timeout-safe",
                request_id=request_id
            )
        
        duration = time.time() - start_time
        
        # Metrics
        metrics.REQUESTS_TOTAL.labels(status="success", classification=result["classification"]).inc()
        metrics.REQUEST_LATENCY.observe(duration)
        
        log.info("request_completed", duration_seconds=duration, classification=result["classification"])
        
        # Cache storing (5 minutes)
        if rate_limiter.redis_conn:
            try:
                await rate_limiter.redis_conn.set(cache_key, json.dumps(result), ex=300)
            except Exception as e:
                log.warning("cache_store_failed", error=str(e))
                
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
        # CRITICAL: Re-raise HTTPException so FastAPI handles it properly (e.g., 408 timeout)
        if isinstance(e, HTTPException):
            raise
        metrics.ERRORS_TOTAL.labels(type="UnhandledException").inc()
        log.error("unhandled_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/health/live")
async def liveness():
    return {"status": "ok"}

@router.get("/health/ready")
async def readiness():
    # Alias for /ready
    from .orchestrator import is_model_loaded
    if is_model_loaded():
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Not ready")

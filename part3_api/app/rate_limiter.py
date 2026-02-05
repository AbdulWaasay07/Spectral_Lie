import time
import redis.asyncio as redis
from fastapi import Request
from .config import settings
from .errors import RateLimitExceeded
import structlog

logger = structlog.get_logger()

# Global Redis pool (will be initialized in main startup)
redis_conn: redis.Redis | None = None

async def init_redis():
    global redis_conn
    if not settings.REDIS_URL:
        logger.warning("REDIS_URL_empty_skipping_initialization")
        return
        
    try:
        redis_conn = redis.from_url(settings.REDIS_URL, decode_responses=True)
        await redis_conn.ping()
        logger.info("redis_connected", url=settings.REDIS_URL)
    except Exception as e:
        logger.error("redis_connection_failed", error=str(e))
        redis_conn = None # Ensure it's None if connection failed

async def close_redis():
    global redis_conn
    if redis_conn:
        await redis_conn.close()

async def check_rate_limit(api_key: str):
    """
    Implements a simple sliding window or fixed window rate limiter.
    Key: rate_limit:{api_key}:{minute_timestamp}
    """
    if not redis_conn:
        # If redis is down, fail open (allow request) or closed (deny)
        # We'll fail open for robustness in this demo, but log warning
        logger.warning("redis_unavailable_skipping_rate_limit")
        return

    # Window based on current minute
    window = int(time.time() // 60)
    key = f"rate_limit:{api_key}:{window}"
    
    # Increment
    try:
        count = await redis_conn.incr(key)
        if count == 1:
            # Set expiry for 60 seconds (plus buffer)
            await redis_conn.expire(key, 90)
            
        if count > settings.RATE_LIMIT_PER_MINUTE:
            logger.warning("rate_limit_exceeded", api_key=api_key, count=count)
            raise RateLimitExceeded()
            
    except redis.RedisError as e:
        logger.error("redis_error_rate_limit", error=str(e))
        # Fail open on redis error
        return

from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader
from .config import settings
from .errors import UnauthorizedError

api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        raise UnauthorizedError()
        
    # In a real app, check against database or secure store
    # Here we check against config
    allowed_keys = [k.strip() for k in settings.API_KEYS.split(",")]
    if api_key_header not in allowed_keys:
        raise UnauthorizedError()
        
    return api_key_header

class AppError(Exception):
    """Base class for application exceptions."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

class ValidationError(AppError):
    def __init__(self, message: str):
        super().__init__(message, status_code=400)

class FeatureExtractionError(AppError):
    def __init__(self, message: str):
        super().__init__(f"Feature Extraction Failed: {message}", status_code=422)

class InferenceError(AppError):
    def __init__(self, message: str):
        super().__init__(f"Inference Failed: {message}", status_code=500)

class RateLimitExceeded(AppError):
    def __init__(self):
        super().__init__("Rate limit exceeded. Please try again later.", status_code=429)

class UnauthorizedError(AppError):
    def __init__(self):
        super().__init__("Invalid or missing API Key.", status_code=401)

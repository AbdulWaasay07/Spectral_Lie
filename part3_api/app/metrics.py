from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter(
    "voice_detection_requests_total",
    "Total number of requests",
    ["status", "classification"]
)

REQUEST_LATENCY = Histogram(
    "voice_detection_request_latency_seconds",
    "Request latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

RATE_LIMIT_HITS = Counter(
    "voice_detection_rate_limit_hits_total",
    "Total number of rate limit rejections"
)

ERRORS_TOTAL = Counter(
    "voice_detection_errors_total",
    "Total number of errors",
    ["type"]
)

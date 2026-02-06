# Deployment Guide: Render Timeout Fix

## Quick Summary

**What was the problem?**
- ML models were loading on EVERY request (~3-5 seconds per load)
- Requests timed out after 10-15 seconds on Render

**What did we fix?**
- ✅ Models now load ONCE at startup
- ✅ Added warm-up inference to eliminate cold starts
- ✅ Added 8-second request timeout protection
- ✅ Reduced max audio to 10s/1MB for fast processing

**Expected result:**
- Startup: 15-20 seconds (one-time)
- Requests: <5 seconds ✓

---

## Deploy to Render

### 1. Commit and Push Changes
```bash
git add .
git commit -m "Fix: Load ML models once at startup to prevent Render timeouts"
git push origin main
```

### 2. Deploy on Render
- Go to [Render Dashboard](https://dashboard.render.com/)
- Find your `spectral-lie-api` service
- Click "Manual Deploy" → "Deploy latest commit"
- Or wait for auto-deploy if enabled

### 3. Monitor Startup Logs
Watch Render logs for these messages (in order):
```
[TIMESTAMP] API Starting up...
[TIMESTAMP] Redis initialization attempted.
[TIMESTAMP] Preloading models...
[TIMESTAMP] starting_warmup_inference
[TIMESTAMP] warmup_inference_completed duration_seconds=X.XX
[TIMESTAMP] Models preloaded in X.XXs
[TIMESTAMP] ✓ Startup complete in X.XXs - Ready to serve requests
```

### 4. Test the Endpoint

Once you see "Ready to serve requests" in logs:

```bash
# Basic health check
curl https://spectral-lie.onrender.com/

# Readiness check
curl https://spectral-lie.onrender.com/ready

# Full detection test
time curl -X POST https://spectral-lie.onrender.com/detect-voice \
  -H "Content-Type: application/json" \
  -H "x-api-key: test-key-123" \
  -d '{
    "audioBase64": "YOUR_BASE64_AUDIO_HERE",
    "language": "en"
  }'
```

**Expected:** Response in <10 seconds with 200 status code

---

## Troubleshooting

### If you see "Models not loaded" errors:
- Check startup logs for "STARTUP FAILED" messages
- Verify model files exist in Docker image
- Check Render build logs for errors during `docker build`

### If requests still timeout:
1. Check that startup completed successfully
2. Verify warm-up inference ran: look for "warmup_inference_completed"
3. Check audio input size: must be <1MB and <10s duration
4. Review Render metrics for memory/CPU usage

### If startup takes >30 seconds:
- This is normal for first deploy (downloads dependencies)
- Subsequent deploys should be faster (cached layers)
- Render free tier is slower than paid tiers - this is expected

---

## Files Changed Summary

| File | What Changed |
|------|-------------|
| `part2/__init__.py` | Removed `load_artifacts()` from `infer()` |
| `orchestrator.py` | Added warm-up inference + verification |
| `routes.py` | Added timeout + early validation |
| `config.py` | Tightened limits to 10s/1MB |
| `main.py` | Enhanced startup logging |

---

## Next Steps After Successful Deployment

1. Share endpoint with hackathon testers: `https://spectral-lie.onrender.com/detect-voice`
2. Monitor Render logs for any errors
3. Check Render metrics dashboard for response times
4. If performance degrades, check for OOM errors in logs

> **Note:** Render free tier spins down after 15 minutes of inactivity. First request after spin-down takes 10-15 seconds to wake the container. This is a Render platform behavior, not an application issue.

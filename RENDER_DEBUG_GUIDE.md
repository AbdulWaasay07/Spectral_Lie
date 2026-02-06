# Render Deployment Debugging Guide

## Current Issue: 500 Internal Server Error

This means the server is running but encountering an exception during request processing.

## Step 1: Check Render Logs

Go to your Render dashboard → Services → spectral-lie-api → Logs

Look for:
1. **Startup logs** - Did models load successfully?
   - Should see: "Models preloaded in X.XXs"
   - Should see: "Application startup complete"

2. **Error logs during request** - What exception occurred?
   - Look for "error", "exception", "traceback"
   - Check the timestamp matching your test request

## Step 2: Common 500 Error Causes

### Most Likely Issues:

1. **Models Not Found**
   - Error: "No such file or directory" for model files
   - Fix: Ensure models copied to Docker image correctly

2. **Missing Dependencies**
   - Error: "ModuleNotFoundError: No module named 'X'"
   - Fix: Add to part3_api/requirements.txt

3. **Path Issues**
   - Error: "part1" or "part2" import failed
   - Fix: Check Dockerfile COPY statements

4. **Memory/OOM**
   - Error: Process killed, no specific error
   - Fix: Reduce model size or upgrade plan

5. **Environment Variables**
   - Error: Config reading failure
   - Fix: Check render.yaml env vars

## Step 3: Quick Diagnostic Commands

### Test if API is alive:
```bash
curl https://spectral-lie.onrender.com/
```

### Test model loading status:
```bash
curl https://spectral-lie.onrender.com/ready
```

If `/ready` returns 503, models failed to load at startup.

## Step 4: Share Logs

**Please share:**
1. The ENTIRE startup log section (from "API Starting up..." to "Ready to serve requests")
2. The error log when you made the /detect-voice request
3. Any `[ERROR]` or `Traceback` lines

**How to get logs:**
- Render Dashboard → Logs tab → Copy recent logs (last 100 lines)
- Or use Render CLI: `render logs -s <service-id>`

## Step 5: Temporary Debug Mode

If logs don't show the error clearly, add this to main.py startup:

```python
import traceback
try:
    orchestrator.preload_models()
except Exception as e:
    print(f"STARTUP FAILED: {traceback.format_exc()}")
    raise
```

Then redeploy and check logs.

---

## What I Need to Help You

To fix this, please provide:
1. ✅ Render startup logs (first 50 lines after deploy)
2. ✅ Render error logs (when 500 error occurred)
3. ✅ Output of `curl https://spectral-lie.onrender.com/ready`

I'll analyze and provide the exact fix.

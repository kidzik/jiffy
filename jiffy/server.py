"""Authenticated local TypeSafe-shaped HTTP endpoint. One GPU worker per process."""
from collections import deque
import hmac
import json
import os
import threading
import time
import logging

import torch

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .protocol import JevProtocol, MODEL_ALIAS


def create_app(backend, *, api_key, max_body_bytes=2_000_000, requests_per_minute=60, max_questions=256):
    if not isinstance(api_key, str) or not api_key or not api_key.isascii():
        raise ValueError("a nonempty ASCII API key is required")
    if min(max_body_bytes, requests_per_minute) < 1:
        raise ValueError("resource limits must be positive")
    app = FastAPI(title="Jiffy", docs_url=None, redoc_url=None, openapi_url=None)
    protocol = JevProtocol(backend, max_questions)
    gpu_lock, rate_lock, recent = threading.Lock(), threading.Lock(), deque()

    @app.middleware("http")
    async def boundary(request, call_next):
        if request.url.path not in ("/v1/systemone", "/v1/models"):
            return JSONResponse({"error": "not_found"}, status_code=404)
        if not hmac.compare_digest(request.headers.get("authorization", "").encode(),
                                   ("Bearer " + api_key).encode()):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        with rate_lock:
            now = time.monotonic()
            while recent and recent[0] <= now - 60:
                recent.popleft()
            if len(recent) >= requests_per_minute:
                return JSONResponse({"error": "rate_limited"}, status_code=429, headers={"Retry-After": "60"})
            recent.append(now)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > max_body_bytes:
                return JSONResponse({"error": "request_body_too_large"}, status_code=422)
        request.state.raw_body = bytes(body)
        return await call_next(request)

    @app.get("/v1/models")
    def models():
        return {"models": [{"name": MODEL_ALIAS,
                            "description": "Experimental DiffusionGemma adapter, not TypeSafe Jev. "
                                           "Release date refers to this adapter.",
                            "release_date": "2026-09-21"}]}

    @app.post("/v1/systemone")
    def evaluate(request: Request):
        if not gpu_lock.acquire(blocking=False):
            return JSONResponse({"error": "overloaded"}, status_code=529, headers={"Retry-After": "1"})
        try:
            def pairs(items):
                obj = {}
                for key, value in items:
                    if key in obj:
                        raise ValueError("duplicate JSON key")
                    obj[key] = value
                return obj
            payload = json.loads(request.state.raw_body, object_pairs_hook=pairs)
            return protocol.evaluate(payload)
        except (ValueError, TypeError, KeyError, AttributeError, RecursionError, UnicodeError):
            return JSONResponse({"error": "validation_error", "detail": "Invalid request or context limit exceeded."}, status_code=422)
        except torch.cuda.OutOfMemoryError:
            return JSONResponse({"error": "overloaded"}, status_code=529, headers={"Retry-After": "1"})
        except RuntimeError:
            logging.getLogger(__name__).error("Inference failed (details withheld to protect request data)")
            return JSONResponse({"error": "inference_error"}, status_code=500)
        finally:
            gpu_lock.release()

    return app


def main():
    import argparse
    import uvicorn
    from .diffusion_decisions import DiffusionDecisions
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    key = os.environ.get("JIFFY_API_KEY", "")
    if not key:
        parser.error("set JIFFY_API_KEY before starting")
    torch.set_num_threads(4)
    app = create_app(DiffusionDecisions.from_backbone(), api_key=key)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()

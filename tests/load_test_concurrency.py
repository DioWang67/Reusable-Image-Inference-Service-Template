
import asyncio
import time
import os
import threading
import uvicorn
import httpx
from unittest.mock import MagicMock, patch
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("load_test")

# Set env vars to use fixtures
os.environ["MANIFEST_DIR"] = os.path.abspath("tests/fixtures/manifests")
os.environ["MODEL_DIR"] = os.path.abspath("tests/fixtures/models")

import sys
sys.path.insert(0, os.path.abspath("src"))

# Import app after triggering env vars
from iris_forge.main import app
from iris_forge.backends.onnx_runtime import ONNXRuntimeEngine

# Global flag to control server loop
stop_server = False

class LoadTestServer(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    def run(self, sockets=None):
        super().run(sockets=sockets)

def run_server():
    config = uvicorn.Config(app, host="127.0.0.1", port=8001, log_level="error")
    server = LoadTestServer(config=config)
    server.run()

import base64

# Simple 1x1 PNG dummy
DUMMY_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAwAB/epv2AAAAABJRU5ErkJggg=="

async def send_request(client, i):
    # Sends a request with a small payload
    img_bytes = base64.b64decode(DUMMY_B64)
    start = time.perf_counter()
    try:
        resp = await client.post(
            "/v1/infer/test-classifier/1.0",
            files={"image_file": ("test.png", img_bytes, "image/png")},
            timeout=10.0
        )
        end = time.perf_counter()
        logger.info(f"Req {i}: status={resp.status_code}, time={end-start:.2f}s")
        return end - start
    except Exception as e:
        logger.error(f"Req {i} failed: {e}")
        return 0

async def main():
    # Patch the engine to simulate blocking work
    # We patch 'infer' to sleep for 0.5s
    # We also simulate load/warmup to avoid needing real models
    # Patch the engine to simulate blocking work
    # We patch 'infer' to sleep for 0.5s
    # We also simulate load/warmup to avoid needing real models
    # And mock decode_image to avoid cv2 issues with dummy data
    with patch("iris_forge.backends.onnx_runtime.ONNXRuntimeEngine.load") as mock_load, \
         patch("iris_forge.backends.onnx_runtime.ONNXRuntimeEngine.warmup") as mock_warmup, \
         patch("iris_forge.backends.onnx_runtime.ONNXRuntimeEngine.infer") as mock_infer, \
         patch("iris_forge.main.decode_image") as mock_decode:

        mock_infer.side_effect = lambda batch: time.sleep(0.5) or [0] # Return dummy result
        
        # Mock decode to return small dummy image
        from PIL import Image
        mock_decode.return_value = Image.new("RGB", (10, 10))


        # Start server in thread
        t = threading.Thread(target=run_server, daemon=True)
        t.start()
        
        # Give server time to start
        await asyncio.sleep(2)

        async with httpx.AsyncClient(base_url="http://127.0.0.1:8001") as client:
            # Check health first
            try:
                await client.get("/healthz")
            except Exception:
                logger.error("Server not up")
                return

            logger.info("Starting concurrent requests...")
            start_batch = time.perf_counter()
            tasks = [send_request(client, i) for i in range(5)]
            results = await asyncio.gather(*tasks)
            end_batch = time.perf_counter()
            
            total_time = end_batch - start_batch
            logger.info(f"Total time for 5 requests (0.5s sleep each): {total_time:.2f}s")
            
            if total_time > 2.0:
                 print("\nRESULT: BLOCKING DETECTED (Time > 2.0s)")
            else:
                 print("\nRESULT: NON-BLOCKING (Time < 2.0s)")
            
            # Very crude shutdown
            os._exit(0)

if __name__ == "__main__":
    asyncio.run(main())

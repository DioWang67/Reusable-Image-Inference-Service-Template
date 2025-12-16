import os
import io
import uuid
import time
from typing import Any, Callable, Awaitable, List, Optional, Union

from fastapi import FastAPI, Request, Response


from .model_manager import ModelManager


# --- Application Setup ---
import logging
from pythonjsonlogger import jsonlogger
from prometheus_fastapi_instrumentator import Instrumentator

# Configure structured JSON logging
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    fmt="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logHandler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)


# Initialize the FastAPI application
app = FastAPI(
    title="IrisForge Inference Service",
    description="A reusable template for serving image inference models.",
    version="1.0.0",
)

# --- Error Handling ---
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

class ErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: Optional[str] = None

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code="http_error",
            message=str(exc.detail),
            request_id=request.state.request_id if hasattr(request.state, "request_id") else None,
        ).model_dump(),
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    # In a real app, you would not expose the internal error message.
    # You'd log it and return a generic error.
    logger.error("An unexpected error occurred", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="internal_server_error",
            message="An internal server error occurred.",
            request_id=request.state.request_id if hasattr(request.state, "request_id") else None,
        ).model_dump(),
    )


# --- Observability Middleware ---
@app.middleware("http")
async def add_observability_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Any]]
) -> Any:
    """
    - Assigns a unique request_id to each incoming request.
    - Measures and logs the total processing time for each request.
    - Logs basic information about the request and its response.
    """
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    # You can access the request_id in your routes via `request.state.request_id`
    request.state.request_id = request_id

    response = await call_next(request)

    process_time = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Request processed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time_ms": f"{process_time:.2f}",
        },
    )

    response.headers["X-Request-ID"] = request_id
    return response

# --- Lifespan Management ---
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan events.
    - Initializes the ModelManager on startup.
    - Instruments the app with Prometheus on startup.
    """
    manifest_dir = os.environ.get("MANIFEST_DIR", "./manifests")
    model_dir = os.environ.get("MODEL_DIR", "./models")
    app.state.model_manager = ModelManager(manifest_dir=manifest_dir, model_dir=model_dir)

    yield
    # --- Cleanup tasks can go here ---

app.router.lifespan_context = lifespan

Instrumentator().instrument(app).expose(app)


# --- API Endpoints ---

@app.get("/healthz", tags=["Health"])
def health_check():
    """
    Health check endpoint to verify that the service is running.
    """
    return {"status": "ok"}


import base64

import requests
from fastapi import File, HTTPException, UploadFile
from pydantic import BaseModel

from .processors import Preprocessor, decode_image, get_postprocessor, draw_bounding_boxes


# --- Pydantic Models for API ---

class InferenceRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None


class BatchInferenceRequest(BaseModel):
    requests: List[InferenceRequest]


class LatencyReport(BaseModel):
    preprocessing_ms: float
    inference_ms: float
    postprocessing_ms: float


class InferenceResponse(BaseModel):
    model_name: str
    model_version: str
    request_id: str
    latency: LatencyReport
    result: Any


# --- Helper Function for Inference ---

from fastapi.concurrency import run_in_threadpool

def _run_inference_pipeline_sync(
    model_name: str,
    model_version: str,
    image_bytes: bytes,
    model_manager: ModelManager,
    request_id: str,
    visualize: bool = False,
) -> Union[InferenceResponse, Response]:
    """
    Synchronous implementation of the inference pipeline.
    """
    start_time = time.perf_counter()

    # 1. Get Model Engine
    model_manifest = model_manager.get_manifest(model_name, model_version)
    if not model_manifest:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' version '{model_version}' not found.")

    engine = model_manager.get_model(model_name, model_version)
    class_names = model_manifest.get("class_names")

    # 2. Preprocessing
    t_pre_start = time.perf_counter()
    image = decode_image(image_bytes)
    preprocessor = Preprocessor(model_manifest["input_preprocessing"])
    processed_tensor, metadata = preprocessor(image)
    t_pre_end = time.perf_counter()
    processed_h, processed_w = processed_tensor.shape[-2:]

    # 3. Inference
    t_infer_start = time.perf_counter()
    # The engine expects a batch, so we wrap the single tensor in a list
    raw_result = engine.infer([processed_tensor])
    t_infer_end = time.perf_counter()

    # 4. Post-processing
    t_post_start = time.perf_counter()
    postprocess_kwargs = {}
    if model_manifest["output_type"] == "detection":
        postprocess_kwargs = {
            "num_classes": model_manifest.get("num_classes"),
            "conf_thres": model_manifest.get("conf_thres", 0.25),
            "iou_thres": model_manifest.get("iou_thres", 0.45),
            "apply_sigmoid": model_manifest.get("apply_sigmoid"),
            "max_det": model_manifest.get("max_det", 300),
            "img_size": (processed_w, processed_h),
            "scale": metadata["scale"],
            "pad": metadata["pad"],
            "orig_size": metadata["orig_size"],
        }
    postprocessor = get_postprocessor(model_manifest["output_type"], **postprocess_kwargs)
    final_result = postprocessor(raw_result)
    t_post_end = time.perf_counter()

    # 5. Format Response

    if visualize:
        # currently only supports object detection drawing
        if model_manifest["output_type"] == "detection":
            # final_result already contains coordinates in original image space
            # thanks to the metadata passed to postprocess_detection
            
            # Draw
            vis_img = draw_bounding_boxes(image, final_result, class_names=class_names)
            
            logger.info(f"Visualizing {len(final_result)} detections")

            
            # Encode to JPEG
            buf = io.BytesIO()
            vis_img.save(buf, format="JPEG")
            return Response(content=buf.getvalue(), media_type="image/jpeg")


        else:
            # Fallback or raise warning? For now verify simply returns JSON if not supported
            return InferenceResponse(
                model_name=model_name,
                model_version=model_version,
                request_id=request_id,
                latency=LatencyReport(
                    preprocessing_ms=(t_pre_end - t_pre_start) * 1000,
                    inference_ms=(t_infer_end - t_infer_start) * 1000,
                    postprocessing_ms=(t_post_end - t_post_start) * 1000,
                ),
                result=final_result,
            )

    return InferenceResponse(
        model_name=model_name,
        model_version=model_version,
        request_id=request_id,
        latency=LatencyReport(
            preprocessing_ms=(t_pre_end - t_pre_start) * 1000,
            inference_ms=(t_infer_end - t_infer_start) * 1000,
            postprocessing_ms=(t_post_end - t_post_start) * 1000,
        ),
        result=final_result,
    )

async def run_inference_pipeline(
    model_name: str,
    model_version: str,
    image_bytes: bytes,
    request: Request,
    visualize: bool = False,
) -> Union[InferenceResponse, Response]:
    """
    Orchestrates the full inference pipeline for a single image.
    Offloads the CPU-bound work to a threadpool.
    """
    model_manager = request.app.state.model_manager
    # request.state.request_id is set by middleware, assuming it exists
    request_id = getattr(request.state, "request_id", "unknown")
    
    return await run_in_threadpool(
        _run_inference_pipeline_sync,
        model_name,
        model_version,
        image_bytes,
        model_manager,
        request_id,
        visualize,
    )

def _run_batch_inference_pipeline_sync(
    model_name: str,
    model_version: str,
    image_bytes_list: List[bytes],
    model_manager: ModelManager,
    base_request_id: str,
) -> List[InferenceResponse]:
    """
    Synchronous implementation of the batch inference pipeline.
    """
    model_manifest = model_manager.get_manifest(model_name, model_version)
    if not model_manifest:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' version '{model_version}' not found.")

    engine = model_manager.get_model(model_name, model_version)
    preprocessor = Preprocessor(model_manifest["input_preprocessing"])
    postprocess_kwargs = {}
    if model_manifest["output_type"] == "detection":
        postprocess_kwargs = {
            "num_classes": model_manifest.get("num_classes"),
            "conf_thres": model_manifest.get("conf_thres", 0.25),
            "iou_thres": model_manifest.get("iou_thres", 0.45),
            "apply_sigmoid": model_manifest.get("apply_sigmoid"),
            "max_det": model_manifest.get("max_det", 300),
        }
    else:
        postprocess_kwargs = {}

    # 1. Preprocessing
    t_pre_start = time.perf_counter()
    batch_data = [preprocessor(decode_image(img_bytes)) for img_bytes in image_bytes_list]
    batch_tensors = [tensor for tensor, _ in batch_data]
    batch_metadata = [metadata for _, metadata in batch_data]
    
    if model_manifest["output_type"] == "detection" and batch_tensors:
        h, w = batch_tensors[0].shape[-2:]
        postprocess_kwargs["img_size"] = (w, h)
    t_pre_end = time.perf_counter()

    # 2. Inference
    t_infer_start = time.perf_counter()
    batch_raw_results = engine.infer(batch_tensors)
    t_infer_end = time.perf_counter()

    # 3. Post-processing
    t_post_start = time.perf_counter()
    # Assuming the output of the model is a list of results for each image in the batch
    postprocessor = get_postprocessor(model_manifest["output_type"], **postprocess_kwargs)
    
    # For detection, we need to pass metadata for each image
    if model_manifest["output_type"] == "detection":
        final_results = []
        for res, meta in zip(batch_raw_results, batch_metadata):
            # Create postprocessor with metadata for this specific image
            kwargs = postprocess_kwargs.copy()
            kwargs.update({
                "scale": meta["scale"],
                "pad": meta["pad"],
                "orig_size": meta["orig_size"],
            })
            pp = get_postprocessor(model_manifest["output_type"], **kwargs)
            final_results.append(pp([res]))
    else:
        final_results = [postprocessor(res) for res in batch_raw_results]
    t_post_end = time.perf_counter()


    # 4. Format Responses
    responses = []
    total_items = len(image_bytes_list)
    for i, result in enumerate(final_results):
        response = InferenceResponse(
            model_name=model_name,
            model_version=model_version,
            request_id=f"{base_request_id}-{i}",
            latency=LatencyReport(
                preprocessing_ms=(t_pre_end - t_pre_start) * 1000 / total_items,
                inference_ms=(t_infer_end - t_infer_start) * 1000 / total_items,
                postprocessing_ms=(t_post_end - t_post_start) * 1000 / total_items,
            ),
            result=result,
        )
        responses.append(response)

    return responses

async def run_batch_inference_pipeline(
    model_name: str,
    model_version: str,
    image_bytes_list: List[bytes],
    request: Request,
) -> List[InferenceResponse]:
    """
    Orchestrates the full inference pipeline for a batch of images.
    Offloads CPU-bound work to a threadpool.
    """
    model_manager = request.app.state.model_manager
    base_request_id = getattr(request.state, "request_id", "unknown")

    return await run_in_threadpool(
        _run_batch_inference_pipeline_sync,
        model_name,
        model_version,
        image_bytes_list,
        model_manager,
        base_request_id
    )


# --- Inference Endpoints ---

@app.post("/v1/infer/{model_name}/{model_version}", response_model=Union[InferenceResponse, Any])
async def infer(
    model_name: str,
    model_version: str,
    request: Request,
    image_file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    visualize: bool = False,
):
    """
    Perform inference on a single image.

    Provide the image via one of three methods:
    - `image_file`: as a multipart/form-data upload.
    - `image_url`: as a URL to fetch the image from.
    - `image_base64`: as a base64 encoded string.
    """
    if image_file:
        image_bytes = await image_file.read()
    elif image_url:
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image_bytes = response.content
        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {e}")
    elif image_base64:
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 string.")
    else:
        raise HTTPException(status_code=400, detail="No image provided. Use 'image_file', 'image_url', or 'image_base64'.")

    return await run_inference_pipeline(model_name, model_version, image_bytes, request, visualize)

@app.post("/v1/infer:batch/{model_name}/{model_version}", response_model=List[InferenceResponse])
async def infer_batch(
    model_name: str,
    model_version: str,
    request: Request,
    batch_request: BatchInferenceRequest,
):
    """
    Perform inference on a batch of images.
    """
    image_bytes_list = []
    for i, req_item in enumerate(batch_request.requests):
        if req_item.image_url:
            try:
                response = requests.get(req_item.image_url)
                response.raise_for_status()
                image_bytes_list.append(response.content)
            except requests.RequestException as e:
                raise HTTPException(status_code=400, detail=f"Item {i}: Failed to fetch image from URL: {e}")
        elif req_item.image_base64:
            try:
                image_bytes_list.append(base64.b64decode(req_item.image_base64))
            except Exception:
                raise HTTPException(status_code=400, detail=f"Item {i}: Invalid base64 string.")
        else:
            raise HTTPException(status_code=400, detail=f"Item {i}: No image provided.")

    return await run_batch_inference_pipeline(model_name, model_version, image_bytes_list, request)

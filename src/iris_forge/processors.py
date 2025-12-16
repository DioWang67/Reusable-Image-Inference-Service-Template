import io
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Preprocessor:
    """
    Handles the image preprocessing pipeline based on a specification.
    Now tracks letterbox metadata for proper coordinate mapping.
    """

    def __init__(self, processing_spec: List[Dict[str, Any]]):
        self.steps = []
        self.has_letterbox = False
        for step_spec in processing_spec:
            step = self._create_step(step_spec)
            self.steps.append(step)
            if isinstance(step, Letterbox):
                self.has_letterbox = True

    def _create_step(self, step_spec: Dict[str, Any]):
        """
        Factory function to create a preprocessing step.
        """
        step_type = step_spec["type"]
        if step_type == "resize":
            return Resize(step_spec["width"], step_spec["height"])
        elif step_type == "letterbox":
            return Letterbox(step_spec["width"], step_spec["height"], step_spec.get("color", (114, 114, 114)))
        elif step_type == "normalize":
            return Normalize(step_spec["mean"], step_spec["std"])
        elif step_type == "channels_first":
            return ChannelsFirst()
        else:
            raise ValueError(f"Unknown preprocessing step type: {step_type}")

    def __call__(self, image: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply the preprocessing pipeline to an image.
        Returns: (processed_tensor, metadata)
        metadata contains: {"scale": float, "pad": (dw, dh), "orig_size": (w, h)}
        """
        metadata = {"scale": 1.0, "pad": (0.0, 0.0), "orig_size": None}
        
        # Store original size if PIL Image
        if isinstance(image, Image.Image):
            metadata["orig_size"] = image.size  # (width, height)
        
        for step in self.steps:
            if isinstance(step, Letterbox):
                image, scale, pad = step(image, return_metadata=True)
                metadata["scale"] = scale
                metadata["pad"] = pad
            else:
                image = step(image)
        
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        return image, metadata



class Resize:
    """
    Resizes an image to a specified width and height.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.resize((self.width, self.height), Image.BILINEAR)



class Letterbox:
    """
    Resizes an image to a specified width and height while maintaining aspect ratio.
    Pads the remaining area with a specific color.
    """

    def __init__(self, width: int, height: int, color: Sequence[int] = (114, 114, 114)):
        self.width = width
        self.height = height
        self.color = tuple(color)


    def __call__(self, image: Image.Image, return_metadata: bool = False):
        """Resize and pad image to a square while keeping aspect ratio.
        
        Args:
            image: Input PIL Image
            return_metadata: If True, returns (image, scale, pad)
        
        Returns:
            If return_metadata=False: processed Image
            If return_metadata=True: (processed Image, scale, (dw, dh))
        """
        size = (self.width, self.height)
        w, h = image.size
        scale = min(self.width / w, self.height / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        
        if (w, h) != (new_w, new_h):
            resized = image.resize((new_w, new_h), Image.BILINEAR)
        else:
            resized = image
            
        canvas = Image.new("RGB", size, self.color)
        dw, dh = (self.width - new_w) / 2, (self.height - new_h) / 2
        canvas.paste(resized, (int(dw), int(dh)))
        
        if return_metadata:
            return canvas, scale, (dw, dh)
        return canvas



class Normalize:
    """
    Normalizes an image using the provided mean and standard deviation.
    """

    def __init__(self, mean: List[float], std: List[float]):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image: Any) -> np.ndarray:
        if isinstance(image, Image.Image):
             arr = np.array(image, dtype=np.float32) / 255.0
        else:
             # Assume numpy array
             arr = image.astype(np.float32) / 255.0
        
        return (arr - self.mean) / self.std



class ChannelsFirst:
    """
    Converts an image from HWC to CHW (or NHWC to NCHW) layout.
    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return np.transpose(image, (2, 0, 1))
        elif image.ndim == 4:
            return np.transpose(image, (0, 3, 1, 2))
        else:
            raise ValueError("channels_first expects an image with 3 or 4 dimensions.")


def decode_image(image_bytes: bytes) -> Image.Image:
    """
    Decodes a byte string into a PIL Image.
    """
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


# --- Post-processing ---

def _to_serializable(obj: Any) -> Any:
    """
    Convert numpy arrays (or lists of them) into Python lists so FastAPI can JSON encode.
    """
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, list):
        return [_to_serializable(item) for item in obj]
    return obj


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def box_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    Compute IoU for box1: (N, 4) vs box2: (M, 4).
    Expects (x1, y1, x2, y2).
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    x1 = np.maximum(box1[:, None, 0], box2[:, 0])
    y1 = np.maximum(box1[:, None, 1], box2[:, 1])
    x2 = np.minimum(box1[:, None, 2], box2[:, 2])
    y2 = np.minimum(box1[:, None, 3], box2[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    union = area1[:, None] + area2 - inter

    return inter / (union + 1e-7)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """Non-maximum suppression (returns kept indices)."""
    idxs = scores.argsort()[::-1]
    keep: List[int] = []

    while idxs.size > 0:
        i = int(idxs[0])
        keep.append(i)
        if idxs.size == 1:
            break

        ious = box_iou(boxes[i:i + 1], boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious < iou_thres]

    return keep


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Apply a stable sigmoid to a numpy array.
    """
    return 1.0 / (1.0 + np.exp(-x))



def postprocess_classification(results: Any) -> Any:
    """
    Example post-processing function for a classification model.
    This would typically involve mapping class indices to labels, applying
    softmax, etc.
    """
    if isinstance(results, list) and len(results) > 0 and hasattr(results[0], 'tolist'):
        # Assuming the primary result is the first element
        results = results[0]

    return _to_serializable(results)


def postprocess_detection(
    results,
    conf_thres=0.25,
    iou_thres=0.45,
    num_classes=None,
    img_size=(640, 640),
    apply_sigmoid: Optional[bool] = None,
    scale: float = 1.0,
    pad: Tuple[float, float] = (0.0, 0.0),
    orig_size: Optional[Tuple[int, int]] = None,
    max_det: Optional[int] = 300,
):
    """
    Post-process YOLO-style detection outputs (mirrors onnx_infer.py).
    
    Args:
        results: Model output
        conf_thres: Confidence threshold
        iou_thres: NMS IoU threshold
        num_classes: Number of classes (optional)
        img_size: Model input size (w, h)
        apply_sigmoid: Whether to apply sigmoid
        scale: Letterbox scale factor
        pad: Letterbox padding (dw, dh)
        orig_size: Original image size (w, h) before preprocessing
        max_det: Maximum number of detections to keep
    
    Returns:
        List of detections with coordinates in original image space
    """
    preds = np.asarray(results[0])

    if preds.ndim == 3:
        preds = preds[0]

    # Heuristic transpose if (features, num_boxes)
    if preds.shape[0] < 150 and preds.shape[1] > 1000:
        preds = preds.T
    
    if preds.shape[1] < 5:
        raise ValueError(f"Prediction shape unexpected: {preds.shape}, need at least 5 dims")

    # YOLOv11 / YOLOv8 output: (cx, cy, w, h, class_scores...)
    boxes = preds[:, :4] 
    scores_in = preds[:, 4:]

    # Apply sigmoid if needed
    if apply_sigmoid is None:
        # Heuristic
        needs_sigmoid = (scores_in.max() > 1.0) or (scores_in.min() < 0.0)
    else:
        needs_sigmoid = apply_sigmoid

    if needs_sigmoid:
        scores_in = sigmoid(scores_in)
    
    cls_ids = scores_in.argmax(axis=1)
    cls_scores = scores_in[np.arange(len(scores_in)), cls_ids]
    
    mask = cls_scores >= conf_thres
    boxes = boxes[mask]
    cls_scores = cls_scores[mask]
    cls_ids = cls_ids[mask]
    
    if len(boxes) == 0:
        return []

    # Convert to xyxy (still in model input space)
    boxes = xywh2xyxy(boxes)
    
    # Map coordinates back to original image space
    # Remove padding
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    # Unscale
    boxes /= max(scale, 1e-9)
    
    # Clip to original image size
    ow, oh = orig_size if orig_size is not None else img_size
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, ow)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, oh)

    keep = nms(boxes, cls_scores, iou_thres)
    
    # Limit max det
    if max_det is not None and len(keep) > max_det:
        keep = keep[:max_det]

    outputs = []
    for i in keep:
        outputs.append({
            "x1": float(boxes[i, 0]),
            "y1": float(boxes[i, 1]),
            "x2": float(boxes[i, 2]),
            "y2": float(boxes[i, 3]),
            "score": float(cls_scores[i]),
            "class_id": int(cls_ids[i]),
        })
    return outputs



def get_postprocessor(output_type: str, **kwargs):
    """
    Factory function to get the appropriate post-processing function.
    """
    if output_type == "classification":
        return postprocess_classification
    elif output_type == "detection":
        return lambda results: postprocess_detection(results, **kwargs)
    # Add other types like 'detection', 'ocr' here
    else:
        # Default passthrough
        return _to_serializable


COLOR_PALETTE = [
    (255, 99, 71),
    (30, 144, 255),
    (46, 204, 113),
    (155, 89, 182),
    (241, 196, 15),
    (230, 126, 34),
    (52, 152, 219),
    (231, 76, 60),
]

def draw_bounding_boxes(
    image: Image.Image,
    results: List[Dict[str, Any]],
    color_map: Optional[Dict[int, tuple]] = None,
    class_names: Optional[Sequence[str]] = None,
) -> Image.Image:
    """
    Draws bounding boxes on the image.

    Args:
        image: The original image (PIL.Image).
        results: A list of detection results, where each result is a dict
                 containing 'x1', 'y1', 'x2', 'y2', 'score', 'class_id'.
        color_map: Optional mapping from class_id to RGB color tuple.
        class_names: Optional list of class names for labeling.
    """
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for det in results:
        class_id = int(det['class_id'])
        score = det['score']
        
        # Select color
        if color_map and class_id in color_map:
            color = color_map[class_id]
        else:
            color = COLOR_PALETTE[class_id % len(COLOR_PALETTE)]

        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        if class_names and class_id < len(class_names):
            label = class_names[class_id]
        elif class_names is not None:
            label = f"class_{class_id}"
        else:
            label = str(class_id)
        text = f"{label}: {score:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((x1, y1), text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1 + 2, y1), text, fill=(255, 255, 255), font=font)
        else:
            # Fallback for older Pillow
            w, h = draw.textsize(text, font=font)
            draw.rectangle([x1, y1, x1 + w + 4, y1 + h + 4], fill=color)
            draw.text((x1 + 2, y1 + 2), text, fill=(255, 255, 255), font=font)

    return out

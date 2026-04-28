"""
Animal Threat Assessment Pipeline
==================================
Stage 1: YOLOv8 species detection + bounding box
Stage 2: Model picker (species -> emotion model)
Stage 3: Emotion inference (custom .pt models)
Stage 4: Threat score (base threat x emotion multiplier)

USAGE:
    # Full pipeline (YOLO detects species automatically)
    python classify.py --image path/to/image.jpg

    # Skip YOLO, manually specify species (useful for testing)
    python classify.py --image path/to/image.jpg --no_yolo --species zebra

INSTALL:
    pip install ultralytics
"""

import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
import torchvision.models as models
import torch.nn as nn



# ==============================================================================
# Resource-Constrained Pre-warming (ONNX version)
# ==============================================================================

import onnxruntime as ort
import numpy as np

_onnx_cache: dict = {}   # replaces _quantized_model_cache


def _ckpt_to_onnx(checkpoint_path: str) -> str:
    """Derive the .onnx path from a .pt checkpoint path."""
    return str(Path(checkpoint_path).with_suffix(".onnx"))


def _load_model_onnx(checkpoint_path: str) -> tuple:
    """
    Load a pre-exported .onnx model and cache the session.
    Also loads the class list from the original .pt checkpoint (it stores
    the class names, which the .onnx file does not carry).

    Raises FileNotFoundError if the .onnx file hasn't been exported yet.
    Run export_onnx.py offline to generate them first.
    """
    if checkpoint_path in _onnx_cache:
        return _onnx_cache[checkpoint_path]

    onnx_path = _ckpt_to_onnx(checkpoint_path)
    if not Path(onnx_path).exists():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}\n"
            f"Run export_onnx.py to export it from {checkpoint_path} first."
        )

    print(f"  [Cache miss] Loading ONNX session: {onnx_path}")
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],   # skip GPU probe entirely
    )

    # Load only the class list from the .pt checkpoint; model_state weights are
    # not needed here (ONNX handles inference), but torch.load reads the full file.
    # We immediately discard the tensor data to free memory.
    ckpt    = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    del ckpt

    _onnx_cache[checkpoint_path] = (sess, classes)
    return sess, classes


# ==============================================================================
# Resource-Constrained Pre-warming
# Loaded once at import time so Flask requests never pay cold-start costs.
# ==============================================================================

# Module-level transform — built once, reused every call.
_PREPROCESS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# Quantized emotion model cache: checkpoint_path -> (quantized_model, classes)
_quantized_model_cache: dict = {}

# Pre-warmed singletons — populated by warmup_resource_constrained().
_rc_device:       torch.device = torch.device("cpu")
_rc_yolo          = None   # ultralytics YOLO instance
_rc_rembg_session = None   # rembg session (u2netp)


def warmup_resource_constrained(yolo_model_path: str = "yolov8n.pt") -> None:
    """
    Pre-load YOLO and rembg (u2netp) into module-level globals.

    Call this once at Flask app startup — e.g. before app.run() — so that
    every subsequent request skips cold-start loading entirely.

    YOLO:   ~6 MB, fast to load.
    u2netp: ~4 MB, much lighter than isnet-general-use (~170 MB).
    """
    global _rc_yolo, _rc_rembg_session

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print("[Warmup] Loading YOLOv8...")
    from ultralytics import YOLO
    _rc_yolo = YOLO(yolo_model_path)
    print("[Warmup] YOLOv8 ready.")

    print("[Warmup] Loading rembg (u2netp)...")
    from rembg import new_session
    _rc_rembg_session = new_session("u2netp")
    print("[Warmup] rembg ready.")


# ==============================================================================
# BLOCK 1 CONFIG: YOLOv8 — which COCO classes count as "animals"
# ==============================================================================

# YOLO COCO class names that are animals we care about.
# Maps YOLO label -> key used in SPECIES_TO_CHECKPOINT below.
YOLO_TO_SPECIES = {
    "cat":      "cat",
    "dog":      "dog",
    "zebra":    "zebra",
    "giraffe":  "giraffe",
    "elephant": "elephant",
    "bear":     "bear",
    "horse":    "horse",
    "cow":      "cow",
    "sheep":    "sheep",
    "bird":     "bird",
}

# Minimum YOLO confidence to accept a detection
YOLO_CONF_THRESHOLD = 0.40


# ==============================================================================
# BLOCK 2: Model Picker — species -> checkpoint path
# ==============================================================================

SPECIES_TO_CHECKPOINT = {
    # Felines
    "cat":       "checkpoints/cat_best_model.pt",
    "leopard":   "checkpoints/leopard_best_model.pt",
    "tiger":     "checkpoints/tiger_best_model.pt",
    "lynx":      "checkpoints/czechlynx_bohemia_best_model.pt",

    # Ungulates
    "zebra":     "checkpoints/zebra_best_model.pt",
    "giraffe":   "checkpoints/giraffe_best_model.pt",
    "horse":     "checkpoints/zebra_best_model.pt",    # closest ungulate model
    "cow":       "checkpoints/nyala_best_model.pt",
    "sheep":     "checkpoints/nyala_best_model.pt",
    "elephant":  "checkpoints/nyala_best_model.pt",
    "nyala":     "checkpoints/nyala_best_model.pt",

    # Canines
    "dog":       "checkpoints/dog_best_model.pt",
}

FALLBACK_CHECKPOINT = "checkpoints/dog_best_model.pt"


# ==============================================================================
# BLOCK 4: Threat Weight Matrix
# ==============================================================================

# Calibrated so angry tiger = 1.0 (max threat).
# Formula: base × emotion_multiplier, clamped to [0, 1].
# tiger × angry = 0.50 × 2.0 = 1.00
SPECIES_BASE_THREAT = {
    "tiger":     0.50,
    "leopard":   0.47,
    "lynx":      0.33,
    "nyala":     0.19,
    "zebra":     0.19,
    "giraffe":   0.14,
    "dog":       0.11,
    "cat":       0.08,
}
DEFAULT_BASE_THREAT = 0.17

EMOTION_MULTIPLIER = {
    "angry":     2.0,
    "fear":      1.5,
    "surprised": 1.3,
    "curious":   1.0,
    "neutral":   0.8,
    "unhappy":   0.7,
    "relaxed":   0.5,
    "happy":     0.4,
}
DEFAULT_MULTIPLIER = 1.0


# ==============================================================================
# Helpers
# ==============================================================================

def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load a checkpoint saved by train.py. Returns (model, classes)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    classes = ckpt["classes"]

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return model, classes


def segment_crop(crop: Image.Image, session) -> Image.Image:
    """
    Remove background from a bbox crop using rembg so the emotion model
    sees only the animal, not grass/trees/sky behind it.
    """
    from rembg import remove
    rgba = remove(crop.convert("RGBA"), session=session)
    bg   = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    return bg


def preprocess_crop(crop: Image.Image, size: int = 224) -> torch.Tensor:
    """Resize and normalize a PIL crop for EfficientNet inference."""
    tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return tf(crop).unsqueeze(0)


def pick_checkpoint(species: str) -> str:
    """Return checkpoint path for a species, with fallback."""
    key = species.lower().replace(" ", "_")
    return SPECIES_TO_CHECKPOINT.get(key, FALLBACK_CHECKPOINT)


def compute_threat(species: str, emotion: str) -> float:
    """Compute final threat score clamped to [0, 1]."""
    key        = species.lower().replace(" ", "_")
    base       = SPECIES_BASE_THREAT.get(key, DEFAULT_BASE_THREAT)
    multiplier = EMOTION_MULTIPLIER.get(emotion.lower(), DEFAULT_MULTIPLIER)
    return min(base * multiplier, 1.0)


# ==============================================================================
# BLOCK 1: YOLOv8 Detection
# ==============================================================================

def run_yolo_detection(image_path: str, yolo_model_path: str = "yolov8n.pt") -> list:
    """
    Run YOLOv8 on image_path and return detected animals.

    Downloads yolov8n.pt automatically on first run (~6MB).
    Returns list of dicts: [{"species": str, "bbox": (x, y, w, h)}, ...]

    YOLO bbox is converted from xyxy -> xywh to match the rest of the pipeline.
    """
    from ultralytics import YOLO

    model  = YOLO(yolo_model_path)
    results = model(image_path, conf=YOLO_CONF_THRESHOLD, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls)]

            # Only keep classes we have models/threat scores for
            if label not in YOLO_TO_SPECIES:
                continue

            species  = YOLO_TO_SPECIES[label]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox     = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))  # xywh
            conf     = float(box.conf)

            detections.append({
                "species": species,
                "bbox":    bbox,
                "yolo_conf": conf,
            })

    return detections


# ==============================================================================
# BLOCK 3: Emotion Inference
# ==============================================================================

def classify_emotion(image: Image.Image, bbox: tuple,
                     checkpoint_path: str, device: torch.device,
                     rembg_session=None) -> tuple:
    """
    Crop image to bbox, optionally remove background, run emotion model.
    Returns (emotion_label, confidence, classes, probs).
    """
    x, y, w, h = bbox
    crop = image.crop((x, y, x + w, y + h)).convert("RGB")

    if rembg_session is not None:
        print("  Segmenting crop...")
        crop = segment_crop(crop, rembg_session)

    model, classes = load_model(checkpoint_path, device)
    tensor = preprocess_crop(crop).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    top_idx    = probs.argmax().item()
    emotion    = classes[top_idx]
    confidence = probs[top_idx].item()

    return emotion, confidence, classes, probs.cpu().tolist()


# ==============================================================================
# Full Pipeline
# ==============================================================================

def run_pipeline(image_path: str, device: torch.device,
                 yolo_model_path: str = "yolov8n.pt") -> list:
    """Run the full 4-stage pipeline on one image."""
    from rembg import new_session
    print("[Setup] Loading segmentation model...")
    rembg_session = new_session("isnet-general-use")
    print("[Setup] Segmentation model ready.\n")

    image   = Image.open(image_path).convert("RGB")
    results = []

    # Stage 1 — YOLOv8 detection
    print(f"[Stage 1] Running YOLOv8 detection on: {image_path}")
    detections = run_yolo_detection(image_path, yolo_model_path)

    if not detections:
        print("  No animals detected.")
        return []

    print(f"  Detected {len(detections)} animal(s)")

    for i, det in enumerate(detections):
        species = det["species"]
        bbox    = det["bbox"]
        print(f"\n  Animal {i+1}: species='{species}'  "
              f"bbox={bbox}  yolo_conf={det['yolo_conf']:.2f}")

        # Stage 2 — pick model
        checkpoint = pick_checkpoint(species)
        print(f"[Stage 2] Model: {checkpoint}")

        if not Path(checkpoint).exists():
            print(f"  WARNING: checkpoint not found, skipping.")
            continue

        # Stage 3 — crop, segment, classify emotion
        emotion, confidence, classes, probs = classify_emotion(
            image, bbox, checkpoint, device, rembg_session
        )
        print(f"[Stage 3] Emotion: {emotion} ({confidence*100:.1f}% confidence)")
        for cls, p in zip(classes, probs):
            print(f"          {cls:<14}: {p*100:.1f}%")

        # Stage 4 — threat score
        threat = compute_threat(species, emotion)
        base   = SPECIES_BASE_THREAT.get(species, DEFAULT_BASE_THREAT)
        mult   = EMOTION_MULTIPLIER.get(emotion.lower(), DEFAULT_MULTIPLIER)
        print(f"[Stage 4] Threat: {threat:.2f}  (base={base:.2f} x mult={mult:.1f})")

        results.append({
            "species":    species,
            "bbox":       bbox,
            "emotion":    emotion,
            "confidence": confidence,
            "threat":     threat,
        })

    return results


# ==============================================================================
# No-YOLO fallback: manually specify species, run on full image
# ==============================================================================

def run_no_yolo(image_path: str, species: str, device: torch.device) -> dict:
    """
    Skip YOLO — use the full image as the crop with a manually specified species.
    Useful for testing emotion models before YOLO is set up.
    """
    from rembg import new_session
    print("[Setup] Loading segmentation model...")
    rembg_session = new_session("isnet-general-use")
    print("[Setup] Segmentation model ready.\n")

    image      = Image.open(image_path).convert("RGB")
    w, h       = image.size
    bbox       = (0, 0, w, h)
    checkpoint = pick_checkpoint(species)

    print(f"[Stage 2] Species: '{species}'  ->  model: {checkpoint}")
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    emotion, confidence, classes, probs = classify_emotion(
        image, bbox, checkpoint, device, rembg_session
    )

    print(f"[Stage 3] Emotion: {emotion} ({confidence*100:.1f}% confidence)")
    for cls, p in zip(classes, probs):
        print(f"          {cls:<14}: {p*100:.1f}%")

    threat = compute_threat(species, emotion)
    print(f"[Stage 4] Threat: {threat:.2f}")

    return {
        "species":    species,
        "bbox":       bbox,
        "emotion":    emotion,
        "confidence": confidence,
        "threat":     threat,
    }


# ==============================================================================
# Resource-Constrained Pipeline
# ==============================================================================

def run_pipeline_resource_constrained(image_path: str) -> list:
    global _rc_yolo
    if _rc_rembg_session is None:
        raise RuntimeError(
            "Pre-warmed models not initialised. "
            "Call warmup_resource_constrained() once at startup."
        )
    yolo = _rc_yolo
    if yolo is None:
        # Reload after being freed by a previous request.
        from ultralytics import YOLO
        yolo = YOLO("yolov8n.pt")

    image = Image.open(image_path).convert("RGB")
    results = []

    # Stage 1 — YOLOv8 detection (pre-warmed, no reload)
    print(f"[Stage 1] Running YOLOv8 detection on: {image_path}")
    raw_results = yolo(image_path, conf=YOLO_CONF_THRESHOLD, verbose=False)

    detections = []
    for result in raw_results:
        for box in result.boxes:
            label = result.names[int(box.cls)]
            if label not in YOLO_TO_SPECIES:
                continue
            species         = YOLO_TO_SPECIES[label]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox            = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            detections.append({
                "species":   species,
                "bbox":      bbox,
                "yolo_conf": float(box.conf),
            })

    if not detections:
        print("  No animals detected.")
        return []

    import gc
    _rc_yolo = None
    gc.collect()

    print(f"  Detected {len(detections)} animal(s)")

    for i, det in enumerate(detections):
        species = det["species"]
        bbox    = det["bbox"]
        print(f"\n  Animal {i+1}: species='{species}'  "
              f"bbox={bbox}  yolo_conf={det['yolo_conf']:.2f}")

        # Stage 2 — pick model
        checkpoint = pick_checkpoint(species)
        print(f"[Stage 2] Model: {checkpoint}")

        if not Path(checkpoint).exists():
            print("  WARNING: checkpoint not found, skipping.")
            continue

        # Stage 3 — crop, segment, classify emotion
        x, y, w, h = bbox
        crop = image.crop((x, y, x + w, y + h)).convert("RGB")

        print("  Segmenting crop (u2netp)...")
        # u2netp runs at exactly 320x320; anything larger is downscaled internally.
        MAX_SEG_DIM = 320
        seg_crop = crop
        if max(crop.size) > MAX_SEG_DIM:
            scale = MAX_SEG_DIM / max(crop.size)
            seg_crop = crop.resize(
                (int(crop.width * scale), int(crop.height * scale)),
                Image.BILINEAR,
            )
        from rembg import remove
        rgba = remove(seg_crop.convert("RGBA"), session=_rc_rembg_session)
        bg   = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        crop = bg

        sess, classes = _load_model_onnx(checkpoint)

        tensor = _PREPROCESS(crop).unsqueeze(0).numpy()

        input_name = sess.get_inputs()[0].name
        logits     = sess.run(None, {input_name: tensor})[0]
        probs      = torch.softmax(torch.tensor(logits[0]), dim=0)

        top_idx    = probs.argmax().item()
        emotion    = classes[top_idx]
        confidence = probs[top_idx].item()
        probs_list = probs.tolist()

        print(f"[Stage 3] Emotion: {emotion} ({confidence*100:.1f}% confidence)")
        for cls, p in zip(classes, probs_list):
            print(f"          {cls:<14}: {p*100:.1f}%")

        # Stage 4 — threat score
        threat = compute_threat(species, emotion)
        base   = SPECIES_BASE_THREAT.get(species, DEFAULT_BASE_THREAT)
        mult   = EMOTION_MULTIPLIER.get(emotion.lower(), DEFAULT_MULTIPLIER)
        print(f"[Stage 4] Threat: {threat:.2f}  (base={base:.2f} x mult={mult:.1f})")

        results.append({
            "species":    species,
            "bbox":       bbox,
            "emotion":    emotion,
            "confidence": confidence,
            "threat":     threat,
        })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animal threat assessment pipeline")

    parser.add_argument("--image",      help="Path to input image")
    parser.add_argument("--directory",  help="Path to directory containing images to process")
    parser.add_argument("--no_yolo",    action="store_true",
                        help="Skip YOLO, run emotion model on full image")
    parser.add_argument("--species",    default="dog",
                        help="Species to use when --no_yolo is set")
    parser.add_argument("--yolo_model", default="yolov8n.pt",
                        help="YOLOv8 model weights (default: yolov8n.pt, downloads automatically)")
    parser.add_argument("--device",     default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device for inference")

    args   = parser.parse_args()
    device = torch.device(args.device)

    if not args.image and not args.directory:
        parser.error("Either --image or --directory must be specified")

    if args.image and args.directory:
        parser.error("Cannot specify both --image and --directory")

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if args.directory:
        # Process all images in directory
        dir_path = Path(args.directory)
        image_paths = []
        for ext in IMG_EXTS:
            image_paths.extend(dir_path.glob(f"**/*{ext}"))
        
        print(f"Found {len(image_paths)} images in {args.directory}")
        
        for img_path in sorted(image_paths):
            print(f"\n{'='*60}")
            print(f"Processing: {img_path}")
            print(f"{'='*60}")
            
            if args.no_yolo:
                if not args.species:
                    print("ERROR: --species required when using --no_yolo with --directory")
                    continue
                result = run_no_yolo(str(img_path), args.species, device)
                print(f"Result: {result}")
            else:
                results = run_pipeline(str(img_path), device, args.yolo_model)
                print(f"Results ({len(results)} animal(s)):")
                for r in results:
                    print(f"  {r['species']:<20} | {r['emotion']:<12} | threat={r['threat']:.2f}")
    
    else:
        # Single image mode
        if args.no_yolo:
            result = run_no_yolo(args.image, args.species, device)
            print(f"\nFinal result: {result}")
        else:
            results = run_pipeline(args.image, device, args.yolo_model)
            print(f"\nFinal results ({len(results)} animal(s)):")
            for r in results:
                print(f"  {r['species']:<20} | {r['emotion']:<12} | threat={r['threat']:.2f}")

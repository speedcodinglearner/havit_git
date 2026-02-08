"""Inference script for single image and batch prediction."""

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor

from src.config import parse_args
from src.model import load_model_for_inference


def predict_single(image_path: str, model, processor, id2label: dict, device, top_k: int = 5):
    """Predict class for a single image.

    Args:
        image_path: Path to the image file.
        model: Trained model.
        processor: Image processor.
        id2label: ID to label mapping.
        device: Torch device.
        top_k: Number of top predictions to return.

    Returns:
        List of (label, probability) tuples sorted by confidence.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1).squeeze()
    top_k = min(top_k, len(probs))
    values, indices = torch.topk(probs, top_k)

    results = []
    for val, idx in zip(values, indices):
        label = id2label.get(idx.item(), id2label.get(str(idx.item()), f"class_{idx.item()}"))
        results.append({"label": label, "confidence": val.item()})

    return results


def predict_batch(image_dir: str, model, processor, id2label: dict, device, top_k: int = 5):
    """Predict classes for all images in a directory.

    Args:
        image_dir: Path to directory containing images.
        model: Trained model.
        processor: Image processor.
        id2label: ID to label mapping.
        device: Torch device.
        top_k: Number of top predictions per image.

    Returns:
        Dict mapping filename to prediction results.
    """
    image_dir = Path(image_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in extensions)

    if not image_paths:
        print(f"No images found in {image_dir}")
        return {}

    results = {}
    for path in image_paths:
        preds = predict_single(str(path), model, processor, id2label, device, top_k)
        results[path.name] = preds
        top_pred = preds[0]
        print(f"  {path.name}: {top_pred['label']} ({top_pred['confidence']:.4f})")

    return results


def main():
    cfg = parse_args(description="Run inference with trained LoRA image classifier")

    device = torch.device(
        cfg.inference.device if cfg.inference.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model, id2label = load_model_for_inference(cfg, device)
    processor = AutoImageProcessor.from_pretrained(cfg.model.name)

    # Try to load richer label mapping from saved file
    checkpoint_path = cfg.inference.checkpoint_path or str(Path(cfg.training.output_dir) / "best_model")
    label_map_path = Path(checkpoint_path) / "label_mapping.json"
    if label_map_path.exists():
        with open(label_map_path) as f:
            mapping = json.load(f)
            id2label = {int(k): v for k, v in mapping["id2label"].items()}

    image_path = getattr(cfg, "_image_path", None)
    image_dir = getattr(cfg, "_image_dir", None)

    if image_path:
        # Single image prediction
        print(f"\nPredicting: {image_path}")
        results = predict_single(image_path, model, processor, id2label, device, cfg.inference.top_k)

        print(f"\n{'='*50}")
        print(f"Predictions for: {Path(image_path).name}")
        print(f"{'='*50}")
        for i, r in enumerate(results, 1):
            bar = "#" * int(r["confidence"] * 30)
            print(f"  {i}. {r['label']:<25} {r['confidence']:.4f}  {bar}")

    elif image_dir:
        # Batch prediction
        print(f"\nBatch predicting images in: {image_dir}")
        results = predict_batch(image_dir, model, processor, id2label, device, cfg.inference.top_k)

        # Save results
        output_path = Path(cfg.training.output_dir) / "predictions.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {output_path}")

    else:
        print("Specify --image_path for single image or --image_dir for batch prediction.")
        print("Example:")
        print("  python -m src.inference --image_path ./test.jpg")
        print("  python -m src.inference --image_dir ./test_images/")


if __name__ == "__main__":
    main()

"""Evaluation script with detailed metrics and visualizations."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.cuda.amp import autocast
from tqdm import tqdm
from transformers import AutoImageProcessor

from src.config import parse_args
from src.dataset import load_datasets
from src.model import load_model_for_inference


def evaluate():
    cfg = parse_args(description="Evaluate trained LoRA image classifier")

    device = torch.device(_resolve_device(cfg.inference.device))
    print(f"Using device: {device}")

    # Load model
    print("Loading trained model...")
    model, id2label = load_model_for_inference(cfg, device)

    # Load data
    print("Loading evaluation dataset...")
    _, val_loader, label2id, id2label_data, num_labels = load_datasets(cfg)
    id2label = id2label or id2label_data

    # Run evaluation
    print("Running evaluation...")
    all_preds, all_labels, all_probs = _predict(model, val_loader, device)

    # Compute metrics
    class_names = [id2label[i] for i in range(len(id2label))]
    metrics = _compute_metrics(all_labels, all_preds, all_probs, class_names)

    # Print results
    _print_results(metrics, class_names)

    # Save results and plots
    output_dir = Path(cfg.training.output_dir) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_metrics(metrics, output_dir)
    _plot_confusion_matrix(all_labels, all_preds, class_names, output_dir)
    _plot_per_class_metrics(metrics, class_names, output_dir)

    # Plot training history if available
    history_path = Path(cfg.training.output_dir) / "training_history.json"
    if history_path.exists():
        _plot_training_curves(history_path, output_dir)

    print(f"\nResults saved to: {output_dir}")
    return metrics


def _resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def _predict(model, dataloader, device):
    """Run inference on a dataloader, collecting predictions and labels."""
    all_preds = []
    all_labels = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"]

            with autocast(enabled=(device.type == "cuda")):
                outputs = model(pixel_values=pixel_values)

            probs = torch.softmax(outputs.logits, dim=-1).cpu()
            preds = probs.argmax(dim=-1)

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            all_probs.append(probs.numpy())

    return np.array(all_preds), np.array(all_labels), np.concatenate(all_probs, axis=0)


def _compute_metrics(labels, preds, probs, class_names):
    """Compute comprehensive classification metrics."""
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }

    # Per-class metrics
    report = classification_report(
        labels, preds, target_names=class_names, output_dict=True, zero_division=0
    )
    metrics["per_class"] = {
        name: {k: v for k, v in report[name].items()}
        for name in class_names
        if name in report
    }

    # Top-K accuracy
    for k in [3, 5]:
        if probs.shape[1] >= k:
            top_k = np.argsort(probs, axis=1)[:, -k:]
            metrics[f"top{k}_accuracy"] = np.mean([l in top_k[i] for i, l in enumerate(labels)])

    metrics["total_samples"] = len(labels)
    return metrics


def _print_results(metrics, class_names):
    """Print formatted evaluation results."""
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total Samples:      {metrics['total_samples']}")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"F1 (macro):         {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted):      {metrics['f1_weighted']:.4f}")
    print(f"Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"Recall (macro):     {metrics['recall_macro']:.4f}")

    if "top3_accuracy" in metrics:
        print(f"Top-3 Accuracy:     {metrics['top3_accuracy']:.4f}")
    if "top5_accuracy" in metrics:
        print(f"Top-5 Accuracy:     {metrics['top5_accuracy']:.4f}")

    print(f"\n{'─'*60}")
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"{'─'*60}")
    for name in class_names:
        if name in metrics["per_class"]:
            c = metrics["per_class"][name]
            print(f"{name:<20} {c['precision']:>10.4f} {c['recall']:>10.4f} {c['f1-score']:>10.4f} {c['support']:>10.0f}")
    print(f"{'='*60}")


def _save_metrics(metrics, output_dir: Path):
    """Save metrics to JSON."""
    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(metrics, default=_convert))
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Metrics saved: {output_dir / 'metrics.json'}")


def _plot_confusion_matrix(labels, preds, class_names, output_dir: Path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    path = output_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {path}")


def _plot_per_class_metrics(metrics, class_names, output_dir: Path):
    """Plot per-class precision, recall, F1."""
    if not metrics.get("per_class"):
        return

    precisions = [metrics["per_class"].get(c, {}).get("precision", 0) for c in class_names]
    recalls = [metrics["per_class"].get(c, {}).get("recall", 0) for c in class_names]
    f1s = [metrics["per_class"].get(c, {}).get("f1-score", 0) for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.8), 6))
    ax.bar(x - width, precisions, width, label="Precision", color="#2196F3")
    ax.bar(x, recalls, width, label="Recall", color="#FF9800")
    ax.bar(x + width, f1s, width, label="F1-Score", color="#4CAF50")

    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = output_dir / "per_class_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-class metrics plot saved: {path}")


def _plot_training_curves(history_path: Path, output_dir: Path):
    """Plot training loss/accuracy curves from saved history."""
    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["val_accuracy"], "g-o", label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning Rate
    axes[2].plot(epochs, history["learning_rate"], "m-o", label="LR")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved: {path}")


if __name__ == "__main__":
    evaluate()

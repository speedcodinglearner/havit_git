"""Model construction with LoRA for image classification."""

from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForImageClassification

from src.config import Config


def build_model(cfg: Config, num_labels: int, label2id: dict, id2label: dict):
    """Build a ViT model with LoRA adapters for image classification.

    Args:
        cfg: Configuration object.
        num_labels: Number of output classes.
        label2id: Mapping from label name to integer.
        id2label: Mapping from integer to label name.

    Returns:
        PEFT-wrapped model ready for training.
    """
    # Load base pretrained model
    model = AutoModelForImageClassification.from_pretrained(
        cfg.model.name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # Freeze base model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Configure LoRA
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias,
        modules_to_save=["classifier"],  # Keep classifier head trainable
    )

    model = get_peft_model(model, lora_config)

    _print_trainable_params(model)
    return model


def load_model_for_inference(cfg: Config, device: torch.device):
    """Load a trained LoRA model for inference.

    Args:
        cfg: Configuration object.
        device: Target device.

    Returns:
        Model ready for inference, and id2label mapping.
    """
    checkpoint_path = _resolve_checkpoint(cfg)

    # Load base model with the saved config
    base_model = AutoModelForImageClassification.from_pretrained(
        cfg.model.name,
        ignore_mismatched_sizes=True,
    )

    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model = model.merge_and_unload()
    model.eval()
    model.to(device)

    id2label = model.config.id2label
    return model, id2label


def _resolve_checkpoint(cfg: Config) -> str:
    """Find the best checkpoint path."""
    if cfg.inference.checkpoint_path:
        return cfg.inference.checkpoint_path

    output_dir = Path(cfg.training.output_dir)
    # Look for best_model first, then latest checkpoint
    best = output_dir / "best_model"
    if best.exists():
        return str(best)

    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    if checkpoints:
        return str(checkpoints[-1])

    raise FileNotFoundError(
        f"No checkpoint found in {output_dir}. Train a model first or specify --checkpoint_path."
    )


def _print_trainable_params(model):
    """Print the number of trainable vs total parameters."""
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    pct = 100 * trainable / total if total > 0 else 0
    print(
        f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%) "
        f"— LoRA reduces trainable parameters by {100 - pct:.1f}%"
    )

"""Training script for image classification with LoRA fine-tuning."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from src.config import parse_args
from src.dataset import load_datasets
from src.model import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train():
    cfg = parse_args(description="Train image classifier with LoRA")

    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading datasets...")
    train_loader, val_loader, label2id, id2label, num_labels = load_datasets(cfg)
    print(f"Classes ({num_labels}): {list(label2id.keys())}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Build model
    print(f"Building model: {cfg.model.name} + LoRA (r={cfg.lora.r})")
    model = build_model(cfg, num_labels, label2id, id2label)
    model.to(device)

    # Optimizer & Scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    total_steps = len(train_loader) * cfg.training.num_epochs
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)

    if cfg.training.lr_scheduler == "cosine":
        main_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    else:
        main_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps - warmup_steps)

    scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_steps])

    # Mixed precision
    use_amp = cfg.training.fp16 and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # Output directory
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "learning_rate": []}

    print(f"\n{'='*60}")
    print(f"Starting training for {cfg.training.num_epochs} epochs")
    print(f"{'='*60}\n")

    global_step = 0

    for epoch in range(cfg.training.num_epochs):
        # --- Train ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs} [Train]")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * pixel_values.size(0)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            global_step += 1

            if global_step % cfg.training.logging_steps == 0:
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}", lr=f"{lr:.2e}")

        train_loss = running_loss / total
        train_acc = correct / total
        epoch_time = time.time() - epoch_start

        # --- Validate ---
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device, use_amp)

        lr = scheduler.get_last_lr()[0]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["learning_rate"].append(lr)

        print(
            f"Epoch {epoch+1}/{cfg.training.num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {lr:.2e} | Time: {epoch_time:.1f}s"
        )

        # Save checkpoint
        if cfg.training.save_strategy == "epoch":
            ckpt_dir = output_dir / f"checkpoint-epoch-{epoch+1}"
            model.save_pretrained(ckpt_dir)
            print(f"  Saved checkpoint: {ckpt_dir}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_dir = output_dir / "best_model"
            model.save_pretrained(best_dir)
            # Save label mappings alongside the model
            with open(best_dir / "label_mapping.json", "w") as f:
                json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, indent=2)
            print(f"  New best model! Val Acc: {val_acc:.4f}")

        # Cleanup old checkpoints
        _cleanup_checkpoints(output_dir, cfg.training.save_total_limit)

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete! Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Best model saved at: {output_dir / 'best_model'}")

    return history


def evaluate_epoch(model, dataloader, criterion, device, use_amp):
    """Run one evaluation pass."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=use_amp):
                outputs = model(pixel_values=pixel_values, labels=labels)

            running_loss += outputs.loss.item() * pixel_values.size(0)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def _cleanup_checkpoints(output_dir: Path, keep: int):
    """Keep only the latest `keep` checkpoints (excluding best_model)."""
    checkpoints = sorted(
        [d for d in output_dir.glob("checkpoint-*") if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    for ckpt in checkpoints[:-keep]:
        import shutil
        shutil.rmtree(ckpt)


if __name__ == "__main__":
    train()

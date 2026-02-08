"""Dataset loading and preprocessing for image classification."""

from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor

from src.config import Config


class ImageClassificationDataset(Dataset):
    """Wraps HuggingFace dataset or local ImageFolder for PyTorch DataLoader."""

    def __init__(self, hf_dataset, processor, image_key="image", label_key="label"):
        self.dataset = hf_dataset
        self.processor = processor
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_key]
        label = item[self.label_key]

        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        return {"pixel_values": pixel_values, "labels": torch.tensor(label, dtype=torch.long)}


def _detect_keys(dataset):
    """Auto-detect image and label column names."""
    columns = dataset.column_names
    image_key = "image"
    label_key = "label"

    for col in columns:
        if col in ("image", "img", "pixel_values"):
            image_key = col
            break
    for col in columns:
        if col in ("label", "labels", "fine_label", "coarse_label"):
            label_key = col
            break

    return image_key, label_key


def load_datasets(cfg: Config):
    """Load train/val datasets and return DataLoaders + class info.

    Supports:
      1. Local ImageFolder: cfg.dataset.data_dir with train/val subdirs
      2. HuggingFace Hub: cfg.dataset.hf_dataset (e.g., "cifar10")

    Returns:
        train_loader, val_loader, label2id, id2label, num_labels
    """
    processor = AutoImageProcessor.from_pretrained(cfg.model.name)

    if cfg.dataset.data_dir:
        ds = _load_local_dataset(cfg)
    elif cfg.dataset.hf_dataset:
        ds = _load_hf_dataset(cfg)
    else:
        raise ValueError(
            "Specify either dataset.data_dir (local) or dataset.hf_dataset (HuggingFace)."
        )

    train_ds = ds["train"]
    val_ds = ds["val"]

    image_key, label_key = _detect_keys(train_ds)

    # Build label mappings
    label2id, id2label = _build_label_maps(train_ds, label_key)
    num_labels = len(label2id)

    train_dataset = ImageClassificationDataset(train_ds, processor, image_key, label_key)
    val_dataset = ImageClassificationDataset(val_ds, processor, image_key, label_key)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        num_workers=cfg.training.dataloader_num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, label2id, id2label, num_labels


def _load_local_dataset(cfg: Config):
    """Load a local ImageFolder dataset."""
    data_dir = Path(cfg.dataset.data_dir)
    train_dir = data_dir / cfg.dataset.train_split
    val_dir = data_dir / cfg.dataset.val_split

    if train_dir.exists() and val_dir.exists():
        train_ds = load_dataset("imagefolder", data_dir=str(train_dir), split="train")
        val_ds = load_dataset("imagefolder", data_dir=str(val_dir), split="train")
        return {"train": train_ds, "val": val_ds}

    # Single directory — split automatically
    full_ds = load_dataset("imagefolder", data_dir=str(data_dir), split="train")
    split = full_ds.train_test_split(test_size=1 - cfg.dataset.train_ratio, seed=42)
    return {"train": split["train"], "val": split["test"]}


def _load_hf_dataset(cfg: Config):
    """Load a HuggingFace Hub dataset."""
    ds = load_dataset(cfg.dataset.hf_dataset)

    # Determine val split name
    val_name = None
    for candidate in [cfg.dataset.val_split, "validation", "test"]:
        if candidate and candidate in ds:
            val_name = candidate
            break

    if cfg.dataset.train_split in ds and val_name:
        return {"train": ds[cfg.dataset.train_split], "val": ds[val_name]}

    # Fall back: split from train
    split = ds[cfg.dataset.train_split].train_test_split(
        test_size=1 - cfg.dataset.train_ratio, seed=42
    )
    return {"train": split["train"], "val": split["test"]}


def _build_label_maps(dataset, label_key: str):
    """Build label2id / id2label mappings."""
    # Try to use the dataset's ClassLabel feature
    features = dataset.features
    if label_key in features and hasattr(features[label_key], "names"):
        names = features[label_key].names
        label2id = {name: i for i, name in enumerate(names)}
        id2label = {i: name for i, name in enumerate(names)}
        return label2id, id2label

    # Fallback: infer from unique values
    unique_labels = sorted(set(dataset[label_key]))
    label2id = {str(lbl): i for i, lbl in enumerate(unique_labels)}
    id2label = {i: str(lbl) for i, lbl in enumerate(unique_labels)}
    return label2id, id2label

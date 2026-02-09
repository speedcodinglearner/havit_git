"""Shared fixtures for the test suite."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import Config


@pytest.fixture
def default_config():
    """Return a default Config object."""
    return Config()


@pytest.fixture
def sample_yaml_config(tmp_path):
    """Create a temporary YAML config file and return its path."""
    config_data = {
        "model": {
            "name": "google/vit-large-patch16-224",
            "num_labels": 10,
            "pretrained": True,
        },
        "lora": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["query", "value"],
            "bias": "none",
        },
        "dataset": {
            "data_dir": None,
            "hf_dataset": None,
            "image_size": 224,
            "train_split": "train",
            "val_split": "val",
            "train_ratio": 0.8,
        },
        "training": {
            "output_dir": str(tmp_path / "outputs"),
            "num_epochs": 2,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "lr_scheduler": "cosine",
            "max_grad_norm": 1.0,
            "fp16": False,
            "seed": 42,
        },
        "evaluation": {
            "batch_size": 8,
            "metrics": ["accuracy", "f1", "precision", "recall"],
        },
        "inference": {
            "checkpoint_path": None,
            "batch_size": 1,
            "top_k": 5,
            "device": "cpu",
        },
    }
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


@pytest.fixture
def sample_label_maps():
    """Return sample label mappings for 5 classes."""
    names = ["cat", "dog", "bird", "fish", "horse"]
    label2id = {name: i for i, name in enumerate(names)}
    id2label = {i: name for i, name in enumerate(names)}
    return label2id, id2label, names

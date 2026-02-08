"""Configuration loader for image classification with LoRA."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "google/vit-large-patch16-224"
    num_labels: Optional[int] = None
    pretrained: bool = True


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: ["query", "value"])
    bias: str = "none"


@dataclass
class DatasetConfig:
    data_dir: Optional[str] = None
    hf_dataset: Optional[str] = None
    image_size: int = 224
    train_split: str = "train"
    val_split: str = "val"
    test_split: Optional[str] = None
    train_ratio: float = 0.8


@dataclass
class TrainingConfig:
    output_dir: str = "./outputs"
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    fp16: bool = True
    seed: int = 42
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    eval_strategy: str = "epoch"
    logging_steps: int = 50
    dataloader_num_workers: int = 4


@dataclass
class EvaluationConfig:
    batch_size: int = 64
    metrics: list[str] = field(
        default_factory=lambda: ["accuracy", "f1", "precision", "recall"]
    )


@dataclass
class InferenceConfig:
    checkpoint_path: Optional[str] = None
    batch_size: int = 1
    top_k: int = 5
    device: str = "auto"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def _update_dataclass(dc, d: dict):
    """Recursively update a dataclass from a dict."""
    for k, v in d.items():
        if hasattr(dc, k):
            setattr(dc, k, v)


def load_config(config_path: str | Path) -> Config:
    """Load configuration from a YAML file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    cfg = Config()
    for section_name, section_dict in raw.items():
        if isinstance(section_dict, dict) and hasattr(cfg, section_name):
            _update_dataclass(getattr(cfg, section_name), section_dict)
    return cfg


def parse_args(description: str = "") -> Config:
    """Parse CLI arguments and return a Config object."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--hf_dataset", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None, help="For inference")
    parser.add_argument("--image_dir", type=str, default=None, help="For batch inference")

    args = parser.parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.data_dir:
        cfg.dataset.data_dir = args.data_dir
    if args.hf_dataset:
        cfg.dataset.hf_dataset = args.hf_dataset
    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.num_epochs:
        cfg.training.num_epochs = args.num_epochs
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.learning_rate:
        cfg.training.learning_rate = args.learning_rate
    if args.lora_r:
        cfg.lora.r = args.lora_r
    if args.checkpoint_path:
        cfg.inference.checkpoint_path = args.checkpoint_path

    # Store extra args for inference convenience
    cfg._image_path = args.image_path
    cfg._image_dir = args.image_dir

    return cfg

"""Tests for src/config.py — configuration loading and dataclass defaults."""

import yaml
import pytest

from src.config import (
    Config,
    ModelConfig,
    LoRAConfig,
    DatasetConfig,
    TrainingConfig,
    EvaluationConfig,
    InferenceConfig,
    load_config,
    _update_dataclass,
)


# ── Dataclass Defaults ──────────────────────────────────────────

class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.name == "google/vit-large-patch16-224"
        assert cfg.num_labels is None
        assert cfg.pretrained is True

    def test_custom_values(self):
        cfg = ModelConfig(name="resnet50", num_labels=100, pretrained=False)
        assert cfg.name == "resnet50"
        assert cfg.num_labels == 100
        assert cfg.pretrained is False


class TestLoRAConfig:
    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.r == 16
        assert cfg.lora_alpha == 32
        assert cfg.lora_dropout == 0.1
        assert cfg.target_modules == ["query", "value"]
        assert cfg.bias == "none"

    def test_custom_rank(self):
        cfg = LoRAConfig(r=4, lora_alpha=8)
        assert cfg.r == 4
        assert cfg.lora_alpha == 8


class TestDatasetConfig:
    def test_defaults(self):
        cfg = DatasetConfig()
        assert cfg.data_dir is None
        assert cfg.hf_dataset is None
        assert cfg.image_size == 224
        assert cfg.train_split == "train"
        assert cfg.val_split == "val"
        assert cfg.test_split is None
        assert cfg.train_ratio == 0.8

    def test_local_data_dir(self):
        cfg = DatasetConfig(data_dir="/tmp/data")
        assert cfg.data_dir == "/tmp/data"


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.output_dir == "./outputs"
        assert cfg.num_epochs == 10
        assert cfg.batch_size == 32
        assert cfg.learning_rate == 5e-4
        assert cfg.weight_decay == 0.01
        assert cfg.warmup_ratio == 0.1
        assert cfg.lr_scheduler == "cosine"
        assert cfg.max_grad_norm == 1.0
        assert cfg.fp16 is True
        assert cfg.seed == 42
        assert cfg.save_strategy == "epoch"
        assert cfg.save_total_limit == 3

    def test_custom_epochs(self):
        cfg = TrainingConfig(num_epochs=50, learning_rate=1e-3)
        assert cfg.num_epochs == 50
        assert cfg.learning_rate == 1e-3


class TestEvaluationConfig:
    def test_defaults(self):
        cfg = EvaluationConfig()
        assert cfg.batch_size == 64
        assert cfg.metrics == ["accuracy", "f1", "precision", "recall"]


class TestInferenceConfig:
    def test_defaults(self):
        cfg = InferenceConfig()
        assert cfg.checkpoint_path is None
        assert cfg.batch_size == 1
        assert cfg.top_k == 5
        assert cfg.device == "auto"


class TestConfig:
    def test_all_sections_exist(self):
        cfg = Config()
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.lora, LoRAConfig)
        assert isinstance(cfg.dataset, DatasetConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert isinstance(cfg.evaluation, EvaluationConfig)
        assert isinstance(cfg.inference, InferenceConfig)


# ── _update_dataclass ────────────────────────────────────────────

class TestUpdateDataclass:
    def test_update_existing_fields(self):
        cfg = ModelConfig()
        _update_dataclass(cfg, {"name": "custom-model", "num_labels": 20})
        assert cfg.name == "custom-model"
        assert cfg.num_labels == 20

    def test_ignore_unknown_fields(self):
        cfg = ModelConfig()
        _update_dataclass(cfg, {"unknown_field": "value"})
        # Should not raise and original fields are unchanged
        assert cfg.name == "google/vit-large-patch16-224"

    def test_partial_update(self):
        cfg = LoRAConfig()
        _update_dataclass(cfg, {"r": 4})
        assert cfg.r == 4
        assert cfg.lora_alpha == 32  # unchanged


# ── load_config ──────────────────────────────────────────────────

class TestLoadConfig:
    def test_load_from_yaml(self, sample_yaml_config):
        cfg = load_config(sample_yaml_config)
        assert isinstance(cfg, Config)
        assert cfg.model.num_labels == 10
        assert cfg.lora.r == 8
        assert cfg.lora.lora_alpha == 16
        assert cfg.lora.lora_dropout == 0.05
        assert cfg.training.num_epochs == 2
        assert cfg.training.batch_size == 4
        assert cfg.training.learning_rate == 1e-3
        assert cfg.evaluation.batch_size == 8
        assert cfg.inference.device == "cpu"

    def test_load_default_yaml(self):
        cfg = load_config("configs/default.yaml")
        assert cfg.model.name == "google/vit-large-patch16-224"
        assert cfg.lora.r == 16
        assert cfg.training.num_epochs == 10

    def test_load_empty_yaml_raises(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        # empty YAML returns None, which causes AttributeError in load_config
        with pytest.raises(AttributeError):
            load_config(empty)

    def test_load_partial_yaml(self, tmp_path):
        partial = tmp_path / "partial.yaml"
        partial.write_text(yaml.dump({"model": {"num_labels": 5}}))
        cfg = load_config(partial)
        assert cfg.model.num_labels == 5
        # Other defaults preserved
        assert cfg.model.name == "google/vit-large-patch16-224"
        assert cfg.lora.r == 16

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

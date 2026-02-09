"""Tests for src/model.py — model construction and checkpoint resolution."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.config import Config
from src.model import _resolve_checkpoint, _print_trainable_params


# ── _resolve_checkpoint ──────────────────────────────────────────

class TestResolveCheckpoint:
    def test_explicit_checkpoint_path(self):
        cfg = Config()
        cfg.inference.checkpoint_path = "/some/checkpoint"
        result = _resolve_checkpoint(cfg)
        assert result == "/some/checkpoint"

    def test_best_model_directory(self, tmp_path):
        cfg = Config()
        cfg.inference.checkpoint_path = None
        cfg.training.output_dir = str(tmp_path)

        best_dir = tmp_path / "best_model"
        best_dir.mkdir()

        result = _resolve_checkpoint(cfg)
        assert result == str(best_dir)

    def test_latest_checkpoint_fallback(self, tmp_path):
        cfg = Config()
        cfg.inference.checkpoint_path = None
        cfg.training.output_dir = str(tmp_path)

        # Create checkpoint dirs (no best_model)
        (tmp_path / "checkpoint-1").mkdir()
        (tmp_path / "checkpoint-2").mkdir()

        result = _resolve_checkpoint(cfg)
        assert "checkpoint-" in result

    def test_no_checkpoint_raises(self, tmp_path):
        cfg = Config()
        cfg.inference.checkpoint_path = None
        cfg.training.output_dir = str(tmp_path)

        with pytest.raises(FileNotFoundError, match="No checkpoint found"):
            _resolve_checkpoint(cfg)


# ── _print_trainable_params ──────────────────────────────────────

class TestPrintTrainableParams:
    def test_prints_output(self, capsys):
        import torch.nn as nn
        model = nn.Linear(10, 5)
        _print_trainable_params(model)
        captured = capsys.readouterr()
        assert "Trainable params:" in captured.out
        assert "LoRA reduces" in captured.out

    def test_frozen_model(self, capsys):
        import torch.nn as nn
        model = nn.Linear(10, 5)
        for p in model.parameters():
            p.requires_grad = False
        _print_trainable_params(model)
        captured = capsys.readouterr()
        assert "0 /" in captured.out or "0.00%" in captured.out

    def test_partially_frozen_model(self, capsys):
        import torch.nn as nn
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 3))
        # Freeze first layer only
        for p in model[0].parameters():
            p.requires_grad = False
        _print_trainable_params(model)
        captured = capsys.readouterr()
        assert "Trainable params:" in captured.out

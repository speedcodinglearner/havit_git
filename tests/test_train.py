"""Tests for src/train.py — training utilities."""

import json
import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.train import set_seed, evaluate_epoch, _cleanup_checkpoints


# ── set_seed ─────────────────────────────────────────────────────

class TestSetSeed:
    def test_reproducible_random(self):
        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        assert a == b

    def test_reproducible_numpy(self):
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_reproducible_torch(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        set_seed(1)
        a = random.random()
        set_seed(2)
        b = random.random()
        assert a != b


# ── evaluate_epoch ───────────────────────────────────────────────

class TestEvaluateEpoch:
    def _make_dataloader(self, n_batches=3, batch_size=4, n_classes=3):
        batches = []
        for _ in range(n_batches):
            batches.append({
                "pixel_values": torch.randn(batch_size, 3, 224, 224),
                "labels": torch.randint(0, n_classes, (batch_size,)),
            })
        return batches

    def test_returns_loss_and_accuracy(self):
        model = nn.Linear(3 * 224 * 224, 3)

        # Wrap model to return object with .loss and .logits
        class WrappedModel(nn.Module):
            def __init__(self, linear):
                super().__init__()
                self.linear = linear
                self.criterion = nn.CrossEntropyLoss()

            def forward(self, pixel_values, labels=None):
                x = pixel_values.view(pixel_values.size(0), -1)
                logits = self.linear(x)
                loss = self.criterion(logits, labels) if labels is not None else None
                return MagicMock(loss=loss, logits=logits)

        wrapped = WrappedModel(model)
        dataloader = self._make_dataloader()
        criterion = nn.CrossEntropyLoss()

        val_loss, val_acc = evaluate_epoch(wrapped, dataloader, criterion, "cpu", False)
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert val_loss > 0
        assert 0 <= val_acc <= 1

    def test_perfect_model_accuracy(self):
        """A model that always predicts correctly should have accuracy ~1.0."""
        class PerfectModel(nn.Module):
            def forward(self, pixel_values, labels=None):
                logits = torch.zeros(labels.size(0), 3)
                for i, lbl in enumerate(labels):
                    logits[i, lbl] = 100.0  # Very high for correct class
                loss = torch.tensor(0.001)
                return MagicMock(loss=loss, logits=logits)

        model = PerfectModel()
        dataloader = self._make_dataloader()
        criterion = nn.CrossEntropyLoss()

        val_loss, val_acc = evaluate_epoch(model, dataloader, criterion, "cpu", False)
        assert val_acc == 1.0


# ── _cleanup_checkpoints ────────────────────────────────────────

class TestCleanupCheckpoints:
    def test_keeps_latest_n(self, tmp_path):
        import time
        for i in range(5):
            (tmp_path / f"checkpoint-{i}").mkdir()
            time.sleep(0.05)  # Ensure different mtime

        _cleanup_checkpoints(tmp_path, keep=2)
        remaining = sorted(tmp_path.glob("checkpoint-*"))
        assert len(remaining) == 2

    def test_does_not_remove_best_model(self, tmp_path):
        (tmp_path / "best_model").mkdir()
        (tmp_path / "checkpoint-1").mkdir()
        (tmp_path / "checkpoint-2").mkdir()

        _cleanup_checkpoints(tmp_path, keep=1)
        assert (tmp_path / "best_model").exists()

    def test_no_checkpoints(self, tmp_path):
        # Should not raise
        _cleanup_checkpoints(tmp_path, keep=3)

    def test_fewer_than_limit(self, tmp_path):
        (tmp_path / "checkpoint-1").mkdir()
        _cleanup_checkpoints(tmp_path, keep=5)
        assert (tmp_path / "checkpoint-1").exists()

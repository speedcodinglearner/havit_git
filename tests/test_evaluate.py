"""Tests for src/evaluate.py — metrics computation and visualization."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluate import (
    _compute_metrics,
    _resolve_device,
    _save_metrics,
    _plot_confusion_matrix,
    _plot_per_class_metrics,
    _plot_training_curves,
)


# ── _resolve_device ──────────────────────────────────────────────

class TestResolveDevice:
    def test_cpu(self):
        assert _resolve_device("cpu") == "cpu"

    def test_cuda(self):
        assert _resolve_device("cuda") == "cuda"

    def test_auto_resolves(self):
        result = _resolve_device("auto")
        assert result in ("cpu", "cuda")


# ── _compute_metrics ─────────────────────────────────────────────

class TestComputeMetrics:
    def test_perfect_predictions(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        preds = np.array([0, 1, 2, 0, 1, 2])
        probs = np.eye(3)[labels]  # One-hot
        class_names = ["A", "B", "C"]

        metrics = _compute_metrics(labels, preds, probs, class_names)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0
        assert metrics["precision_macro"] == 1.0
        assert metrics["recall_macro"] == 1.0
        assert metrics["total_samples"] == 6

    def test_random_predictions(self):
        np.random.seed(42)
        labels = np.random.randint(0, 3, size=100)
        preds = np.random.randint(0, 3, size=100)
        probs = np.random.dirichlet([1, 1, 1], size=100)
        class_names = ["A", "B", "C"]

        metrics = _compute_metrics(labels, preds, probs, class_names)
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1
        assert metrics["total_samples"] == 100

    def test_per_class_metrics_present(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        preds = np.array([0, 1, 1, 0, 2, 2])
        probs = np.eye(3)[preds]
        class_names = ["cat", "dog", "bird"]

        metrics = _compute_metrics(labels, preds, probs, class_names)
        assert "per_class" in metrics
        for name in class_names:
            assert name in metrics["per_class"]
            assert "precision" in metrics["per_class"][name]
            assert "recall" in metrics["per_class"][name]
            assert "f1-score" in metrics["per_class"][name]

    def test_top_k_accuracy(self):
        labels = np.array([0, 1, 2, 3, 4])
        preds = np.array([0, 0, 0, 0, 0])
        # Create probabilities where true label is in top-3
        probs = np.full((5, 5), 0.05)
        for i in range(5):
            probs[i, labels[i]] = 0.5
            probs[i, 0] = max(probs[i, 0], 0.3)
        class_names = [f"c{i}" for i in range(5)]

        metrics = _compute_metrics(labels, preds, probs, class_names)
        assert "top3_accuracy" in metrics
        assert "top5_accuracy" in metrics
        assert metrics["top5_accuracy"] == 1.0  # all true labels present

    def test_binary_classification(self):
        labels = np.array([0, 0, 1, 1, 0, 1])
        preds = np.array([0, 1, 1, 0, 0, 1])
        probs = np.column_stack([1 - preds * 0.8, preds * 0.8])
        class_names = ["negative", "positive"]

        metrics = _compute_metrics(labels, preds, probs, class_names)
        assert 0 <= metrics["accuracy"] <= 1
        assert metrics["total_samples"] == 6

    def test_single_class_all_correct(self):
        labels = np.array([0, 0, 0])
        preds = np.array([0, 0, 0])
        probs = np.array([[1.0], [1.0], [1.0]])
        class_names = ["only"]

        metrics = _compute_metrics(labels, preds, probs, class_names)
        assert metrics["accuracy"] == 1.0


# ── _save_metrics ────────────────────────────────────────────────

class TestSaveMetrics:
    def test_saves_json_file(self, tmp_path):
        metrics = {
            "accuracy": 0.95,
            "f1_macro": 0.93,
            "total_samples": 100,
            "per_class": {"A": {"precision": 0.9, "recall": 0.8}},
        }
        _save_metrics(metrics, tmp_path)
        json_path = tmp_path / "metrics.json"
        assert json_path.exists()

        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded["accuracy"] == 0.95

    def test_numpy_serialization(self, tmp_path):
        metrics = {
            "accuracy": np.float64(0.95),
            "count": np.int64(42),
            "array": np.array([1, 2, 3]),
            "total_samples": 100,
        }
        _save_metrics(metrics, tmp_path)
        with open(tmp_path / "metrics.json") as f:
            loaded = json.load(f)
        assert loaded["accuracy"] == 0.95
        assert loaded["count"] == 42
        assert loaded["array"] == [1, 2, 3]


# ── Visualization Tests ─────────────────────────────────────────

class TestPlotConfusionMatrix:
    def test_creates_png(self, tmp_path):
        labels = np.array([0, 0, 1, 1, 2, 2])
        preds = np.array([0, 1, 1, 0, 2, 2])
        class_names = ["cat", "dog", "bird"]

        _plot_confusion_matrix(labels, preds, class_names, tmp_path)
        assert (tmp_path / "confusion_matrix.png").exists()
        assert (tmp_path / "confusion_matrix.png").stat().st_size > 0


class TestPlotPerClassMetrics:
    def test_creates_png(self, tmp_path):
        metrics = {
            "per_class": {
                "cat": {"precision": 0.9, "recall": 0.8, "f1-score": 0.85},
                "dog": {"precision": 0.7, "recall": 0.9, "f1-score": 0.79},
                "bird": {"precision": 0.95, "recall": 0.95, "f1-score": 0.95},
            }
        }
        class_names = ["cat", "dog", "bird"]
        _plot_per_class_metrics(metrics, class_names, tmp_path)
        assert (tmp_path / "per_class_metrics.png").exists()

    def test_empty_per_class(self, tmp_path):
        metrics = {}
        _plot_per_class_metrics(metrics, [], tmp_path)
        # Should not create file or raise
        assert not (tmp_path / "per_class_metrics.png").exists()


class TestPlotTrainingCurves:
    def test_creates_png(self, tmp_path):
        history = {
            "train_loss": [1.5, 1.2, 0.9, 0.7, 0.5],
            "val_loss": [1.6, 1.3, 1.0, 0.8, 0.6],
            "val_accuracy": [0.3, 0.5, 0.6, 0.7, 0.75],
            "learning_rate": [5e-4, 4e-4, 3e-4, 2e-4, 1e-4],
        }
        history_path = tmp_path / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f)

        _plot_training_curves(history_path, tmp_path)
        assert (tmp_path / "training_curves.png").exists()
        assert (tmp_path / "training_curves.png").stat().st_size > 0

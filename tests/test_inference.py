"""Tests for src/inference.py — prediction utilities."""

import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

from src.inference import predict_single, predict_batch


# ── predict_single ───────────────────────────────────────────────

class TestPredictSingle:
    def _setup_mock_model(self, num_classes=5):
        model = MagicMock()
        logits = torch.randn(1, num_classes)
        logits[0, 2] = 10.0  # Make class 2 the top prediction
        model.return_value = MagicMock(logits=logits)
        return model

    def _setup_mock_processor(self):
        processor = MagicMock()
        processor.return_value = MagicMock()
        processor.return_value.to = MagicMock(return_value={"pixel_values": torch.randn(1, 3, 224, 224)})
        return processor

    def test_returns_list_of_predictions(self, tmp_path):
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)

        model = self._setup_mock_model()
        processor = self._setup_mock_processor()
        id2label = {i: f"class_{i}" for i in range(5)}

        results = predict_single(str(img_path), model, processor, id2label, "cpu", top_k=3)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_top_prediction_has_highest_confidence(self, tmp_path):
        img = Image.new("RGB", (64, 64))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)

        model = self._setup_mock_model()
        processor = self._setup_mock_processor()
        id2label = {i: f"class_{i}" for i in range(5)}

        results = predict_single(str(img_path), model, processor, id2label, "cpu", top_k=5)
        confidences = [r["confidence"] for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_result_structure(self, tmp_path):
        img = Image.new("RGB", (64, 64))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)

        model = self._setup_mock_model()
        processor = self._setup_mock_processor()
        id2label = {i: f"class_{i}" for i in range(5)}

        results = predict_single(str(img_path), model, processor, id2label, "cpu", top_k=1)
        assert "label" in results[0]
        assert "confidence" in results[0]
        assert isinstance(results[0]["confidence"], float)

    def test_top_k_clamped_to_num_classes(self, tmp_path):
        img = Image.new("RGB", (64, 64))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)

        model = self._setup_mock_model(num_classes=3)
        processor = self._setup_mock_processor()
        id2label = {i: f"c{i}" for i in range(3)}

        results = predict_single(str(img_path), model, processor, id2label, "cpu", top_k=10)
        assert len(results) == 3  # Clamped to num_classes


# ── predict_batch ────────────────────────────────────────────────

class TestPredictBatch:
    def test_empty_directory(self, tmp_path):
        model = MagicMock()
        processor = MagicMock()
        id2label = {0: "a"}

        results = predict_batch(str(tmp_path), model, processor, id2label, "cpu")
        assert results == {}

    def test_ignores_non_image_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("not an image")
        (tmp_path / "data.csv").write_text("a,b,c")

        model = MagicMock()
        processor = MagicMock()
        id2label = {0: "a"}

        results = predict_batch(str(tmp_path), model, processor, id2label, "cpu")
        assert results == {}

    def test_processes_image_files(self, tmp_path):
        # Create test images
        for name in ["a.jpg", "b.png", "c.bmp"]:
            img = Image.new("RGB", (64, 64))
            img.save(tmp_path / name)

        model = MagicMock()
        logits = torch.tensor([[1.0, 0.0]])
        model.return_value = MagicMock(logits=logits)

        processor = MagicMock()
        processor.return_value = MagicMock()
        processor.return_value.to = MagicMock(
            return_value={"pixel_values": torch.randn(1, 3, 224, 224)}
        )
        id2label = {0: "pos", 1: "neg"}

        results = predict_batch(str(tmp_path), model, processor, id2label, "cpu", top_k=1)
        assert len(results) == 3
        assert "a.jpg" in results
        assert "b.png" in results
        assert "c.bmp" in results

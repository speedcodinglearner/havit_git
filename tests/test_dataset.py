"""Tests for src/dataset.py — dataset utilities and data pipeline."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from src.config import Config
from src.dataset import (
    ImageClassificationDataset,
    _detect_keys,
    _build_label_maps,
)


# ── _detect_keys ─────────────────────────────────────────────────

class TestDetectKeys:
    def _make_dataset(self, columns):
        ds = MagicMock()
        ds.column_names = columns
        return ds

    def test_standard_keys(self):
        ds = self._make_dataset(["image", "label"])
        img_key, lbl_key = _detect_keys(ds)
        assert img_key == "image"
        assert lbl_key == "label"

    def test_alternative_image_key(self):
        ds = self._make_dataset(["img", "label"])
        img_key, _ = _detect_keys(ds)
        assert img_key == "img"

    def test_pixel_values_key(self):
        ds = self._make_dataset(["pixel_values", "label"])
        img_key, _ = _detect_keys(ds)
        assert img_key == "pixel_values"

    def test_fine_label_key(self):
        ds = self._make_dataset(["image", "fine_label"])
        _, lbl_key = _detect_keys(ds)
        assert lbl_key == "fine_label"

    def test_labels_key(self):
        ds = self._make_dataset(["image", "labels"])
        _, lbl_key = _detect_keys(ds)
        assert lbl_key == "labels"

    def test_coarse_label_key(self):
        ds = self._make_dataset(["image", "coarse_label"])
        _, lbl_key = _detect_keys(ds)
        assert lbl_key == "coarse_label"

    def test_fallback_defaults(self):
        ds = self._make_dataset(["data", "target"])
        img_key, lbl_key = _detect_keys(ds)
        # Falls through all checks, uses defaults
        assert img_key == "image"
        assert lbl_key == "label"


# ── _build_label_maps ────────────────────────────────────────────

class TestBuildLabelMaps:
    def test_with_class_label_feature(self):
        ds = MagicMock()
        feature = MagicMock()
        feature.names = ["cat", "dog", "bird"]
        ds.features = {"label": feature}

        label2id, id2label = _build_label_maps(ds, "label")
        assert label2id == {"cat": 0, "dog": 1, "bird": 2}
        assert id2label == {0: "cat", 1: "dog", 2: "bird"}

    def test_without_class_label_feature(self):
        ds = MagicMock()
        feature = MagicMock(spec=[])  # No 'names' attribute
        ds.features = {"label": feature}
        ds.__getitem__ = MagicMock(return_value=[2, 0, 1, 1, 0, 2])

        label2id, id2label = _build_label_maps(ds, "label")
        assert label2id == {"0": 0, "1": 1, "2": 2}
        assert id2label == {0: "0", 1: "1", 2: "2"}

    def test_single_class(self):
        ds = MagicMock()
        feature = MagicMock()
        feature.names = ["only_class"]
        ds.features = {"label": feature}

        label2id, id2label = _build_label_maps(ds, "label")
        assert len(label2id) == 1
        assert label2id["only_class"] == 0

    def test_many_classes(self):
        ds = MagicMock()
        names = [f"class_{i}" for i in range(100)]
        feature = MagicMock()
        feature.names = names
        ds.features = {"label": feature}

        label2id, id2label = _build_label_maps(ds, "label")
        assert len(label2id) == 100
        assert id2label[50] == "class_50"


# ── ImageClassificationDataset ───────────────────────────────────

class TestImageClassificationDataset:
    def _make_mock_processor(self):
        import torch
        processor = MagicMock()
        processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
        return processor

    def _make_mock_hf_dataset(self, n=5):
        items = []
        for i in range(n):
            img = Image.new("RGB", (64, 64), color=(i * 50, 100, 150))
            items.append({"image": img, "label": i % 3})

        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=n)
        ds.__getitem__ = MagicMock(side_effect=lambda idx: items[idx])
        return ds

    def test_len(self):
        ds = self._make_mock_hf_dataset(10)
        processor = self._make_mock_processor()
        dataset = ImageClassificationDataset(ds, processor)
        assert len(dataset) == 10

    def test_getitem_returns_correct_keys(self):
        ds = self._make_mock_hf_dataset(3)
        processor = self._make_mock_processor()
        dataset = ImageClassificationDataset(ds, processor)
        item = dataset[0]
        assert "pixel_values" in item
        assert "labels" in item

    def test_getitem_pixel_values_shape(self):
        ds = self._make_mock_hf_dataset(3)
        processor = self._make_mock_processor()
        dataset = ImageClassificationDataset(ds, processor)
        item = dataset[0]
        assert item["pixel_values"].shape == (3, 224, 224)

    def test_getitem_label_is_tensor(self):
        import torch
        ds = self._make_mock_hf_dataset(3)
        processor = self._make_mock_processor()
        dataset = ImageClassificationDataset(ds, processor)
        item = dataset[0]
        assert isinstance(item["labels"], torch.Tensor)
        assert item["labels"].dtype == torch.long

    def test_grayscale_image_converted_to_rgb(self):
        img = Image.new("L", (64, 64), color=128)
        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=1)
        ds.__getitem__ = MagicMock(return_value={"image": img, "label": 0})

        processor = self._make_mock_processor()
        dataset = ImageClassificationDataset(ds, processor)
        item = dataset[0]
        # Should not raise; processor receives an RGB image
        assert "pixel_values" in item

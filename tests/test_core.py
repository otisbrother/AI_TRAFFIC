from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from inspect_dataset import audit_split
from traffic_utils import detection_summary, safe_source_name


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class CoreTests(unittest.TestCase):
    def test_stream_credentials_are_redacted(self):
        source = "rtsp://camera-user:secret@192.0.2.1/live"
        self.assertEqual(safe_source_name(source), "rtsp://192.0.2.1/live")

    def test_detection_summary(self):
        boxes = SimpleNamespace(cls=FakeTensor([0, 1, 0]), conf=FakeTensor([0.9, 0.8, 0.7]))
        result = SimpleNamespace(boxes=boxes, names={0: "car", 1: "person"})
        summary = detection_summary([result])
        self.assertEqual(summary["frames"], 1)
        self.assertEqual(summary["detections"], 3)
        self.assertEqual(summary["by_class"], {"car": 2, "person": 1})
        self.assertEqual(summary["mean_confidence"], 0.8)

    def test_audit_split_finds_invalid_coordinate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            (images / "sample.jpg").touch()
            (labels / "sample.txt").write_text("0 0.5 1.2 0.2 0.2\n", encoding="utf-8")
            report = audit_split(images, {0: "car"})
        self.assertEqual(report["objects"], 1)
        self.assertEqual(len(report["errors"]), 1)
        self.assertIn("[0, 1]", report["errors"][0])


if __name__ == "__main__":
    unittest.main()

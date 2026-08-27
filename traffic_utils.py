"""Shared helpers for the traffic-detection commands."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data.yaml"
DEFAULT_MODEL = ROOT / "runs" / "detect" / "train2" / "weights" / "best.pt"


def configure_utf8_console() -> None:
    """Make Vietnamese CLI output reliable on Windows and redirected consoles."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def existing_path(value: str | Path, description: str) -> Path:
    """Return a resolved path or raise a useful CLI error."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{description} không tồn tại: {path}")
    return path


def load_yolo(model_path: str | Path):
    """Load Ultralytics lazily so utility modules remain importable without it."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Chưa cài Ultralytics. Hãy chạy: pip install -r requirements.txt"
        ) from exc
    return YOLO(str(existing_path(model_path, "Model")))


def normalize_source(source: str) -> str | int:
    """Convert a numeric source to a webcam index; resolve local paths."""
    if source.isdigit():
        return int(source)
    if source.startswith(("http://", "https://", "rtsp://", "rtmp://")):
        return source
    return str(existing_path(source, "Nguồn đầu vào"))


def safe_source_name(source: str) -> str:
    """Remove credentials from stream URLs before logging them."""
    return re.sub(r"(?<=://)[^/@]+@", "", source)


def detection_summary(results: Iterable[Any]) -> dict[str, Any]:
    """Build a JSON-serializable summary from Ultralytics results."""
    counts: Counter[str] = Counter()
    confidences: list[float] = []
    frames = 0
    for result in results:
        frames += 1
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        names = getattr(result, "names", {})
        for class_id, confidence in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            counts[str(names.get(int(class_id), int(class_id)))] += 1
            confidences.append(float(confidence))
    return {
        "frames": frames,
        "detections": sum(counts.values()),
        "by_class": dict(sorted(counts.items())),
        "mean_confidence": round(sum(confidences) / len(confidences), 4)
        if confidences
        else None,
    }


def json_ready(value: Any) -> Any:
    """Recursively convert common tensor/scalar values for JSON output."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(payload: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path.resolve()

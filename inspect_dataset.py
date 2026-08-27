"""Audit a YOLO-format dataset and report class distribution and label errors."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from traffic_utils import DEFAULT_DATA, configure_utf8_console, existing_path, write_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def audit_split(image_dir: Path, class_names: dict[int, str]) -> dict[str, Any]:
    label_dir = image_dir.parent / "labels"
    images = {
        path.stem: path
        for path in image_dir.glob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    labels = {path.stem: path for path in label_dir.glob("*.txt")} if label_dir.exists() else {}
    counts: Counter[int] = Counter()
    errors: list[str] = []
    empty_labels = 0

    for stem in sorted(images.keys() - labels.keys()):
        errors.append(f"Thiếu nhãn: {images[stem]}")
    for stem in sorted(labels.keys() - images.keys()):
        errors.append(f"Nhãn không có ảnh: {labels[stem]}")

    for label_path in labels.values():
        lines = label_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            empty_labels += 1
        for line_number, line in enumerate(lines, start=1):
            parts = line.split()
            location = f"{label_path}:{line_number}"
            if len(parts) != 5:
                errors.append(f"Sai số cột ({len(parts)}): {location}")
                continue
            try:
                class_id = int(parts[0])
                x, y, width, height = map(float, parts[1:])
            except ValueError:
                errors.append(f"Giá trị không hợp lệ: {location}")
                continue
            if class_id not in class_names:
                errors.append(f"Lớp {class_id} không khai báo: {location}")
            if not all(0 <= value <= 1 for value in (x, y, width, height)):
                errors.append(f"Tọa độ ngoài [0, 1]: {location}")
            if width <= 0 or height <= 0:
                errors.append(f"Kích thước bbox không dương: {location}")
            counts[class_id] += 1

    return {
        "images": len(images),
        "labels": len(labels),
        "empty_labels": empty_labels,
        "objects": sum(counts.values()),
        "class_distribution": {
            class_names.get(class_id, str(class_id)): count
            for class_id, count in sorted(counts.items())
        },
        "errors": errors,
    }


def audit_dataset(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configured_root = Path(config.get("path", "."))
    dataset_root = (
        configured_root
        if configured_root.is_absolute()
        else (config_path.parent / configured_root).resolve()
    )
    raw_names = config.get("names", {})
    class_names = (
        dict(enumerate(raw_names))
        if isinstance(raw_names, list)
        else {int(key): value for key, value in raw_names.items()}
    )

    report: dict[str, Any] = {
        "dataset": str(dataset_root),
        "classes": class_names,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        split_value = config.get(split)
        if not split_value:
            continue
        split_path = Path(split_value)
        image_dir = split_path if split_path.is_absolute() else dataset_root / split_path
        report["splits"][split] = (
            audit_split(image_dir, class_names)
            if image_dir.exists()
            else {"errors": [f"Thư mục không tồn tại: {image_dir}"]}
        )
    report["error_count"] = sum(
        len(split_report["errors"]) for split_report in report["splits"].values()
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output", default="runs/dataset-report.json")
    parser.add_argument("--strict", action="store_true", help="Trả mã lỗi nếu dataset có lỗi")
    return parser


def main() -> None:
    configure_utf8_console()
    args = build_parser().parse_args()
    report = audit_dataset(existing_path(args.data, "Cấu hình dataset"))
    output = write_json(report, args.output)
    for split, result in report["splits"].items():
        print(
            f"{split}: {result.get('images', 0)} ảnh, "
            f"{result.get('objects', 0)} đối tượng, {len(result['errors'])} lỗi"
        )
    print(f"Báo cáo: {output}")
    if args.strict and report["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

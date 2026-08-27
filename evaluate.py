"""Evaluate a trained traffic detector and export metrics as JSON."""

from __future__ import annotations

import argparse

from traffic_utils import (
    DEFAULT_DATA,
    DEFAULT_MODEL,
    configure_utf8_console,
    existing_path,
    load_yolo,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="runs/evaluation/metrics.json")
    return parser


def main() -> None:
    configure_utf8_console()
    args = build_parser().parse_args()
    model = load_yolo(args.model)
    metrics = model.val(
        data=str(existing_path(args.data, "Cấu hình dataset")),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True,
    )
    payload = {
        "model": str(existing_path(args.model, "Model")),
        "metrics": metrics.results_dict,
        "speed_ms": metrics.speed,
    }
    output = write_json(payload, args.output)
    print(f"Đã lưu kết quả đánh giá: {output}")


if __name__ == "__main__":
    main()

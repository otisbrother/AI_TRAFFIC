"""Train a YOLOv8 traffic-object detector."""

from __future__ import annotations

import argparse

from traffic_utils import DEFAULT_DATA, ROOT, configure_utf8_console, existing_path, load_yolo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n.pt", help="Model/weights khởi tạo")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Tệp cấu hình dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Ví dụ: cpu, 0; mặc định tự chọn")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--name", default="traffic-yolov8n")
    parser.add_argument("--project", default=str(ROOT / "runs" / "detect"))
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    configure_utf8_console()
    args = build_parser().parse_args()
    if args.epochs < 1 or args.imgsz < 32 or args.batch == 0:
        raise SystemExit("epochs >= 1, imgsz >= 32 và batch phải khác 0")
    model = load_yolo(args.model)
    model.train(
        data=str(existing_path(args.data, "Cấu hình dataset")),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        resume=args.resume,
        pretrained=True,
        seed=42,
        deterministic=True,
        plots=True,
    )


if __name__ == "__main__":
    main()

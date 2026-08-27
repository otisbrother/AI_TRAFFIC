"""Detect traffic objects in an image, video, stream, directory, or webcam."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_utils import (
    DEFAULT_MODEL,
    ROOT,
    configure_utf8_console,
    detection_summary,
    load_yolo,
    normalize_source,
    safe_source_name,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Đường dẫn, URL stream hoặc số webcam (ví dụ 0)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default=str(ROOT / "runs" / "detect"))
    parser.add_argument("--name", default="predict")
    parser.add_argument("--show", action="store_true", help="Mở cửa sổ xem trực tiếp")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    return parser


def main() -> None:
    configure_utf8_console()
    args = build_parser().parse_args()
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        raise SystemExit("conf và iou phải nằm trong khoảng [0, 1]")
    model = load_yolo(args.model)
    results = model.predict(
        source=normalize_source(args.source),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        show=args.show,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=False,
        stream=True,
    )
    summary = detection_summary(results)
    summary["source"] = safe_source_name(args.source)
    output_dir = Path(args.project).expanduser() / args.name
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    summary["output_dir"] = str(output_dir)
    output = write_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Đã lưu kết quả: {output.parent}")


if __name__ == "__main__":
    main()

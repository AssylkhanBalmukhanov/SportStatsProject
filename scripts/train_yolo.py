from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLO football detector weights.")
    parser.add_argument("--data", required=True, type=Path, help="Path to dataset data.yaml.")
    parser.add_argument("--model", default="yolov8x.pt", help="Base model, for example yolov8x.pt or yolo11m.pt.")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--device", default=None, help="Device string, for example 0, cpu, or mps.")
    parser.add_argument("--project", default="runs/football", type=Path)
    parser.add_argument("--name", default="train")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install vision dependencies first: python -m pip install -r requirements-vision.txt") from exc

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        project=str(args.project),
        name=args.name,
    )
    print(results.save_dir)
    print(f"Copy {results.save_dir}/weights/best.pt to models/best.pt for the app.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

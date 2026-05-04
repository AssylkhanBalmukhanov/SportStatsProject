from __future__ import annotations

import argparse
from pathlib import Path

from sportstats.services.yolo import analyze_video


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SportStats YOLO analysis on a local video.")
    parser.add_argument("video", type=Path, help="Path to the input football video.")
    parser.add_argument("--model", default="", help="Optional YOLO weights path, for example models/best.pt.")
    parser.add_argument("--output-dir", type=Path, default=Path("static/outputs"))
    parser.add_argument("--max-frames", type=int, default=120)
    args = parser.parse_args()

    result = analyze_video(
        args.video,
        output_dir=args.output_dir,
        model_path=args.model,
        max_frames=args.max_frames,
    )
    print(result.to_dict())
    return 0 if result.status in {"complete", "unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

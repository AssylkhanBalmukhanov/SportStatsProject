from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter


@dataclass(frozen=True)
class VisionAnalysis:
    status: str
    message: str
    frames_processed: int = 0
    detections: int = 0
    output_video: str | None = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def is_allowed_video(filename: str, allowed_extensions: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def analyze_video(
    video_path: Path,
    *,
    output_dir: Path,
    model_path: str | None = None,
    max_frames: int = 120,
) -> VisionAnalysis:
    if not video_path.exists():
        return VisionAnalysis(status="error", message="Video file does not exist.")

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError:
        return VisionAnalysis(
            status="unavailable",
            message="Install opencv-python and ultralytics, then set YOLO_MODEL_PATH to enable video analysis.",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_path or "yolov8n.pt"
    started_at = perf_counter()
    model = YOLO(model_name)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return VisionAnalysis(status="error", message="OpenCV could not read the uploaded video.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = output_dir / f"{video_path.stem}_annotated.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frames_processed = 0
    detections = 0
    try:
        while frames_processed < max_frames:
            success, frame = capture.read()
            if not success:
                break
            result = model(frame, verbose=False)[0]
            detections += len(result.boxes)
            writer.write(result.plot())
            frames_processed += 1
    finally:
        capture.release()
        writer.release()

    elapsed = perf_counter() - started_at
    return VisionAnalysis(
        status="complete",
        message="Video analysis completed.",
        frames_processed=frames_processed,
        detections=detections,
        output_video=f"outputs/{output_path.name}",
        elapsed_seconds=round(elapsed, 2),
    )

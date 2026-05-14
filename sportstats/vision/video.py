from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    width: int
    height: int
    frame_count: int


def read_video(video_path: Path, max_frames: int | None = None) -> tuple[list, VideoMetadata]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not read video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames = []
    while True:
        if max_frames is not None and len(frames) >= max_frames:
            break
        success, frame = capture.read()
        if not success:
            break
        frames.append(frame)

    capture.release()
    return frames, VideoMetadata(fps=fps, width=width, height=height, frame_count=source_frame_count)


def save_video(frames: list, output_path: Path, fps: float = 24.0) -> None:
    import cv2

    if not frames:
        raise ValueError("No frames were provided for saving.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

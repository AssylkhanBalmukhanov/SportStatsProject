from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(250 * 1024 * 1024)))
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", str(BASE_DIR / "static" / "uploads"))
    OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", str(BASE_DIR / "static" / "outputs"))
    ALLOWED_VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv"}
    YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "")
    VISION_MAX_FRAMES = int(os.getenv("VISION_MAX_FRAMES", "120"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


class TestConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = False
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "/tmp/sportstats-test/uploads")
    OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "/tmp/sportstats-test/outputs")
    VISION_MAX_FRAMES = 5

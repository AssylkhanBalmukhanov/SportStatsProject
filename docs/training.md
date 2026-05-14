# Training And Data Setup

## What Is Already Implemented

- Dataset download helper for Roboflow exports.
- YOLO training helper.
- Colab training template.
- Runtime pipeline that expects `models/best.pt`.

## What You Must Do

1. Download the match video.

   Put the Bundesliga sample clip in:

   ```text
   football_analysis/input_videos/08fd33_4.mp4
   ```

   The repository intentionally does not track video files.

2. Get the football detector dataset.

   Create or use your own Roboflow API key, then run:

   ```bash
   export ROBOFLOW_API_KEY=your_key
   python scripts/download_roboflow_dataset.py \
     --workspace your-workspace \
     --project your-project \
     --version 1 \
     --format yolov8
   ```

3. Train on GPU.

   Use Colab or another GPU machine:

   ```bash
   python scripts/train_yolo.py --data datasets/data.yaml --model yolov8x.pt --epochs 100 --imgsz 640
   ```

   If training is too slow, use `yolov8m.pt` or `yolov8s.pt`. If your installed Ultralytics version supports newer model names, `yolo11m.pt` is also a good option.

4. Copy trained weights.

   Copy:

   ```text
   runs/football/train/weights/best.pt
   ```

   to:

   ```text
   models/best.pt
   ```

5. Run a short smoke test.

   ```bash
   python yolo_inference.py football_analysis/input_videos/08fd33_4.mp4 --model models/best.pt --max-frames 120
   ```

6. Run the Flask demo.

   ```bash
   flask --app app run --debug
   ```

## Practical Demo Advice

- Keep `VISION_MAX_FRAMES=120` or `300` during development so the demo is fast.
- Use cached stubs unless you change the model or calibration.
- Run with `--no-stubs` after changing weights, camera-motion logic, or perspective vertices.
- Do not commit `.pt` files, videos, generated outputs, stubs, or `runs/` folders.

# Colab Football Detector Training

Run these cells in Google Colab with a GPU runtime.

```python
!pip install ultralytics roboflow
```

```python
import os
from roboflow import Roboflow

ROBOFLOW_API_KEY = os.environ["ROBOFLOW_API_KEY"]
WORKSPACE = "your-workspace"
PROJECT = "your-football-project"
VERSION = 1

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
dataset = project.version(VERSION).download("yolov8")
print(dataset.location)
```

```python
from ultralytics import YOLO

model = YOLO("yolov8x.pt")
results = model.train(data=f"{dataset.location}/data.yaml", epochs=100, imgsz=640)
print(results.save_dir)
```

Download `runs/detect/train/weights/best.pt` and place it in this repository as `models/best.pt`.

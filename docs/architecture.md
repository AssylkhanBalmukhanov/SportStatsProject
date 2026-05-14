# Architecture

```text
Browser
  |
  v
Flask routes in sportstats.create_app
  |
  +-- services/prediction.py          Poisson xG and Monte Carlo simulation
  +-- services/passing_network.py     Pass-event aggregation and graph data
  +-- services/yolo.py                Web/CLI wrapper around the CV pipeline
  |
  v
sportstats/vision
  +-- tracking.py                     Ultralytics YOLO + ByteTrack object tracks
  +-- team_assignment.py              Shirt-color k-means team assignment
  +-- ball_assignment.py              Ball possession and team control
  +-- camera_movement.py              Optical-flow camera displacement
  +-- view_transformer.py             Perspective transform to meters
  +-- speed_distance.py               Speed and distance metrics
  +-- annotations.py                  Output video overlays
```

## Module Responsibilities

- `app.py`: WSGI entry point for local, Docker, and CI smoke checks.
- `sportstats/__init__.py`: Flask app factory, routes, request validation, upload handling, and logging.
- `sportstats/config.py`: Environment-driven application settings.
- `sportstats/services/prediction.py`: Match prediction using team strength, Elo adjustment, Poisson scorelines, and Monte Carlo outcomes.
- `sportstats/services/passing_network.py`: Converts pass events into average player positions and weighted directed links.
- `sportstats/services/yolo.py`: Checks optional dependencies, resolves weights, and runs the football-analysis pipeline.
- `sportstats/vision/`: The full computer-vision workstream from the tutorial, adapted for current Ultralytics APIs.
- `tests/`: API and service checks used by CI.

## Design Notes

The heavy computer-vision runtime is optional. The Flask app and CI tests run without downloading YOLO weights or installing OpenCV. When `models/best.pt` is present and `requirements-vision.txt` is installed, the same app runs the complete CV workflow.

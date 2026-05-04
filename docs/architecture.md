# Architecture

```text
Browser
  |
  v
Flask routes in sportstats.create_app
  |
  +-- prediction.py        Poisson xG and Monte Carlo match simulation
  +-- passing_network.py   Pass-event aggregation and graph data
  +-- yolo.py              Optional YOLO/OpenCV video analysis
  |
  v
Jinja templates and JSON API responses
```

## Module Responsibilities

- `app.py`: WSGI entry point for local, Docker, and CI smoke checks.
- `sportstats/__init__.py`: Flask app factory, routes, request validation, upload handling, and logging.
- `sportstats/config.py`: Environment-driven application settings.
- `sportstats/services/prediction.py`: Match prediction using team strength, Elo adjustment, Poisson scorelines, and Monte Carlo outcomes.
- `sportstats/services/passing_network.py`: Converts pass events into average player positions and weighted directed links.
- `sportstats/services/yolo.py`: Lazily imports `opencv-python` and `ultralytics`; runs bounded video analysis when weights and dependencies are installed.
- `templates/` and `static/`: Server-rendered UI.
- `tests/`: API and service checks used by CI.

## Design Notes

The YOLO dependencies are optional at runtime because they are heavy and model weights are not stored in Git. If the dependencies are missing, the web app still runs and reports the setup step needed to enable video processing. This keeps the project demoable while the computer-vision model is developed independently.

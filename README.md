# SportStatsProject

SportStatsProject is a Flask football analytics application that combines statistical match prediction, passing-network visualization, and an optional YOLO/OpenCV video-analysis pipeline.

The project is structured to satisfy the seminar assessment requirements: demoable user stories, reproducible run instructions, Docker support, automated checks in CI, architecture documentation, and a GitHub issue/PR workflow.

## Demoable Use Cases

1. Match prediction: estimate expected goals and outcome probabilities using Poisson scoring rates and Monte Carlo simulation.
2. Elo/team-strength tuning: adjust the prediction with team ratings, attack strength, and defensive strength.
3. Passing network graphics: paste pass-event JSON and generate average player positions plus weighted passing links.
4. YOLO match-video analysis: upload a short football clip and run bounded YOLO/OpenCV detection when the optional ML dependencies and model weights are installed.
5. Architecture/process view: show the module responsibilities and project workflow used for the seminar.

## Tech Stack

- Python 3.12
- Flask
- Jinja templates and CSS
- Optional: Ultralytics YOLO and OpenCV
- Pytest and Ruff in CI
- Docker and Docker Compose

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
cp .env.example .env
flask --app app run --debug
```

Open http://127.0.0.1:5000.

For the full YOLO/OpenCV workflow, install the optional vision dependencies too:

```bash
python -m pip install -r requirements-vision.txt
```

If you only want the lightweight web app and tests, Flask and NumPy are enough. The YOLO page will stay available and explain what is missing until `opencv-python`, `ultralytics`, and model weights are configured.

## Docker

```bash
cp .env.example .env
docker compose up --build
```

Open http://127.0.0.1:5000.

To build the image with the optional YOLO runtime:

```bash
docker compose build --build-arg INSTALL_VISION=true
docker compose up
```

## Configuration

Copy `.env.example` to `.env` and adjust:

- `SECRET_KEY`: Flask secret key.
- `UPLOAD_FOLDER`: where uploaded videos are stored.
- `OUTPUT_FOLDER`: where annotated videos are written.
- `YOLO_MODEL_PATH`: optional local model path, for example `models/best.pt`.
- `VISION_MAX_FRAMES`: frame cap for demo-time video processing.
- `MAX_CONTENT_LENGTH`: upload size limit in bytes.

Model weights, uploaded videos, generated outputs, and large raw media are intentionally ignored by Git.

## Tests and Linting

```bash
ruff check .
pytest
```

Without pytest installed locally, the standard-library test runner also works:

```bash
python -m unittest discover
```

CI runs Ruff and Pytest on every push to `main` and every pull request.

## Project Structure

```text
app.py                         Flask entry point
sportstats/                    Application package
  config.py                    Environment configuration
  services/prediction.py       Poisson, Elo adjustment, Monte Carlo
  services/passing_network.py  Pass-event graph aggregation
  services/yolo.py             Optional YOLO/OpenCV analysis boundary
templates/                     Server-rendered pages
static/css/app.css             UI styling
tests/                         Service and route tests
requirements-vision.txt        Optional YOLO/OpenCV dependencies
docs/architecture.md           System diagram and responsibilities
docs/process.md                GitHub issue/PR workflow
.github/workflows/ci.yml       CI merge gate
```

## Team Roles

- Daniel: YOLO/OpenCV football video analysis, model integration, annotated-video export, CV demo preparation.
- Assylkhan Balmukhanov: Flask website, statistical predictor, Elo/Monte Carlo workstream, CI/CD, documentation, release workflow.
- Shared: issues, PR reviews, demo script, README updates, and final seminar presentation.

## Development Workflow

Use GitHub Issues as the tracker. Each task should include acceptance criteria and an assignee. Each PR should reference an issue with `Closes #123`, pass CI, and receive at least one meaningful review from the other teammate before merge.

See `docs/process.md` for the suggested board columns and evidence needed for a full contribution coefficient.

## YOLO Workstream Notes

The computer-vision path is based on the planned football-analysis tutorial: YOLO detects players, referees, and the ball; OpenCV handles video IO; later extensions can add team-color clustering, ball possession, optical flow camera compensation, perspective transformation, speed, and distance.

Reference video: https://www.youtube.com/watch?v=neBZ6huolkg

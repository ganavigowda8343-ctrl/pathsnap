# PathSnap – Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon. PathSnap matches noisy GPS observations
to a road network with a learned candidate classifier plus an HMM/Viterbi decoder.
It is deployed on two laptops: a FastAPI model server and a browser UI.

```
raw trip ──► projection ──► candidate arcs ──► features ──► classifier P(arc is true)
 (.track/.nodes/.arcs)  (local AEQD, m)  (≤10 arcs within 60 m)        │
                                                                        ▼
                     predicted route ◄── gap filling ◄── Viterbi over candidates (HMM)
```

## Repository layout

| Folder / file | Contents |
|---|---|
| `src/pathsnap/config.py` | All paths and pipeline constants (radius, K, label threshold, feature list) |
| `src/pathsnap/data_io.py` | Raw-file readers. `load_trip` (prediction input) is kept separate from `load_route` (labels) |
| `src/pathsnap/network.py` | Per-trip local projection, road graph, KD-tree candidate search, bounded Dijkstra |
| `src/pathsnap/labels.py` | Assigns one reference arc to each GPS point by monotone DP alignment with `.route` |
| `src/pathsnap/features.py` | Candidate generation and feature engineering |
| `src/pathsnap/metrics.py` | Accuracy/P/R/F1/log loss, Top-1, candidate miss rate, continuity, route P/R/F1 |
| `src/pathsnap/hmm.py` | HMM definition and Viterbi decoder (handles breaks and gaps) |
| `src/pathsnap/pipeline.py` | End-to-end inference for one trip, route gap filling, server-side evaluation |
| `scripts/01_eda.py` | Dataset exploration: inventory, statistics, plots, Folium maps, data-quality notes |
| `scripts/02_build_features.py` | Builds the candidate/feature tables for train and student_test |
| `scripts/03_train_models.py` | Trains and compares 7 models (GroupKFold CV + student-test evaluation) |
| `scripts/04_bayes_opt.py` | Optuna TPE optimisation of the selected model, exports the final model, out-of-fold predictions |
| `scripts/05_hmm.py` | Tunes the decoder on training OOF predictions; compares classifier-only and HMM; trip maps |
| `scripts/06_collect_results.py` | Collects every result table into `results/RESULTS_SUMMARY.md` |
| `scripts/07_report_assets.py` | Train/CV/test fit diagnostics, feature importance, extra EDA figures |
| `scripts/08_export_model_zoo.py` | Refits all 7 compared models and saves them to `models/zoo/` so the API can serve any of them |
| `scripts/09_evaluator_run.py` | Evaluation day only: runs the deployed pipeline on the evaluator trips → `results/evaluator/` + maps (inference only, nothing is tuned) |
| `report/` | Technical report (`report.html` → `PathSnap_Report.pdf` via `build_pdf.sh`) and its screenshots |
| `docs/VIVA_PREP.md` | Viva preparation: likely questions and answers, with pointers to the code |
| `api/main.py` | FastAPI backend (Laptop 1) |
| `ui/index.html` | Evaluator UI (Laptop 2): static page using Leaflet |
| `models/final_model.joblib` | Exported classifier and feature list |
| `configs/hmm_config.json` | Tuned decoder parameters |
| `results/` | CSV results, experiment logs, plots and Folium HTML maps from the scripts |

## Setup

Python 3.12. On macOS, LightGBM/XGBoost also need OpenMP: `brew install libomp`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Data layout (not committed):

```
data/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
data/student_test/<trip_id>/...
data/evaluator_test/<trip_id>/...      # only on Laptop 1 at evaluation time
```

`data/train` and `data/student_test` can be symlinks to the folders handed out.
The data root can be overridden with `PATHSNAP_DATA_DIR`.

## Reproducing the experiments

```bash
python scripts/01_eda.py              # results/eda/
python scripts/02_build_features.py   # data_cache/*.parquet, results/features/
python scripts/03_train_models.py     # results/models/   (~30 min on 8 cores)
python scripts/04_bayes_opt.py 30     # results/bayes_opt/, models/final_model.joblib
python scripts/05_hmm.py              # results/hmm/, results/maps/, configs/hmm_config.json
```

The random seed is fixed (`config.SEED = 42`). Every number in the report comes
from the CSV/JSON files these scripts write.

## Deployment (two laptops, same Wi-Fi)

**Laptop 1: model + API**

```bash
source .venv/bin/activate
PATHSNAP_EVAL_DIR=/path/to/evaluator_test uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- If `PATHSNAP_EVAL_DIR` is not set, the API serves the student-test trips in demo mode, and `/health` says so.
- Find the laptop's IP with `ipconfig getifaddr en0` (macOS) or `ipconfig` (Windows).
- Allow port 8000 through the firewall.
- Every request and its result appear in this terminal (`PREDICT …` / `RESULT …` lines). These are the API logs to show the evaluator.

**Laptop 2: UI**

```bash
python3 ui/serve.py            # http://0.0.0.0:8501
```

`ui/serve.py` serves the page and also proxies OpenStreetMap tiles through
`/tiles/{z}/{x}/{y}.png`. This matters on campus networks: the browser does not trust
the TLS-inspection certificate, so public CDNs and HTTPS tile servers fail with
`ERR_CERT_AUTHORITY_INVALID`. Leaflet itself is vendored in `ui/vendor/leaflet` (no
CDN needed); the tile proxy fetches tiles server-side and caches them under
`$TMPDIR/pathsnap_tiles`. Plain `python3 -m http.server` also works, but the map
background stays blank.

Open `http://localhost:8501/?api=http://<LAPTOP1_IP>:8000`, or type the API URL in the
header and press **Connect**. The page lists the evaluation trips. Clicking one runs
`POST /predict` with the deployed model (tuned XGBoost + HMM) and draws the GPS points
and the predicted route, with the accuracy the API computed on the server. It also calls
`POST /compare` and shows every model's score on that trip **for comparison only**. That
ranking is scored against the reference route, so it never chooses which prediction is
displayed.

API endpoints:

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Model name, decoder configuration and the trip source being served |
| GET | `/trips` | Trip IDs and point counts (no labels) |
| GET | `/models` | Every selectable model (all 7 compared models + the deployed tuned XGBoost) with its offline CV/student-test scores |
| POST | `/predict` | Body `{"trip_id": "...", "model": "<id>", "use_hmm": true}`. `model` and `use_hmm` are optional; the default is the deployed tuned XGBoost with the HMM, which is what the UI calls. Returns GPS points, predicted route polylines, metrics and timings |
| POST | `/compare` | Body `{"trip_id": "...", "use_hmm": true}`. Runs every model on one trip (features computed once) and returns a ranked table of server-side metrics |

Model IDs: `xgboost_tuned` (default), `xgboost`, `lightgbm`, `random_forest`, `mlp`,
`logistic_regression`, `gaussian_nb`, `knn`. The files live in `models/zoo/` and are
produced by `scripts/08_export_model_zoo.py`. The evaluator can try any model from the
browser at `http://<LAPTOP1_IP>:8000/docs` (Swagger UI → *Try it out*). The HMM
parameters were tuned for the deployed XGBoost and are reused unchanged for the other models.

## Leakage controls

- Training, CV and Bayesian optimisation use only the 80 training trips, with folds grouped by trip.
- Model comparison uses the 10 student-test trips.
- HMM parameters are tuned on out-of-fold training predictions.
- The evaluator trips are never read by any training script.
- The predictor (`PathSnapModel.predict_trip`) reads only `.track`, `.nodes` and `.arcs`.
- `.route` is read only in `evaluate_prediction`, after prediction. Only aggregate metrics leave the server; the UI never receives reference geometry.
- No trip-specific values are hard-coded anywhere.

## Key design decisions (details in the report)

- **Per-trip projection.** Trips span the globe (longitude −100…137), so each trip uses an azimuthal-equidistant projection centred on its median GPS point instead of one UTM zone.
- **Binary candidate classification.** Arc IDs only mean something within one trip, so the model scores candidate arcs ("is this the true arc?") rather than predicting arc IDs.
- **Label derivation.** `.route` lists the arcs driven but not which arc each point belongs to. Monotone DP alignment assigns one; labels more than 60 m from their arc are marked unreliable.
- **Excluded trips.** Two training trips (00000054, 00000033) are excluded from training because their GPS mostly lies more than 100 m from the reference route.
- **HMM.** Emissions come from the classifier probabilities. Transitions follow Newson & Krumm (2009), using the mismatch between network distance and straight-line distance. The chain is cut on large gaps or when no transition is possible, and the route is gap-filled with shortest paths.

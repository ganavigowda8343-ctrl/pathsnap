"""PathSnap FastAPI backend (runs on Laptop 1).

Endpoints
  GET  /health            model + decoder info, which trip directory is served
  GET  /trips             ids of the evaluation trips (no labels)
  POST /predict           {"trip_id": "..."} -> predicted route + server-side evaluation

Ground truth (.route) is only read inside `evaluate_prediction` AFTER the
prediction is finished; it is never returned to the client.  Only aggregate
accuracy numbers leave the server.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
      PATHSNAP_EVAL_DIR=/path/to/evaluator_test uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap import config as C
from pathsnap.data_io import list_trips
from pathsnap.pipeline import PathSnapModel, evaluate_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap.api")

STATE = {}


def trip_dir():
    if C.EVAL_DIR.exists() and list_trips(C.EVAL_DIR):
        return C.EVAL_DIR, "evaluation"
    return C.STUDENT_TEST_DIR, "demo (student_test - set PATHSNAP_EVAL_DIR for evaluation)"


@asynccontextmanager
async def lifespan(app):
    STATE["model"] = PathSnapModel()
    d, mode = trip_dir()
    STATE["dir"], STATE["mode"] = d, mode
    log.info("Loaded model '%s' | HMM %s", STATE["model"].model_name, STATE["model"].hmm_cfg.to_dict())
    log.info("Serving %d trips from %s [%s]", len(list_trips(d)), d, mode)
    yield


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    client = request.client.host if request.client else "?"
    log.info("%s %s from %s -> %d (%.0f ms)", request.method, request.url.path, client,
             resp.status_code, 1000 * (time.time() - t0))
    return resp


class PredictRequest(BaseModel):
    trip_id: str


def lonlat_to_latlon(arr):
    return [[round(float(lat), 6), round(float(lon), 6)] for lon, lat in arr]


def route_polylines(route_arcs, nodes, arcs):
    """Merge the arc sequence into continuous polylines ([[lat,lon],...])."""
    lines, cur = [], []
    for a in route_arcs:
        u, v = arcs[a]
        if cur and cur_end == u:
            cur.append(v)
        else:
            if cur:
                lines.append(cur)
            cur = [u, v]
        cur_end = v
    if cur:
        lines.append(cur)
    return [lonlat_to_latlon(nodes[l]) for l in lines]


@app.get("/health")
def health():
    m = STATE["model"]
    return {"status": "ok", "model": m.model_name, "hmm_config": m.hmm_cfg.to_dict(),
            "trip_source": STATE["mode"], "n_trips": len(list_trips(STATE["dir"]))}


@app.get("/trips")
def trips():
    d = STATE["dir"]
    out = []
    for t in list_trips(d):
        n = sum(1 for _ in open(d / t / f"{t}.track"))
        out.append({"trip_id": t, "n_points": n})
    return {"source": STATE["mode"], "trips": out}


@app.post("/predict")
def predict(req: PredictRequest):
    d = STATE["dir"]
    if req.trip_id not in list_trips(d):  # also prevents path traversal
        raise HTTPException(404, f"unknown trip_id {req.trip_id}")
    tdir = d / req.trip_id
    log.info("PREDICT trip=%s : running pipeline (features -> classifier -> Viterbi)", req.trip_id)
    res = STATE["model"].predict_trip(tdir)
    # --- server-side evaluation (reference route never leaves the server) ---
    ev = evaluate_prediction(tdir, res)
    trip = res["trip"]
    track = trip["track"]
    step = max(1, len(track) // 3000)
    metrics = {
        "top1_accuracy_hmm": ev["hmm"]["top1_accuracy"],
        "top1_accuracy_classifier_only": ev["clf"]["top1_accuracy"],
        "route_precision": ev["hmm_route"]["route_precision"],
        "route_recall": ev["hmm_route"]["route_recall"],
        "route_f1": ev["hmm_route"]["route_f1"],
        "route_continuity": ev["hmm_route"]["route_continuity"],
        "pointwise_continuity_hmm": ev["hmm"]["route_continuity"],
        "pointwise_continuity_classifier_only": ev["clf"]["route_continuity"],
        "n_points": ev["n_points"], "n_labelled_points": ev["n_labelled"],
        "n_unmatched_points": ev["n_unmatched_points"],
        "n_predicted_route_arcs": ev["hmm_route"]["n_route_arcs"],
        "hmm_chain_breaks": res["hmm_breaks"], "unfilled_route_gaps": res["route_unfilled_gaps"],
    }
    metrics = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in metrics.items()}
    log.info("RESULT trip=%s top1(HMM)=%.4f top1(clf)=%.4f routeF1=%.4f continuity=%.4f time=%.2fs",
             req.trip_id, metrics["top1_accuracy_hmm"], metrics["top1_accuracy_classifier_only"],
             metrics["route_f1"], metrics["route_continuity"], res["timings"]["total_s"])
    return {
        "trip_id": req.trip_id,
        "model": STATE["model"].model_name,
        "gps_points": lonlat_to_latlon(track[::step, :2]),
        "predicted_route": route_polylines(res["route_arcs"], trip["nodes"], trip["arcs"]),
        "metrics": metrics,
        "timings_s": {k: round(v, 3) for k, v in res["timings"].items()},
    }

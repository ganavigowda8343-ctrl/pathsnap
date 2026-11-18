"""PathSnap FastAPI backend (runs on Laptop 1).

Endpoints
  GET  /health            deployed model + decoder info, which trip directory is served
  GET  /trips             ids of the evaluation trips (no labels)
  GET  /models            every selectable model with its offline (CV / student-test) scores
  POST /predict           {"trip_id": "...", "model": "<id>" (optional), "use_hmm": true (optional)}
                          -> predicted route + server-side evaluation.
                          Without "model" the deployed tuned XGBoost is used (what the UI calls).
  POST /compare           {"trip_id": "...", "use_hmm": true} -> runs EVERY model on that trip and
                          returns one row of metrics per model (features computed once).

Ground truth (.route) is only read AFTER prediction, inside the evaluation
functions; it is never returned to the client. Only aggregate accuracy numbers
leave the server. Interactive docs (model picker for the evaluator): /docs

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
      PATHSNAP_EVAL_DIR=/path/to/evaluator_test uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap import config as C
from pathsnap.data_io import list_trips
from pathsnap.pipeline import (featurize, decode, load_hmm_config, reference_labels,
                               evaluate_prediction)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap.api")

STATE = {"bundles": {}}
DEFAULT_MODEL = "xgboost_tuned"
ZOO = C.MODELS_DIR / "zoo"


def trip_dir():
    if C.EVAL_DIR.exists() and list_trips(C.EVAL_DIR):
        return C.EVAL_DIR, "evaluation"
    return C.STUDENT_TEST_DIR, "demo (student_test - set PATHSNAP_EVAL_DIR for evaluation)"


def model_registry():
    """id -> metadata for every model whose file exists. Falls back to final_model.joblib."""
    reg = {}
    idx = ZOO / "index.json"
    if idx.exists():
        for m in json.loads(idx.read_text()):
            path = ZOO / f"{m['id']}.joblib"
            reg[m["id"]] = {**m, "path": path, "available": path.exists()}
    if DEFAULT_MODEL not in reg or not reg[DEFAULT_MODEL]["available"]:
        reg[DEFAULT_MODEL] = {"id": DEFAULT_MODEL, "name": "XGBoost (Bayesian-tuned) - deployed",
                              "family": "Boosting", "default": True, "available": True,
                              "path": C.MODELS_DIR / "final_model.joblib"}
    return reg


def get_bundle(model_id):
    """Lazy-load and cache a model bundle."""
    reg = STATE["registry"]
    if model_id not in reg:
        raise HTTPException(404, f"unknown model '{model_id}'. Available: {sorted(reg)}")
    if not reg[model_id]["available"]:
        raise HTTPException(409, f"model '{model_id}' file is missing on this server "
                                 f"(run scripts/08_export_model_zoo.py)")
    if model_id not in STATE["bundles"]:
        t0 = time.time()
        STATE["bundles"][model_id] = joblib.load(reg[model_id]["path"])
        log.info("Loaded model '%s' in %.1fs", model_id, time.time() - t0)
    return STATE["bundles"][model_id]


@asynccontextmanager
async def lifespan(app):
    STATE["registry"] = model_registry()
    STATE["hmm_cfg"] = load_hmm_config()
    get_bundle(DEFAULT_MODEL)  # deployed model loaded eagerly; others on first use
    d, mode = trip_dir()
    STATE["dir"], STATE["mode"] = d, mode
    avail = [k for k, v in STATE["registry"].items() if v["available"]]
    log.info("Default model '%s' | HMM %s", DEFAULT_MODEL, STATE["hmm_cfg"].to_dict())
    log.info("Selectable models (%d): %s", len(avail), ", ".join(avail))
    log.info("Serving %d trips from %s [%s]", len(list_trips(d)), d, mode)
    yield


app = FastAPI(title="PathSnap API", version="1.1", lifespan=lifespan)
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
    model: Optional[str] = None   # model id from GET /models; default = deployed tuned XGBoost
    use_hmm: bool = True


class CompareRequest(BaseModel):
    trip_id: str
    use_hmm: bool = True


def lonlat_to_latlon(arr):
    return [[round(float(lat), 6), round(float(lon), 6)] for lon, lat in arr]


def route_polylines(route_arcs, nodes, arcs):
    """Merge the arc sequence into continuous polylines ([[lat,lon],...])."""
    lines, cur, cur_end = [], [], None
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


def r4(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 4)
    return v


def checked_trip(trip_id):
    d = STATE["dir"]
    if trip_id not in list_trips(d):  # also prevents path traversal
        raise HTTPException(404, f"unknown trip_id {trip_id}")
    return d / trip_id


def summarize(ev, res):
    used = res["used_hmm"]
    return {k: r4(v) for k, v in {
        "top1_accuracy_hmm": ev["hmm"]["top1_accuracy"] if used else None,
        "top1_accuracy_classifier_only": ev["clf"]["top1_accuracy"],
        "route_precision": ev["hmm_route"]["route_precision"],
        "route_recall": ev["hmm_route"]["route_recall"],
        "route_f1": ev["hmm_route"]["route_f1"],
        "route_continuity": ev["hmm_route"]["route_continuity"],
        "pointwise_continuity_hmm": ev["hmm"]["route_continuity"] if used else None,
        "pointwise_continuity_classifier_only": ev["clf"]["route_continuity"],
        "n_points": ev["n_points"], "n_labelled_points": ev["n_labelled"],
        "n_unmatched_points": ev["n_unmatched_points"],
        "n_predicted_route_arcs": ev["hmm_route"]["n_route_arcs"],
        "hmm_chain_breaks": res["hmm_breaks"], "unfilled_route_gaps": res["route_unfilled_gaps"],
    }.items()}


@app.get("/health")
def health():
    reg = STATE["registry"]
    return {"status": "ok", "model": reg[DEFAULT_MODEL]["name"], "default_model_id": DEFAULT_MODEL,
            "selectable_models": [k for k, v in reg.items() if v["available"]],
            "hmm_config": STATE["hmm_cfg"].to_dict(),
            "trip_source": STATE["mode"], "n_trips": len(list_trips(STATE["dir"]))}


@app.get("/trips")
def trips():
    d = STATE["dir"]
    out = []
    for t in list_trips(d):
        n = sum(1 for _ in open(d / t / f"{t}.track"))
        out.append({"trip_id": t, "n_points": n})
    return {"source": STATE["mode"], "trips": out}


@app.get("/models")
def models():
    """All selectable models with the offline scores recorded during development
    (CV on train trips, evaluation on the 10 student-test trips)."""
    rows = []
    for k, v in STATE["registry"].items():
        rows.append({"id": k, "name": v["name"], "family": v.get("family"), "default": k == DEFAULT_MODEL,
                     "available": v["available"], "config": v.get("config"),
                     "offline_scores": v.get("offline")})
    return {"default": DEFAULT_MODEL, "hmm_config_tuned_for": DEFAULT_MODEL, "models": rows}


@app.post("/predict")
def predict(req: PredictRequest):
    tdir = checked_trip(req.trip_id)
    model_id = req.model or DEFAULT_MODEL
    bundle = get_bundle(model_id)
    log.info("PREDICT trip=%s model=%s hmm=%s : running pipeline", req.trip_id, model_id, req.use_hmm)
    res = decode(featurize(tdir), bundle["model"], bundle["features"], STATE["hmm_cfg"], req.use_hmm)
    ev = evaluate_prediction(tdir, res)  # server-side, after prediction
    metrics = summarize(ev, res)
    trip, track = res["trip"], res["trip"]["track"]
    step = max(1, len(track) // 3000)
    log.info("RESULT trip=%s model=%s top1(HMM)=%s top1(clf)=%.4f routeF1=%.4f continuity=%.4f time=%.2fs",
             req.trip_id, model_id, metrics["top1_accuracy_hmm"], metrics["top1_accuracy_classifier_only"],
             metrics["route_f1"], metrics["route_continuity"] or 0.0, res["timings"]["total_s"])
    return {
        "trip_id": req.trip_id,
        "model": STATE["registry"][model_id]["name"],
        "model_id": model_id,
        "used_hmm": req.use_hmm,
        "gps_points": lonlat_to_latlon(track[::step, :2]),
        "predicted_route": route_polylines(res["route_arcs"], trip["nodes"], trip["arcs"]),
        "metrics": metrics,
        "timings_s": {k: round(v, 3) for k, v in res["timings"].items()},
    }


@app.post("/compare")
def compare(req: CompareRequest):
    """Run every available model on one trip; features and reference labels computed once."""
    tdir = checked_trip(req.trip_id)
    log.info("COMPARE trip=%s hmm=%s : featurising once, scoring all models", req.trip_id, req.use_hmm)
    feat = featurize(tdir)
    reference = reference_labels(tdir, feat["prep"])  # server-side only
    rows = []
    for model_id, meta in STATE["registry"].items():
        if not meta["available"]:
            rows.append({"model_id": model_id, "name": meta["name"], "available": False})
            continue
        b = get_bundle(model_id)
        res = decode(feat, b["model"], b["features"], STATE["hmm_cfg"], req.use_hmm)
        ev = evaluate_prediction(tdir, res, reference)
        m = summarize(ev, res)
        row = {"model_id": model_id, "name": meta["name"], "family": meta.get("family"), "available": True,
               "top1_accuracy_classifier_only": m["top1_accuracy_classifier_only"],
               "top1_accuracy_hmm": m["top1_accuracy_hmm"], "route_f1": m["route_f1"],
               "route_precision": m["route_precision"], "route_recall": m["route_recall"],
               "route_continuity": m["route_continuity"],
               "time_s": round(res["timings"]["classifier_s"] + res["timings"]["viterbi_s"]
                               + res["timings"]["route_fill_s"], 3)}
        rows.append(row)
        log.info("  %-20s top1(clf)=%.4f top1(HMM)=%s routeF1=%.4f", model_id,
                 row["top1_accuracy_classifier_only"], row["top1_accuracy_hmm"], row["route_f1"])
    key = "top1_accuracy_hmm" if req.use_hmm else "top1_accuracy_classifier_only"
    rows.sort(key=lambda r: -(r.get(key) or -1))
    return {"trip_id": req.trip_id, "used_hmm": req.use_hmm, "ranked_by": key,
            "features_s": round(feat["timings"]["features_s"] + feat["timings"]["load_and_index_s"], 3),
            "models": rows}

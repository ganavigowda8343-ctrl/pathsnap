"""End-to-end inference for one trip (used by the API and the HMM experiments).

The predictor sees ONLY .track/.nodes/.arcs. Reference labels are computed by
`evaluate_prediction`, which is called separately (server-side) after prediction.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from . import config as C
from .data_io import load_trip, load_route
from .features import prepare_trip, build_candidate_table
from .hmm import HMMConfig, viterbi_decode
from .labels import derive_labels
from .metrics import top1_accuracy, continuity, route_prf


def load_hmm_config(hmm_path=None):
    hp = Path(hmm_path or C.CONFIGS_DIR / "hmm_config.json")
    return HMMConfig(**json.loads(hp.read_text())) if hp.exists() else HMMConfig()


def featurize(trip_dir):
    """Stage 1 (model-independent): load trip, project, build candidates + features."""
    t0 = time.time()
    trip = load_trip(trip_dir)
    prep = prepare_trip(trip)
    t1 = time.time()
    cand = build_candidate_table(prep)
    return dict(trip=trip, prep=prep, cand=cand,
                timings={"load_and_index_s": t1 - t0, "features_s": time.time() - t1})


def decode(feat, model, features, hmm_cfg, use_hmm=True):
    """Stage 2: score candidates with `model`, then (optionally) HMM/Viterbi + gap filling.

    Without the HMM the route is simply the sequence of per-point argmax arcs (no
    gap filling), so the classifier-only result is shown as it really is.
    """
    prep, cand = feat["prep"], feat["cand"]
    timings = dict(feat["timings"])
    t1 = time.time()
    prob = model.predict_proba(cand[features].to_numpy(np.float32))[:, 1]
    timings["classifier_s"] = time.time() - t1
    n = len(prep["P"])
    clf_pred = np.full(n, -1, dtype=np.int64)
    order = np.lexsort((-prob, cand["obs"].to_numpy()))
    obs_sorted = cand["obs"].to_numpy()[order]
    first = np.r_[True, obs_sorted[1:] != obs_sorted[:-1]]
    clf_pred[obs_sorted[first]] = cand["arc"].to_numpy()[order][first]
    if use_hmm:
        t1 = time.time()
        hmm_pred, breaks = viterbi_decode(cand, prob, prep["P"], prep["t"], prep["net"], hmm_cfg)
        timings["viterbi_s"] = time.time() - t1
        t1 = time.time()
        route_arcs, n_unfilled = fill_route(collapse_sequence(hmm_pred), prep["net"])
        timings["route_fill_s"] = time.time() - t1
    else:
        hmm_pred, breaks, n_unfilled = clf_pred, 0, 0
        route_arcs = collapse_sequence(clf_pred)
        timings["viterbi_s"] = timings["route_fill_s"] = 0.0
    timings["total_s"] = sum(timings.values())
    return dict(trip=feat["trip"], prep=prep, cand=cand, prob=prob, clf_pred=clf_pred,
                hmm_pred=hmm_pred, hmm_breaks=breaks, route_arcs=route_arcs, used_hmm=use_hmm,
                route_unfilled_gaps=n_unfilled, timings=timings)


class PathSnapModel:
    def __init__(self, model_path=None, hmm_path=None):
        bundle = joblib.load(model_path or C.MODELS_DIR / "final_model.joblib")
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.model_name = bundle.get("name", "model")
        self.hmm_cfg = load_hmm_config(hmm_path)

    def predict_trip(self, trip_dir, use_hmm=True):
        return decode(featurize(trip_dir), self.model, self.features, self.hmm_cfg, use_hmm)


def collapse_sequence(pred):
    """Ordered list of distinct consecutive arcs."""
    p = pred[pred >= 0]
    if len(p) == 0:
        return p
    return p[np.r_[True, p[1:] != p[:-1]]]


def fill_route(seq, net, cutoff=3000.0):
    """Insert the shortest network path between consecutive predicted arcs that are
    not directly connected (e.g. a short arc crossed between two 1 Hz samples, or a
    GPS gap). Gaps that cannot be bridged within `cutoff` metres are left open."""
    if len(seq) == 0:
        return np.array([], dtype=np.int64), 0
    out, unfilled = [int(seq[0])], 0
    for b in seq[1:]:
        a = out[-1]; b = int(b)
        if net.arcs[a, 1] != net.arcs[b, 0]:
            path = net.path_arcs(int(net.arcs[a, 1]), int(net.arcs[b, 0]), cutoff)
            if path is None:
                unfilled += 1
            else:
                out.extend(path)
        out.append(b)
    return np.array(out, dtype=np.int64), unfilled


def route_continuity(route_arcs, arcs):
    if len(route_arcs) < 2:
        return np.nan
    return float(np.mean(arcs[route_arcs[:-1], 1] == arcs[route_arcs[1:], 0]))


def reference_labels(trip_dir, prep):
    """SERVER-SIDE ONLY: reference route + per-point reference arc (-1 = unreliable)."""
    route = load_route(trip_dir)
    label, _ = derive_labels(prep["P"], route, prep["net"])
    return route, label


def evaluate_prediction(trip_dir, result, reference=None):
    """SERVER-SIDE ONLY: compare predictions with the reference route of this trip.
    `reference` = output of reference_labels(), to avoid recomputing it per model."""
    prep = result["prep"]
    route, label = reference if reference is not None else reference_labels(trip_dir, prep)
    arcs = prep["net"].arcs
    out = {}
    for key in ("clf_pred", "hmm_pred"):
        pred = result[key]
        m = pred >= 0
        pr, rc, f1 = route_prf(collapse_sequence(pred), route)
        out[key.replace("_pred", "")] = dict(
            top1_accuracy=top1_accuracy(pred, label),
            route_continuity=continuity(pred[m], arcs[pred[m], 0], arcs[pred[m], 1]),
            route_precision=pr, route_recall=rc, route_f1=f1,
        )
    pr, rc, f1 = route_prf(result["route_arcs"], route)
    out["hmm_route"] = dict(route_precision=pr, route_recall=rc, route_f1=f1,
                            route_continuity=route_continuity(result["route_arcs"], arcs),
                            n_route_arcs=int(len(result["route_arcs"])), n_reference_arcs=int(len(route)))
    out["n_points"] = int(len(label))
    out["n_labelled"] = int((label >= 0).sum())
    out["n_unmatched_points"] = int((result["hmm_pred"] < 0).sum())
    return out

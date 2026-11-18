"""Evaluation metrics.

Candidate-level (binary "is this the true arc?"):
    accuracy, precision, recall, macro F1, log loss, confusion matrix
Observation-level:
    Top-1 accuracy  - argmax candidate == reference arc (over labelled obs;
                      obs whose reference arc is not in the candidate set count as wrong)
    Candidate miss  - fraction of labelled obs whose reference arc is absent from candidates
Sequence-level:
    Route continuity - fraction of consecutive predictions that are the same arc
                       or directly connected (arc_a.to == arc_b.from)
    Route P/R/F1     - set overlap between predicted arc sequence and the .route arcs
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             log_loss, confusion_matrix)


def candidate_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return dict(
        accuracy=accuracy_score(y, yhat),
        precision_macro=precision_score(y, yhat, average="macro", zero_division=0),
        recall_macro=recall_score(y, yhat, average="macro", zero_division=0),
        f1_macro=f1_score(y, yhat, average="macro", zero_division=0),
        precision_pos=precision_score(y, yhat, zero_division=0),
        recall_pos=recall_score(y, yhat, zero_division=0),
        log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
        confusion=confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    )


def argmax_per_obs(df, score):
    """Pick the best-scoring candidate per (trip, obs). Returns one row per observation."""
    d = df[["trip", "obs", "arc", "u", "v", "label_arc"]].copy()
    d["s"] = score
    idx = d.sort_values(["trip", "obs", "s"], ascending=[True, True, False]).groupby(["trip", "obs"], sort=False).head(1).index
    best = d.loc[idx].sort_values(["trip", "obs"]).reset_index(drop=True)
    return best.rename(columns={"arc": "pred_arc"})


def top1_accuracy(pred_arc, label_arc):
    m = label_arc >= 0
    return float(np.mean(pred_arc[m] == label_arc[m])) if m.any() else np.nan


def candidate_miss_rate(df):
    lab = df[df.label_arc >= 0]
    hit = lab.groupby(["trip", "obs"])["y"].max()
    return float(1 - hit.mean())


def continuity(pred_arc, u, v):
    """pred_arc/u/v: arrays for consecutive observations of ONE trip (u,v = endpoints of pred arc)."""
    if len(pred_arc) < 2:
        return np.nan
    same = pred_arc[1:] == pred_arc[:-1]
    conn = v[:-1] == u[1:]
    return float(np.mean(same | conn))


def route_prf(pred_arcs, route):
    P, R = set(np.asarray(pred_arcs).tolist()), set(np.asarray(route).tolist())
    tp = len(P & R)
    prec = tp / len(P) if P else 0.0
    rec = tp / len(R) if R else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def sequence_metrics(best, routes):
    """best: output of argmax_per_obs (possibly multiple trips); routes: dict trip -> route array."""
    rows = []
    for trip, g in best.groupby("trip", sort=False):
        pr, rc, f1 = route_prf(g.pred_arc.to_numpy(), routes[trip]) if trip in routes else (np.nan,) * 3
        rows.append(dict(trip=trip,
                         top1=top1_accuracy(g.pred_arc.to_numpy(), g.label_arc.to_numpy()),
                         continuity=continuity(g.pred_arc.to_numpy(), g.u.to_numpy(), g.v.to_numpy()),
                         route_precision=pr, route_recall=rc, route_f1=f1))
    per_trip = pd.DataFrame(rows)
    overall = dict(
        top1=top1_accuracy(best.pred_arc.to_numpy(), best.label_arc.to_numpy()),
        continuity=float(per_trip.continuity.mean()),
        route_precision=float(per_trip.route_precision.mean()),
        route_recall=float(per_trip.route_recall.mean()),
        route_f1=float(per_trip.route_f1.mean()),
    )
    return overall, per_trip

"""Extra evidence for the report.

1. Over/under-fitting diagnostics: every model from 03_train_models.py is refit on
   the full training partition with the same configuration, then scored on the
   SAME training data, so training scores can be compared with CV and
   student-test scores.  -> results/models/train_vs_cv_vs_test.csv + plot
2. Feature importance of the deployed model (XGBoost gain). -> results/models/feature_importance.*
3. Static road-network figure for one training trip (network, reference route,
   GPS). -> results/eda/network_trip_00000001.png
"""
import sys, importlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from pathsnap import config as C
from pathsnap.experiment import load_partition
from pathsnap.metrics import candidate_metrics, argmax_per_obs, top1_accuracy
from pathsnap.data_io import load_trip, load_route
from pathsnap.features import prepare_trip

T = importlib.import_module("03_train_models")
OUT = C.RESULTS_DIR / "models"


def fit_diagnostics():
    train, _, _ = load_partition("train")
    comp = pd.read_csv(OUT / "model_comparison.csv").set_index("model")
    rows = []
    for name in T.MODELS:
        m, _ = T.fit_model(name, train)
        p, _ = T.predict(m, train)
        cm = candidate_metrics(train.y.to_numpy(), p)
        b = argmax_per_obs(train, p)
        rows.append(dict(model=name, train_top1=top1_accuracy(b.pred_arc.to_numpy(), b.label_arc.to_numpy()),
                         train_f1_macro=cm["f1_macro"], train_log_loss=cm["log_loss"],
                         cv_top1=comp.loc[name, "cv_top1_mean"], cv_f1_macro=comp.loc[name, "cv_f1_macro_mean"],
                         cv_log_loss=comp.loc[name, "cv_log_loss_mean"],
                         test_top1=comp.loc[name, "test_top1"], test_f1_macro=comp.loc[name, "test_f1_macro"],
                         test_log_loss=comp.loc[name, "test_log_loss"]))
        rows[-1]["gap_train_minus_cv_top1"] = rows[-1]["train_top1"] - rows[-1]["cv_top1"]
        print(name, {k: round(v, 4) for k, v in rows[-1].items() if k != "model"}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "train_vs_cv_vs_test.csv", index=False)
    x = np.arange(len(df)); w = 0.27
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
    for i, (col, lab) in enumerate([("train_top1", "train (fit data)"), ("cv_top1", "5-fold CV"), ("test_top1", "student-test")]):
        ax[0].bar(x + (i - 1) * w, df[col], w, label=lab)
    ax[0].set_xticks(x, df.model, rotation=25); ax[0].set_ylim(0.8, 1.0); ax[0].legend()
    ax[0].set_title("Top-1 accuracy: training vs CV vs student-test")
    for i, (col, lab) in enumerate([("train_log_loss", "train"), ("cv_log_loss", "CV"), ("test_log_loss", "student-test")]):
        ax[1].bar(x + (i - 1) * w, df[col], w, label=lab)
    ax[1].set_xticks(x, df.model, rotation=25); ax[1].set_yscale("log"); ax[1].legend()
    ax[1].set_title("Log loss (log scale): training vs CV vs student-test")
    plt.tight_layout(); plt.savefig(OUT / "train_vs_cv_vs_test.png", dpi=130); plt.close()


def feature_importance():
    b = joblib.load(C.MODELS_DIR / "final_model.joblib")
    booster = b["model"].get_booster()
    gain = booster.get_score(importance_type="gain")
    imp = pd.Series({f: gain.get(f"f{i}", gain.get(f, 0.0)) for i, f in enumerate(b["features"])})
    imp = (imp / imp.sum()).sort_values()
    imp.rename("relative_gain").to_csv(OUT / "feature_importance.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(imp.index, imp.values, color="#3949ab")
    ax.set(title=f"Feature importance - {b['name']} (relative gain)", xlabel="share of total gain")
    plt.tight_layout(); plt.savefig(OUT / "feature_importance.png", dpi=130); plt.close()
    print(imp.sort_values(ascending=False).round(4).head(10))


def network_figure(tid="00000027"):
    d = C.TRAIN_DIR / tid
    trip = load_trip(d); route = load_route(d); prep = prepare_trip(trip)
    net, P = prep["net"], prep["P"]
    near = np.unique(net.sample_arc)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.add_collection(LineCollection(np.stack([net.A[near], net.B[near]], 1), colors="#bdbdbd", linewidths=0.6,
                                     label="road network (arcs near track)"))
    ax.add_collection(LineCollection(np.stack([net.A[route], net.B[route]], 1), colors="#2e7d32", linewidths=3,
                                     label="reference route (.route)"))
    ax.scatter(P[:, 0], P[:, 1], s=2, c="#1565c0", label="GPS observations", zorder=3)
    ax.set_aspect("equal"); ax.autoscale()
    ax.set(title=f"Train trip {tid}: network, reference route and GPS (local AEQD metres)", xlabel="x (m)", ylabel="y (m)")
    ax.legend(loc="upper right", markerscale=4)
    plt.tight_layout(); plt.savefig(C.RESULTS_DIR / "eda" / f"network_trip_{tid}.png", dpi=130); plt.close()


def bad_trip_zoom(tid="00000054", half=600.0):
    """Zoom on a section of an excluded trip where GPS and reference route disagree."""
    from pathsnap.network import point_segment
    d = C.TRAIN_DIR / tid
    trip = load_trip(d); route = load_route(d); prep = prepare_trip(trip)
    net, P = prep["net"], prep["P"]
    dmin = np.full(len(P), np.inf)
    for s in range(0, len(route), 300):
        r = route[s:s + 300]
        dd, _, _ = point_segment(P[:, None, :], net.A[r][None], net.B[r][None])
        dmin = np.minimum(dmin, dd.min(1))
    c = P[np.argsort(np.abs(dmin - np.median(dmin)))[0]]
    near = np.unique(net.sample_arc)
    box = lambda X: (np.abs(X[:, 0] - c[0]) < half) & (np.abs(X[:, 1] - c[1]) < half)
    sel = near[box(net.A[near]) | box(net.B[near])]
    rs = route[box(net.A[route]) | box(net.B[route])]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.add_collection(LineCollection(np.stack([net.A[sel], net.B[sel]], 1), colors="#bdbdbd", linewidths=0.8, label="road network"))
    ax.add_collection(LineCollection(np.stack([net.A[rs], net.B[rs]], 1), colors="#2e7d32", linewidths=3, label="reference route"))
    m = box(P)
    ax.scatter(P[m, 0], P[m, 1], s=6, c="#1565c0", label="GPS", zorder=3)
    ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half); ax.set_aspect("equal")
    ax.set(title=f"Excluded train trip {tid}: median GPS-route distance {np.median(dmin):.0f} m", xlabel="x (m)", ylabel="y (m)")
    ax.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(C.RESULTS_DIR / "eda" / f"bad_trip_{tid}_zoom.png", dpi=130); plt.close()


if __name__ == "__main__":
    bad_trip_zoom()
    network_figure()
    feature_importance()
    fit_diagnostics()

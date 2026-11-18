"""HMM / Viterbi route reconstruction: decoder tuning + classifier-only vs HMM comparison.

1. Decoder parameters (beta, emission weight) are tuned with a grid search on a
   subset of TRAINING trips using OUT-OF-FOLD classifier probabilities (so the
   classifier did not see those trips).  Student-test trips are not used for tuning.
2. The final model + tuned decoder are run on the 10 student-test trips through the
   same pipeline the API uses, and compared with classifier-only (argmax) predictions.
3. Folium maps for every student-test trip are written to results/maps/.
"""
import sys, json, time, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium

from pathsnap import config as C
from pathsnap.data_io import list_trips, load_trip, load_route
from pathsnap.features import prepare_trip
from pathsnap.hmm import HMMConfig, viterbi_decode
from pathsnap.labels import derive_labels
from pathsnap.metrics import top1_accuracy, continuity, route_prf
from pathsnap.pipeline import PathSnapModel, evaluate_prediction, collapse_sequence

OUT = C.RESULTS_DIR / "hmm"; OUT.mkdir(parents=True, exist_ok=True)
MAPS = C.RESULTS_DIR / "maps"; MAPS.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(C.SEED)


def seq_scores(pred, label, arcs, route):
    m = pred >= 0
    pr, rc, f1 = route_prf(collapse_sequence(pred), route)
    return dict(top1=top1_accuracy(pred, label),
                continuity=continuity(pred[m], arcs[pred[m], 0], arcs[pred[m], 1]),
                route_precision=pr, route_recall=rc, route_f1=f1)


def tune_on_train():
    full = pd.read_parquet(C.CACHE_DIR / "train.parquet")
    oof = pd.read_parquet(C.CACHE_DIR / "train_oof.parquet")
    summ = pd.read_csv(C.RESULTS_DIR / "features" / "train_trip_summary.csv", dtype={"trip": str})
    pool = summ[(summ.frac_labelled >= C.MIN_TRIP_LABEL_FRAC) & (summ.n_points <= 6000)].trip.tolist()
    pool = [t for t in pool if t in set(oof.trip)]
    tune_trips = sorted(rng.choice(pool, size=min(16, len(pool)), replace=False).tolist())
    print("tuning trips:", tune_trips)
    cache = []
    for t in tune_trips:
        prep = prepare_trip(load_trip(C.TRAIN_DIR / t))
        cand = full[full.trip == t].merge(oof[oof.trip == t], on=["trip", "obs", "arc"]).sort_values(["obs", "dist"])
        label, _ = derive_labels(prep["P"], load_route(C.TRAIN_DIR / t), prep["net"])
        cache.append((t, prep, cand.reset_index(drop=True), label))
    grid = list(itertools.product([1.0, 2.0, 3.0, 6.0, 12.0, 25.0, 50.0], [0.25, 0.5, 1.0, 2.0]))
    rows = []
    for beta, w in grid:
        cfg = HMMConfig(beta=beta, emission_weight=w)
        correct = total = 0; t0 = time.time()
        for t, prep, cand, label in cache:
            pred, _ = viterbi_decode(cand, cand.p.to_numpy(), prep["P"], prep["t"], prep["net"], cfg)
            m = label >= 0
            correct += (pred[m] == label[m]).sum(); total += m.sum()
        rows.append(dict(beta=beta, emission_weight=w, top1=correct / total, seconds=time.time() - t0))
        print(rows[-1], flush=True)
    # classifier-only reference on the same trips
    correct = total = 0
    for t, prep, cand, label in cache:
        b = cand.sort_values(["obs", "p"], ascending=[True, False]).drop_duplicates("obs")
        pred = np.full(len(label), -1); pred[b.obs.to_numpy()] = b.arc.to_numpy()
        m = label >= 0; correct += (pred[m] == label[m]).sum(); total += m.sum()
    grid_df = pd.DataFrame(rows)
    grid_df["classifier_only_top1"] = correct / total
    grid_df.to_csv(OUT / "decoder_grid_search_train_oof.csv", index=False)
    best = grid_df.sort_values("top1", ascending=False).iloc[0]
    cfg = HMMConfig(beta=float(best.beta), emission_weight=float(best.emission_weight))
    C.CONFIGS_DIR.mkdir(exist_ok=True)
    (C.CONFIGS_DIR / "hmm_config.json").write_text(json.dumps(cfg.to_dict(), indent=1))
    piv = grid_df.pivot(index="beta", columns="emission_weight", values="top1")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(piv.values, cmap="viridis"); plt.colorbar(im)
    ax.set_xticks(range(len(piv.columns)), piv.columns); ax.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.4f}", ha="center", va="center", color="w", fontsize=8)
    ax.set(xlabel="emission weight", ylabel="beta (m)", title="HMM top-1 (train OOF, 16 trips)")
    plt.tight_layout(); plt.savefig(OUT / "decoder_grid.png", dpi=130); plt.close()
    print("selected decoder config:", cfg)
    return cfg


def add_arcs(fg, arcs_ids, nodes, arcs, color, weight, opacity=0.9, tooltip=None):
    for a in arcs_ids:
        u, v = arcs[a]
        folium.PolyLine([[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]], color=color,
                        weight=weight, opacity=opacity, tooltip=tooltip).add_to(fg)


def trip_map(res, route, path, metrics):
    trip = res["trip"]; nodes, arcs, tr = trip["nodes"], trip["arcs"], trip["track"]
    m = folium.Map(location=[float(np.median(tr[:, 1])), float(np.median(tr[:, 0]))], zoom_start=14,
                   tiles="OpenStreetMap")
    groups = {
        "Road network (near track)": ("#bdbdbd", 1, np.unique(res["cand"].arc)),
        "Reference route": ("#2e7d32", 7, np.unique(route)),
        "Classifier-only prediction": ("#fb8c00", 4, np.unique(res["clf_pred"][res["clf_pred"] >= 0])),
        "HMM/Viterbi reconstructed route": ("#d32f2f", 3, np.unique(res["route_arcs"])),
    }
    for name, (col, w, ids) in groups.items():
        fg = folium.FeatureGroup(name=name, show=name != "Classifier-only prediction")
        add_arcs(fg, ids, nodes, arcs, col, w, tooltip=name); fg.add_to(m)
    fg = folium.FeatureGroup(name="Raw GPS observations")
    step = max(1, len(tr) // 1500)
    for lon, lat, t in tr[::step]:
        folium.CircleMarker([lat, lon], radius=2, color="#1565c0", fill=True, tooltip=f"t={t:.0f}s").add_to(fg)
    fg.add_to(m)
    legend = (f"<div style='position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px;"
              f"border:1px solid #999;font-size:12px'><b>Trip {trip['trip_id']}</b><br>"
              f"<span style='color:#1565c0'>&#9679;</span> GPS &nbsp; <span style='color:#2e7d32'>&#9473;</span> reference"
              f"<br><span style='color:#fb8c00'>&#9473;</span> classifier-only "
              f"(top-1 {metrics['clf']['top1_accuracy']:.3f})<br><span style='color:#d32f2f'>&#9473;</span> HMM "
              f"(top-1 {metrics['hmm']['top1_accuracy']:.3f})</div>")
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    m.save(str(path))


def evaluate_student_test(cfg):
    model = PathSnapModel()
    model.hmm_cfg = cfg
    rows = []
    for t in list_trips(C.STUDENT_TEST_DIR):
        d = C.STUDENT_TEST_DIR / t
        res = model.predict_trip(d)
        ev = evaluate_prediction(d, res)
        common = dict(trip=t, n_points=ev["n_points"], hmm_breaks=res["hmm_breaks"],
                      total_time_s=res["timings"]["total_s"], viterbi_s=res["timings"]["viterbi_s"])
        rows.append(dict(method="classifier-only", **common, **ev["clf"]))
        rows.append(dict(method="classifier+HMM", **common, **ev["hmm"]))
        filled = {k: v for k, v in ev["hmm_route"].items() if k.startswith("route_")}
        rows.append(dict(method="classifier+HMM+gap-fill", **common,
                         top1_accuracy=ev["hmm"]["top1_accuracy"], **filled,
                         unfilled_gaps=res["route_unfilled_gaps"]))
        trip_map(res, load_route(d), MAPS / f"student_test_{t}.html", ev)
        print(t, f"clf={ev['clf']['top1_accuracy']:.4f} hmm={ev['hmm']['top1_accuracy']:.4f} "
                 f"cont {ev['clf']['route_continuity']:.3f}->{ev['hmm']['route_continuity']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "student_test_clf_vs_hmm_per_trip.csv", index=False)
    w = df.assign(correct=df.top1_accuracy * df.n_points)
    agg = df.groupby("method")[["top1_accuracy", "route_continuity", "route_precision", "route_recall", "route_f1"]].mean()
    agg["top1_pointwise_pooled"] = w.groupby("method").correct.sum() / w.groupby("method").n_points.sum()
    agg.to_csv(OUT / "student_test_clf_vs_hmm_summary.csv")
    print(agg.round(4).to_string())

    piv = df.pivot(index="trip", columns="method", values="top1_accuracy")
    pc = df.pivot(index="trip", columns="method", values="route_continuity")
    pf = df.pivot(index="trip", columns="method", values="route_f1")
    meths = ["classifier-only", "classifier+HMM", "classifier+HMM+gap-fill"]
    cols = ["#fb8c00", "#d32f2f", "#6a1b9a"]
    fig, ax = plt.subplots(1, 3, figsize=(20, 4.8))
    piv[meths[:2]].plot.bar(ax=ax[0], color=cols[:2])
    ax[0].set(title="Top-1 accuracy per student-test trip", ylim=(max(0, piv[meths[:2]].values.min() - 0.05), 1))
    pc[meths].plot.bar(ax=ax[1], color=cols)
    ax[1].set(title="Route continuity per student-test trip", ylim=(max(0, pc.values.min() - 0.05), 1.01))
    pf[meths].plot.bar(ax=ax[2], color=cols)
    ax[2].set(title="Route arc F1 vs .route per student-test trip", ylim=(max(0, pf.values.min() - 0.05), 1.01))
    plt.tight_layout(); plt.savefig(OUT / "clf_vs_hmm.png", dpi=130); plt.close()


if __name__ == "__main__":
    if "--skip-tune" in sys.argv and (C.CONFIGS_DIR / "hmm_config.json").exists():
        cfg = HMMConfig(**json.loads((C.CONFIGS_DIR / "hmm_config.json").read_text()))
    else:
        cfg = tune_on_train()
    evaluate_student_test(cfg)

"""Train and compare >=5 models from different algorithm families.

Protocol
  * Fit / cross-validate on the TRAIN partition only (GroupKFold by trip, 5 folds,
    so no trip is split across folds).
  * Development comparison on the 10 STUDENT_TEST trips (model refit on all train).
  * Primary selection metric: CV observation-level Top-1 accuracy (it is exactly
    what the decoder consumes: "which candidate is the right arc"). Log loss is the
    tie-breaker because the HMM uses the probabilities as emissions.

Outputs: results/models/{model_comparison.csv, cv_folds.csv, per_trip_student_test.csv,
         experiment_log.json, *.png}
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from pathsnap import config as C
from pathsnap.experiment import load_partition, subsample_by_obs
from pathsnap.metrics import candidate_metrics, argmax_per_obs, sequence_metrics, candidate_miss_rate

OUT = C.RESULTS_DIR / "models"
OUT.mkdir(parents=True, exist_ok=True)
FEATS = C.FEATURES

# name -> (family, factory, config dict, max training rows (None = all))
MODELS = {
    "LogisticRegression": ("Linear / probabilistic",
        lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        dict(C=1.0, penalty="l2", scaler="standard"), None),
    "GaussianNB": ("Probabilistic (generative)",
        lambda: GaussianNB(var_smoothing=1e-9), dict(var_smoothing=1e-9), None),
    "KNN": ("Instance-based",
        lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=50, weights="distance", n_jobs=-1)),
        dict(n_neighbors=50, weights="distance", scaler="standard", train_rows=80000), 80000),
    "RandomForest": ("Ensemble (bagging)",
        lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt",
                                       max_samples=0.3, n_jobs=-1, random_state=C.SEED),
        dict(n_estimators=200, min_samples_leaf=5, max_features="sqrt", max_samples=0.3), None),
    "LightGBM": ("Boosting",
        lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, random_state=C.SEED, verbose=-1),
        dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8), None),
    "XGBoost": ("Boosting",
        lambda: XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8,
                              colsample_bytree=0.8, tree_method="hist", random_state=C.SEED, n_jobs=-1,
                              eval_metric="logloss"),
        dict(n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8), None),
    "MLP": ("Neural network",
        lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                              batch_size=512, learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                              validation_fraction=0.1, n_iter_no_change=6, random_state=C.SEED)),
        dict(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4, batch_size=512,
             max_iter=60, early_stopping=True, train_rows=400000), 400000),
}


def fit_model(name, df):
    fam, factory, cfg, max_rows = MODELS[name]
    tr = subsample_by_obs(df, max_rows) if max_rows else df
    m = factory()
    t0 = time.time()
    m.fit(tr[FEATS].to_numpy(np.float32), tr.y.to_numpy())
    return m, time.time() - t0


def predict(m, df):
    t0 = time.time()
    p = m.predict_proba(df[FEATS].to_numpy(np.float32))[:, 1]
    return p, time.time() - t0


def evaluate(df, p, routes):
    cm = candidate_metrics(df.y.to_numpy(), p)
    best = argmax_per_obs(df, p)
    seq, per_trip = sequence_metrics(best, routes)
    return {**cm, **seq}, per_trip


def main(only=None):
    train, routes_tr, excluded = load_partition("train")
    test, routes_te, _ = load_partition("student_test", drop_bad_trips=False)
    print(f"train rows={len(train):,} trips={train.trip.nunique()} (excluded {excluded}); "
          f"student_test rows={len(test):,}")
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(train, groups=train.trip))
    cv_rows, summary, per_trip_all, log = [], [], [], {"excluded_train_trips": excluded, "seed": C.SEED,
                                                        "features": FEATS, "models": {}}
    names = only or list(MODELS)
    for name in names:
        fam, _, cfg, _ = MODELS[name]
        print(f"\n=== {name} ({fam})")
        fold_res = []
        for k, (tri, vai) in enumerate(folds):
            tr, va = train.iloc[tri], train.iloc[vai]
            m, ttrain = fit_model(name, tr)
            p, tinf = predict(m, va)
            r, _ = evaluate(va, p, routes_tr)
            r.update(model=name, fold=k, train_time_s=ttrain, infer_time_s=tinf)
            r.pop("confusion")
            fold_res.append(r); cv_rows.append(r)
            print(f"  fold {k}: top1={r['top1']:.4f} f1={r['f1_macro']:.4f} ll={r['log_loss']:.4f} ({ttrain:.0f}s)")
        cv = pd.DataFrame(fold_res)
        m, ttrain = fit_model(name, train)
        p, tinf = predict(m, test)
        rt, per_trip = evaluate(test, p, routes_te)
        per_trip.insert(0, "model", name); per_trip_all.append(per_trip)
        n_obs = test[["trip", "obs"]].drop_duplicates().shape[0]
        row = dict(model=name, family=fam, config=json.dumps(cfg, default=str),
                   cv_top1_mean=cv.top1.mean(), cv_top1_std=cv.top1.std(),
                   cv_f1_macro_mean=cv.f1_macro.mean(), cv_f1_macro_std=cv.f1_macro.std(),
                   cv_log_loss_mean=cv.log_loss.mean(), cv_continuity_mean=cv.continuity.mean(),
                   test_accuracy=rt["accuracy"], test_precision_macro=rt["precision_macro"],
                   test_recall_macro=rt["recall_macro"], test_f1_macro=rt["f1_macro"],
                   test_log_loss=rt["log_loss"], test_top1=rt["top1"], test_continuity=rt["continuity"],
                   test_route_f1=rt["route_f1"], train_time_s=ttrain, infer_time_s=tinf,
                   infer_ms_per_obs=1000 * tinf / n_obs)
        summary.append(row)
        log["models"][name] = dict(family=fam, config=cfg, cv_folds=fold_res, student_test={k: v for k, v in rt.items()})
        print(f"  student_test: top1={rt['top1']:.4f} f1={rt['f1_macro']:.4f} ll={rt['log_loss']:.4f} "
              f"cont={rt['continuity']:.4f}")
        # confusion matrix
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ConfusionMatrixDisplay(np.array(rt["confusion"]), display_labels=["not arc", "true arc"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"{name} - student_test"); plt.tight_layout()
        plt.savefig(OUT / f"confusion_{name}.png", dpi=120); plt.close()
        if name == "MLP":  # only model with an epoch-wise history
            mlp = m[-1]
            fig, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(mlp.loss_curve_, label="training loss (log loss)")
            ax1.set_xlabel("epoch"); ax1.set_ylabel("training loss")
            ax2 = ax1.twinx(); ax2.plot(mlp.validation_scores_, color="tab:orange", label="validation accuracy")
            ax2.set_ylabel("validation accuracy (10% hold-out)")
            fig.legend(loc="center right"); ax1.set_title("MLP training history"); plt.tight_layout()
            plt.savefig(OUT / "mlp_training_history.png", dpi=120); plt.close()

    cvdf = pd.DataFrame(cv_rows)
    summ = pd.DataFrame(summary)
    tag = "" if only is None else "_partial"
    cvdf.to_csv(OUT / f"cv_folds{tag}.csv", index=False)
    summ.to_csv(OUT / f"model_comparison{tag}.csv", index=False)
    pd.concat(per_trip_all).to_csv(OUT / f"per_trip_student_test{tag}.csv", index=False)
    with open(OUT / f"experiment_log{tag}.json", "w") as f:
        json.dump(log, f, indent=1, default=str)

    # selection rule
    s = summ.sort_values(["cv_top1_mean", "cv_log_loss_mean"], ascending=[False, True])
    best = s.iloc[0].model
    with open(OUT / f"selection{tag}.json", "w") as f:
        json.dump({"selected_model": best, "rule": "max CV top-1 accuracy, tie-break min CV log loss",
                   "ranking": s.model.tolist()}, f, indent=1)
    print("\nRanking by CV top-1:", s.model.tolist(), "-> selected:", best)

    # comparison plots
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    order = s.model.tolist()
    ax[0].boxplot([cvdf[cvdf.model == m].f1_macro for m in order], tick_labels=order)
    ax[0].set_title("CV Macro F1 (5 GroupKFold folds)"); ax[0].tick_params(axis="x", rotation=30)
    x = np.arange(len(order)); w = 0.38
    st_ = summ.set_index("model").loc[order]
    ax[1].bar(x - w / 2, st_.test_top1, w, label="Top-1 accuracy")
    ax[1].bar(x + w / 2, st_.test_f1_macro, w, label="Macro F1")
    ax[1].set_xticks(x, order, rotation=30); ax[1].set_ylim(min(0.5, st_.test_f1_macro.min() - 0.05), 1)
    ax[1].legend(); ax[1].set_title("Student-test performance")
    ax[2].bar(order, st_.test_log_loss, color="#8e24aa"); ax[2].tick_params(axis="x", rotation=30)
    ax[2].set_title("Student-test log loss (lower is better)")
    plt.tight_layout(); plt.savefig(OUT / f"model_comparison{tag}.png", dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(st_.infer_time_s, st_.test_top1)
    for m_, r in st_.iterrows():
        ax.annotate(m_, (r.infer_time_s, r.test_top1))
    ax.set(xscale="log", xlabel="inference time on student_test (s, log)", ylabel="student-test Top-1",
           title="Accuracy vs inference cost")
    plt.tight_layout(); plt.savefig(OUT / f"accuracy_vs_time{tag}.png", dpi=130); plt.close()
    print(summ.drop(columns="config").round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or None)

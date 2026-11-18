"""Bayesian hyper-parameter optimisation (Optuna TPE) of the selected model.

  * Objective: mean observation-level Top-1 accuracy over a 3-fold GroupKFold
    (grouped by trip) on the TRAIN partition only.  3 folds instead of 5 keeps
    the cost of each trial manageable; folds never split a trip.
  * Search space: see SPACES (justification in README / report).
  * The tuned model is refit on all training trips, compared with the untuned
    configuration on the student_test trips, and exported to models/final_model.joblib.
  * Out-of-fold (OOF) probabilities on the training trips are saved so that the HMM
    decoder parameters can be tuned on predictions the model has NOT been trained on.
"""
import sys, json, time, importlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import joblib
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from pathsnap import config as C
from pathsnap.experiment import load_partition
from pathsnap.metrics import argmax_per_obs, top1_accuracy

T = importlib.import_module("03_train_models")
OUT = C.RESULTS_DIR / "bayes_opt"; OUT.mkdir(parents=True, exist_ok=True)
FEATS = C.FEATURES
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def space_lgbm(tr):
    return LGBMClassifier(
        n_estimators=tr.suggest_int("n_estimators", 100, 800, step=50),
        learning_rate=tr.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=tr.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=tr.suggest_int("min_child_samples", 10, 300, log=True),
        subsample=tr.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
        colsample_bytree=tr.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=tr.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        random_state=C.SEED, verbose=-1)


def space_xgb(tr):
    return XGBClassifier(
        n_estimators=tr.suggest_int("n_estimators", 100, 800, step=50),
        learning_rate=tr.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=tr.suggest_int("max_depth", 3, 12),
        min_child_weight=tr.suggest_float("min_child_weight", 0.5, 20, log=True),
        subsample=tr.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=tr.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=tr.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        tree_method="hist", random_state=C.SEED, n_jobs=-1, eval_metric="logloss")


def space_rf(tr):
    return RandomForestClassifier(
        n_estimators=tr.suggest_int("n_estimators", 100, 400, step=50),
        max_depth=tr.suggest_int("max_depth", 8, 40),
        min_samples_leaf=tr.suggest_int("min_samples_leaf", 1, 30, log=True),
        max_features=tr.suggest_float("max_features", 0.2, 1.0),
        max_samples=tr.suggest_float("max_samples", 0.1, 0.5),
        n_jobs=-1, random_state=C.SEED)


SPACES = {"LightGBM": space_lgbm, "XGBoost": space_xgb, "RandomForest": space_rf}


def cv_top1(make_model, df, folds):
    scores = []
    for tri, vai in folds:
        tr, va = df.iloc[tri], df.iloc[vai]
        m = make_model()
        m.fit(tr[FEATS].to_numpy(np.float32), tr.y.to_numpy())
        p = m.predict_proba(va[FEATS].to_numpy(np.float32))[:, 1]
        b = argmax_per_obs(va, p)
        scores.append(top1_accuracy(b.pred_arc.to_numpy(), b.label_arc.to_numpy()))
    return float(np.mean(scores)), scores


def main():
    sel = json.loads((C.RESULTS_DIR / "models" / "selection.json").read_text())
    name = sel["selected_model"]
    if name not in SPACES:
        # fall back to the best model that has a search space defined
        name = next(m for m in sel["ranking"] if m in SPACES)
    print("Optimising", name)
    train, routes_tr, _ = load_partition("train")
    test, routes_te, _ = load_partition("student_test", drop_bad_trips=False)
    folds3 = list(GroupKFold(n_splits=3).split(train, groups=train.trip))

    t_start = time.time()

    def objective(trial):
        t0 = time.time()
        mean, scores = cv_top1(lambda: SPACES[name](trial), train, folds3)
        trial.set_user_attr("fold_scores", scores)
        trial.set_user_attr("seconds", time.time() - t0)
        print(f"trial {trial.number}: {mean:.4f} {trial.params}", flush=True)
        return mean

    # baseline (untuned) under the same 3-fold protocol, enqueued as context
    base_cv, base_scores = cv_top1(T.MODELS[name][1], train, folds3)
    print(f"untuned 3-fold CV top1 = {base_cv:.4f}")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.SEED),
                                study_name=f"pathsnap_{name}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_TRIALS)
    opt_seconds = time.time() - t_start

    trials = study.trials_dataframe()
    trials.to_csv(OUT / "optuna_trials.csv", index=False)
    best_params = study.best_params
    print("best", study.best_value, best_params)

    # plots
    fig, ax = plt.subplots(figsize=(7, 4))
    v = trials.value.to_numpy()
    ax.plot(v, "o", alpha=0.6, label="trial")
    ax.plot(np.maximum.accumulate(v), "r-", label="best so far")
    ax.axhline(base_cv, color="gray", ls="--", label="untuned")
    ax.set(xlabel="trial", ylabel="3-fold CV Top-1", title=f"Bayesian optimisation history ({name})")
    ax.legend(); plt.tight_layout(); plt.savefig(OUT / "optimisation_history.png", dpi=130); plt.close()
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(list(imp)[::-1], list(imp.values())[::-1]); ax.set_title("Hyper-parameter importance (fANOVA)")
        plt.tight_layout(); plt.savefig(OUT / "param_importance.png", dpi=130); plt.close()
    except Exception as e:  # pragma: no cover
        print("importance failed:", e)

    # ---- compare untuned vs tuned on full train -> student_test ----
    from pathsnap.metrics import candidate_metrics, sequence_metrics
    tuned_factory = lambda: SPACES[name](optuna.trial.FixedTrial(best_params))
    rows, models = [], {}
    folds5 = list(GroupKFold(n_splits=5).split(train, groups=train.trip))
    for label, factory in [("untuned", T.MODELS[name][1]), ("tuned", tuned_factory)]:
        cv5 = []
        from sklearn.metrics import f1_score
        for tri, vai in folds5:
            m = factory(); m.fit(train.iloc[tri][FEATS].to_numpy(np.float32), train.iloc[tri].y.to_numpy())
            p = m.predict_proba(train.iloc[vai][FEATS].to_numpy(np.float32))[:, 1]
            cv5.append(f1_score(train.iloc[vai].y, p >= 0.5, average="macro"))
        m = factory(); t0 = time.time()
        m.fit(train[FEATS].to_numpy(np.float32), train.y.to_numpy()); ttrain = time.time() - t0
        t0 = time.time(); p = m.predict_proba(test[FEATS].to_numpy(np.float32))[:, 1]; tinf = time.time() - t0
        cm = candidate_metrics(test.y.to_numpy(), p)
        seq, _ = sequence_metrics(argmax_per_obs(test, p), routes_te)
        rows.append(dict(version=label, cv5_f1_macro=np.mean(cv5), cv3_top1=base_cv if label == "untuned" else study.best_value,
                         test_top1=seq["top1"], test_f1_macro=cm["f1_macro"], test_log_loss=cm["log_loss"],
                         test_continuity=seq["continuity"], train_time_s=ttrain, infer_time_s=tinf))
        models[label] = m
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "tuned_vs_untuned.csv", index=False)
    print(comp.round(4).to_string())

    # choose the version to deploy using CV evidence only (not the student-test numbers)
    deploy = "tuned" if study.best_value >= base_cv else "untuned"
    final = models[deploy]
    C.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": FEATS, "name": f"{name} ({deploy})",
                 "params": best_params if deploy == "tuned" else T.MODELS[name][2]},
                C.MODELS_DIR / "final_model.joblib", compress=3)
    json.dump({"model": name, "deployed_version": deploy, "best_params": best_params,
               "best_cv3_top1": study.best_value, "untuned_cv3_top1": base_cv, "n_trials": N_TRIALS,
               "optimisation_seconds": opt_seconds, "objective": "mean 3-fold GroupKFold top-1 accuracy (train only)"},
              open(OUT / "best_params.json", "w"), indent=1)

    # ---- OOF probabilities on train for HMM tuning ----
    # predict on ALL candidate rows of each held-out trip (incl. unlabelled observations),
    # because the HMM needs a probability for every state in the chain.
    factory = tuned_factory if deploy == "tuned" else T.MODELS[name][1]
    full = pd.read_parquet(C.CACHE_DIR / "train.parquet")
    full = full[full.trip.isin(train.trip.unique())].reset_index(drop=True)
    oof = np.zeros(len(full))
    trips = np.array(sorted(train.trip.unique()))
    for k, (_, va_t) in enumerate(GroupKFold(n_splits=5).split(trips, groups=trips)):
        val_trips = set(trips[va_t])
        m = factory()
        tr = train[~train.trip.isin(val_trips)]
        m.fit(tr[FEATS].to_numpy(np.float32), tr.y.to_numpy())
        mask = full.trip.isin(val_trips).to_numpy()
        oof[mask] = m.predict_proba(full.loc[mask, FEATS].to_numpy(np.float32))[:, 1]
    pd.DataFrame({"trip": full.trip, "obs": full.obs, "arc": full.arc, "p": oof}).to_parquet(
        C.CACHE_DIR / "train_oof.parquet", index=False)
    print("saved final model + OOF predictions")


if __name__ == "__main__":
    main()

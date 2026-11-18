"""Export every compared model so the API can serve any of them.

Each model is refit on ALL usable training trips with exactly the configuration
used in scripts/03_train_models.py (same subsampling for KNN / MLP, seed 42),
then saved to models/zoo/<id>.joblib together with its offline evaluation
numbers from results/models/model_comparison.csv.

The deployed tuned XGBoost (models/final_model.joblib) is copied in as
"xgboost_tuned" - it is NOT retrained here.

Large files (> 95 MB, e.g. Random Forest) are written but listed in
models/zoo/.gitignore because GitHub rejects files over 100 MB; re-create them
on Laptop 1 by running this script.
"""
import sys, json, time, importlib, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

import joblib
import pandas as pd

from pathsnap import config as C
from pathsnap.experiment import load_partition

T = importlib.import_module("03_train_models")
ZOO = C.MODELS_DIR / "zoo"
ZOO.mkdir(parents=True, exist_ok=True)

IDS = {"LogisticRegression": "logistic_regression", "GaussianNB": "gaussian_nb", "KNN": "knn",
       "RandomForest": "random_forest", "LightGBM": "lightgbm", "XGBoost": "xgboost", "MLP": "mlp"}
LIMIT_MB = 95


def main(only=None):
    comp = pd.read_csv(C.RESULTS_DIR / "models" / "model_comparison.csv").set_index("model")
    tuned = pd.read_csv(C.RESULTS_DIR / "bayes_opt" / "tuned_vs_untuned.csv").set_index("version")
    index, big = [], []

    # deployed model, copied as-is
    fb = joblib.load(C.MODELS_DIR / "final_model.joblib")
    shutil.copy(C.MODELS_DIR / "final_model.joblib", ZOO / "xgboost_tuned.joblib")
    index.append(dict(id="xgboost_tuned", name="XGBoost (Bayesian-tuned) - deployed", family="Boosting",
                      default=True, config=fb.get("params", {}),
                      offline=dict(cv_top1=None, test_top1=float(tuned.loc["tuned", "test_top1"]),
                                   test_f1_macro=float(tuned.loc["tuned", "test_f1_macro"]),
                                   test_log_loss=float(tuned.loc["tuned", "test_log_loss"]))))

    train, _, _ = load_partition("train")
    for name, mid in IDS.items():
        if only and mid not in only:
            continue
        fam, _, cfg, _ = T.MODELS[name]
        t0 = time.time()
        m, _ = T.fit_model(name, train)
        path = ZOO / f"{mid}.joblib"
        joblib.dump({"model": m, "features": C.FEATURES, "name": name, "params": cfg}, path, compress=3)
        mb = path.stat().st_size / 1e6
        if mb > LIMIT_MB:
            big.append(path.name)
        r = comp.loc[name]
        index.append(dict(id=mid, name=name, family=fam, default=False, config=cfg, size_mb=round(mb, 1),
                          offline=dict(cv_top1=float(r.cv_top1_mean), test_top1=float(r.test_top1),
                                       test_f1_macro=float(r.test_f1_macro), test_log_loss=float(r.test_log_loss))))
        print(f"{mid:22s} {mb:7.1f} MB  ({time.time() - t0:.0f}s)", flush=True)

    (ZOO / "index.json").write_text(json.dumps(index, indent=1))
    (ZOO / ".gitignore").write_text("".join(f"{b}\n" for b in big))
    print("large files kept local only:", big or "none")


if __name__ == "__main__":
    main(sys.argv[1:] or None)

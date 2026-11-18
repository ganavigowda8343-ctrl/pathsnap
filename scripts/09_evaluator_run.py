"""Evaluation-day run: the DEPLOYED pipeline (tuned XGBoost + HMM + gap filling) on the
evaluator trips, exactly as the API does it.

This script only runs inference and scores the result; nothing is trained, tuned or
selected from these numbers. It exists so that the evaluator results quoted in the
report (Section 7.1) are reproducible from the repository.

Usage: python scripts/09_evaluator_run.py          (uses PATHSNAP_EVAL_DIR or data/evaluator_test)
Output: results/evaluator/evaluator_results.csv, results/evaluator/summary.json,
        results/maps/evaluator_<trip>.html
"""
import sys, json, importlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from pathsnap import config as C
from pathsnap.data_io import list_trips, load_route
from pathsnap.pipeline import PathSnapModel, evaluate_prediction

H = importlib.import_module("05_hmm")  # reuse the Folium trip-map code
OUT = C.RESULTS_DIR / "evaluator"; OUT.mkdir(parents=True, exist_ok=True)


def main():
    trips = list_trips(C.EVAL_DIR)
    if not trips:
        sys.exit(f"no evaluator trips found in {C.EVAL_DIR}")
    model = PathSnapModel()
    rows = []
    for t in trips:
        d = C.EVAL_DIR / t
        res = model.predict_trip(d)
        ev = evaluate_prediction(d, res)
        rows.append(dict(trip=t, n_points=ev["n_points"], n_labelled=ev["n_labelled"],
                         top1_classifier_only=ev["clf"]["top1_accuracy"], top1_hmm=ev["hmm"]["top1_accuracy"],
                         route_precision=ev["hmm_route"]["route_precision"], route_recall=ev["hmm_route"]["route_recall"],
                         route_f1=ev["hmm_route"]["route_f1"], route_continuity=ev["hmm_route"]["route_continuity"],
                         hmm_chain_breaks=res["hmm_breaks"], unmatched_points=ev["n_unmatched_points"],
                         seconds=res["timings"]["total_s"]))
        H.trip_map(res, load_route(d), C.RESULTS_DIR / "maps" / f"evaluator_{t}.html", ev)
        print(t, f"clf={rows[-1]['top1_classifier_only']:.4f} hmm={rows[-1]['top1_hmm']:.4f} "
                 f"routeF1={rows[-1]['route_f1']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "evaluator_results.csv", index=False)
    lab = df.n_labelled.sum()
    summ = dict(model=model.model_name, n_trips=len(df), n_points=int(df.n_points.sum()), n_labelled=int(lab),
                mean_top1_classifier_only=df.top1_classifier_only.mean(), mean_top1_hmm=df.top1_hmm.mean(),
                pooled_top1_classifier_only=float((df.top1_classifier_only * df.n_labelled).sum() / lab),
                pooled_top1_hmm=float((df.top1_hmm * df.n_labelled).sum() / lab),
                mean_route_f1=df.route_f1.mean(), mean_route_precision=df.route_precision.mean(),
                mean_route_recall=df.route_recall.mean(), mean_route_continuity=df.route_continuity.mean(),
                seconds_min=df.seconds.min(), seconds_max=df.seconds.max())
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

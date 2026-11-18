"""Build candidate/feature tables for the train and student_test partitions.

Output: data_cache/<partition>.parquet (one row per observation x candidate)
        results/features/<partition>_trip_summary.csv
Labels come from the reference route via monotone alignment (labels.py).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap import config as C
from pathsnap.data_io import list_trips, load_trip, load_route
from pathsnap.features import prepare_trip, build_candidate_table
from pathsnap.labels import derive_labels


def process(part_dir, trip_id):
    t0 = time.time()
    d = Path(part_dir) / trip_id
    prep = prepare_trip(load_trip(d))
    df = build_candidate_table(prep)
    route = load_route(d)
    label, ldist = derive_labels(prep["P"], route, prep["net"])
    df["label_arc"] = label[df["obs"].to_numpy()]
    df["y"] = (df["arc"].to_numpy() == df["label_arc"].to_numpy()).astype(np.int8)
    df.insert(0, "trip", trip_id)
    n = len(prep["P"])
    labelled = label >= 0
    covered = np.zeros(n, bool)
    covered[df.loc[df.y == 1, "obs"].to_numpy()] = True
    summ = dict(trip=trip_id, n_points=n, n_labelled=int(labelled.sum()),
                frac_labelled=float(labelled.mean()),
                label_dist_median=float(np.median(ldist)),
                cand_rows=len(df), mean_cands=len(df) / n,
                candidate_miss_rate=float(1 - covered[labelled].mean()) if labelled.any() else np.nan,
                seconds=time.time() - t0)
    return df, summ


def main():
    C.CACHE_DIR.mkdir(exist_ok=True)
    out = C.RESULTS_DIR / "features"; out.mkdir(parents=True, exist_ok=True)
    for part, pdir in [("train", C.TRAIN_DIR), ("student_test", C.STUDENT_TEST_DIR)]:
        trips = list_trips(pdir)
        print(f"[{part}] {len(trips)} trips")
        res = Parallel(n_jobs=-1, verbose=5)(delayed(process)(pdir, t) for t in trips)
        df = pd.concat([r[0] for r in res], ignore_index=True)
        summ = pd.DataFrame([r[1] for r in res])
        df.to_parquet(C.CACHE_DIR / f"{part}.parquet", index=False)
        summ.to_csv(out / f"{part}_trip_summary.csv", index=False)
        lab = df[df.label_arc >= 0]
        print(f"[{part}] rows={len(df):,} obs={summ.n_points.sum():,} labelled={summ.n_labelled.sum():,} "
              f"pos_rate={lab.y.mean():.3f} cand_miss={summ.candidate_miss_rate.mean():.4f}")


if __name__ == "__main__":
    main()

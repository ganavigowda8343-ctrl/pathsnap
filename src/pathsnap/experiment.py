"""Shared helpers for the training / tuning / HMM scripts."""
import numpy as np
import pandas as pd

from . import config as C
from .data_io import list_trips, load_route


def load_partition(part, drop_bad_trips=True):
    """Load cached candidate table for a partition.

    Returns (df_labelled, routes). Only observations with a reliable label are kept,
    and (for training) trips with too few reliable labels are excluded.
    """
    df = pd.read_parquet(C.CACHE_DIR / f"{part}.parquet")
    summ = pd.read_csv(C.RESULTS_DIR / "features" / f"{part}_trip_summary.csv", dtype={"trip": str})
    excluded = []
    if drop_bad_trips:
        excluded = summ.loc[summ.frac_labelled < C.MIN_TRIP_LABEL_FRAC, "trip"].tolist()
        df = df[~df.trip.isin(excluded)]
    df = df[df.label_arc >= 0].reset_index(drop=True)
    pdir = C.TRAIN_DIR if part == "train" else C.STUDENT_TEST_DIR
    routes = {t: load_route(pdir / t) for t in df.trip.unique()}
    return df, routes, excluded


def subsample_by_obs(df, n_rows, seed=C.SEED):
    """Subsample whole observations (keeps candidate groups intact)."""
    if len(df) <= n_rows:
        return df
    keys = df[["trip", "obs"]].drop_duplicates()
    frac = n_rows / len(df)
    keep = keys.sample(frac=frac, random_state=seed)
    return df.merge(keep, on=["trip", "obs"])

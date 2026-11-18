"""Candidate generation and per-candidate feature engineering.

Problem framing: road-network arc ids are local to each trip (every trip has its
own network), so we cannot classify "which arc id".  Instead, for every GPS
observation we generate up to K nearby candidate arcs and solve a BINARY task:
"is this candidate the true arc for this observation?"  The classifier score is
later used (a) directly via argmax per observation, and (b) as the HMM emission.

Feature groups
  geometry   : perpendicular distance, rank and relative distance among the
               candidates of the same observation, position along the arc.
  direction  : agreement between GPS heading and the DIRECTED arc bearing
               (distinguishes the two directions of a two-way street).
  kinematics : speed and sampling gaps.
  context    : how close the neighbouring GPS points (i-3..i+3) are to the same
               arc -> a vehicle stays on a road for several seconds, so noise on a
               single point is smoothed out.
  topology   : node degrees and whether a reverse arc exists.
"""
import numpy as np
import pandas as pd

from . import config as C
from .network import LocalProjection, RoadNetwork, point_segment


def prepare_trip(trip):
    """Project a raw trip and build its network + spatial index."""
    track = trip["track"]
    lon0, lat0 = np.median(track[:, 0]), np.median(track[:, 1])
    proj = LocalProjection(lon0, lat0)
    P = proj.to_xy(track[:, 0], track[:, 1])
    net = RoadNetwork(trip["nodes"], trip["arcs"], proj, focus_xy=P)
    return {"trip_id": trip["trip_id"], "P": P, "t": track[:, 2], "lonlat": track[:, :2],
            "net": net, "proj": proj}


def _heading(P, h=C.HEADING_HALF_WINDOW):
    n = len(P)
    i0 = np.clip(np.arange(n) - h, 0, n - 1)
    i1 = np.clip(np.arange(n) + h, 0, n - 1)
    d = P[i1] - P[i0]
    disp = np.linalg.norm(d, axis=1)
    return np.arctan2(d[:, 1], d[:, 0]), disp >= C.MIN_HEADING_DISPLACEMENT_M


def build_candidate_table(prep):
    """Return a long DataFrame: one row per (observation, candidate arc)."""
    P, t, net = prep["P"], prep["t"], prep["net"]
    n = len(P)
    arc_k, dist_k = net.candidates(P)
    K = arc_k.shape[1]
    obs = np.repeat(np.arange(n), K)
    arc = arc_k.ravel()
    dist = dist_k.ravel()
    keep = arc >= 0
    obs, arc, dist = obs[keep], arc[keep], dist[keep]

    n_cand = np.bincount(obs, minlength=n)
    dmin = np.full(n, np.inf)
    np.minimum.at(dmin, obs, dist)
    # rank within observation (rows are already sorted by distance per obs)
    first = np.r_[0, np.cumsum(n_cand)[:-1]]
    rank = np.arange(len(obs)) - first[obs]

    Pi = P[obs]
    _, tt, t_raw = point_segment(Pi, net.A[arc], net.B[arc])
    L = net.length[arc]
    beyond = np.maximum(np.maximum(-t_raw, t_raw - 1.0), 0.0) * L

    head, hvalid = _heading(P)
    dtheta = np.angle(np.exp(1j * (head[obs] - net.bearing[arc])))
    cos_h = np.where(hvalid[obs], np.cos(dtheta), 0.0)
    abs_h = np.where(hvalid[obs], np.degrees(np.abs(dtheta)), 90.0)

    dt = np.diff(t)
    dt_prev = np.r_[1.0, dt]
    dt_next = np.r_[dt, 1.0]
    step = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1)]
    speed = step / np.maximum(dt_prev, 1e-6)
    speed = np.minimum(speed, 80.0)

    # context: distance of neighbouring points to the same candidate arc
    ctx = []
    for o in C.CONTEXT_OFFSETS:
        j = np.clip(obs + o, 0, n - 1)
        d_o, _, _ = point_segment(P[j], net.A[arc], net.B[arc])
        ctx.append(d_o)
    ctx = np.stack(ctx, 1)
    ctx_mean = ctx.mean(1)
    ctx_max = ctx.max(1)
    cmin = np.full(n, np.inf)
    np.minimum.at(cmin, obs, ctx_mean)
    # rank of context score within observation
    df_tmp = pd.DataFrame({"obs": obs, "c": ctx_mean})
    ctx_rank = df_tmp.groupby("obs")["c"].rank(method="first").to_numpy() - 1

    df = pd.DataFrame({
        "obs": obs.astype(np.int32),
        "arc": arc.astype(np.int64),
        "dist": dist,
        "dist_rank": rank,
        "dist_minus_min": dist - dmin[obs],
        "dist_ratio": dist / (dmin[obs] + 1.0),
        "n_cand": n_cand[obs],
        "t_frac": tt,
        "beyond_m": beyond,
        "log_arc_len": np.log1p(L),
        "heading_valid": hvalid[obs].astype(np.int8),
        "cos_heading": cos_h,
        "abs_heading_deg": abs_h,
        "speed_mps": speed[obs],
        "dt_prev": dt_prev[obs],
        "dt_next": dt_next[obs],
        "ctx_mean_dist": ctx_mean,
        "ctx_max_dist": ctx_max,
        "ctx_mean_minus_min": ctx_mean - cmin[obs],
        "ctx_rank": ctx_rank,
        "has_reverse": net.has_reverse[arc].astype(np.int8),
        "out_deg": net.out_deg[net.arcs[arc, 1]],
        "in_deg": net.in_deg[net.arcs[arc, 0]],
        # --- bookkeeping columns (NOT model features) ---
        "along_m": tt * L,      # used by the HMM
        "u": net.arcs[arc, 0],  # arc start node (route continuity / HMM)
        "v": net.arcs[arc, 1],  # arc end node
    })
    return df

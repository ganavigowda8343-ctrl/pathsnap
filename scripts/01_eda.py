"""Exploratory data analysis (train + student_test partitions only).

Outputs in results/eda/:
  inventory.csv            per-trip file sizes, row counts, time span, extent, network stats
  partition_summary.csv    aggregated statistics per partition
  data_quality_notes.md    anomalies found + decisions
  *.png                    plots
  map_trip_<id>.html       Folium maps (GPS + network + reference route)
  map_world.html           Folium overview of where trips are located
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap import config as C
from pathsnap.data_io import list_trips, load_trip, load_route
from pathsnap.features import prepare_trip
from pathsnap.network import point_segment

OUT = C.RESULTS_DIR / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def trip_stats(part, pdir, tid):
    d = Path(pdir) / tid
    sizes = {ext: (d / f"{tid}.{ext}").stat().st_size for ext in ["track", "nodes", "arcs", "route"]}
    trip = load_trip(d)
    route = load_route(d)
    prep = prepare_trip(trip)
    net, P, t = prep["net"], prep["P"], prep["t"]
    tr = trip["track"]
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    spd = step / np.maximum(dt, 1e-6)
    n_nodes, arcs = net.n_nodes, trip["arcs"]
    g = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n_nodes, n_nodes))
    n_weak, lab_w = connected_components(g, directed=True, connection="weak")
    n_strong, lab_s = connected_components(g, directed=True, connection="strong")
    deg = net.out_deg + net.in_deg
    route_cont = np.mean(arcs[route[:-1], 1] == arcs[route[1:], 0]) if len(route) > 1 else np.nan
    # GPS -> nearest reference-route arc distance (every 5th point to save time)
    Ps = P[::5]
    dmin = np.full(len(Ps), np.inf)
    for s in range(0, len(route), 500):
        r = route[s:s + 500]
        dd, _, _ = point_segment(Ps[:, None, :], net.A[r][None], net.B[r][None])
        dmin = np.minimum(dmin, dd.min(1))
    return dict(
        partition=part, trip=tid, **{f"size_{k}_kb": v / 1024 for k, v in sizes.items()},
        n_points=len(tr), n_nodes=n_nodes, n_arcs=len(arcs), n_route_arcs=len(route),
        missing_values=int((~np.isfinite(tr)).sum()),
        duration_s=float(t[-1] - t[0]), dt_median=float(np.median(dt)), dt_max=float(dt.max()),
        frac_dt_1s=float(np.mean(np.isclose(dt, 1.0))), n_gaps_gt_10s=int((dt > 10).sum()),
        non_increasing_ts=int((dt <= 0).sum()),
        path_length_km=float(step.sum() / 1000), speed_p95_mps=float(np.percentile(spd, 95)),
        n_speed_gt_60mps=int((spd > 60).sum()),
        lon_min=tr[:, 0].min(), lon_max=tr[:, 0].max(), lat_min=tr[:, 1].min(), lat_max=tr[:, 1].max(),
        extent_x_km=float(np.ptp(P[:, 0]) / 1000), extent_y_km=float(np.ptp(P[:, 1]) / 1000),
        weak_components=n_weak, strong_components=n_strong,
        largest_strong_frac=float(np.bincount(lab_s).max() / n_nodes),
        isolated_nodes=int((deg == 0).sum()), zero_length_arcs=int((net.length < 0.01).sum()),
        arc_len_median_m=float(np.median(net.length)),
        bidirectional_frac=float(net.has_reverse.mean()),
        route_continuity=float(route_cont),
        route_arcs_unique_frac=float(len(np.unique(route)) / len(route)),
        gps_route_dist_median=float(np.median(dmin)), gps_route_dist_p95=float(np.percentile(dmin, 95)),
        frac_gps_gt_50m_from_route=float(np.mean(dmin > 50)),
    ), dt, dmin


def trip_map(pdir, tid, path):
    d = Path(pdir) / tid
    trip = load_trip(d); route = load_route(d); prep = prepare_trip(trip)
    net = prep["net"]; nodes = trip["nodes"]; arcs = trip["arcs"]; tr = trip["track"]
    c = [float(np.median(tr[:, 1])), float(np.median(tr[:, 0]))]
    m = folium.Map(location=c, zoom_start=14, tiles="OpenStreetMap")
    fg_net = folium.FeatureGroup(name="Road network (arcs near track)")
    near = np.unique(net.sample_arc)
    for a in near[:6000]:
        u, v = arcs[a]
        folium.PolyLine([[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]],
                        color="#9e9e9e", weight=1, opacity=0.6).add_to(fg_net)
    fg_ref = folium.FeatureGroup(name="Reference route (.route)")
    for a in np.unique(route):
        u, v = arcs[a]
        folium.PolyLine([[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]],
                        color="#2e7d32", weight=5, opacity=0.8).add_to(fg_ref)
    fg_gps = folium.FeatureGroup(name="Raw GPS observations")
    step = max(1, len(tr) // 1500)
    for lon, lat, t in tr[::step]:
        folium.CircleMarker([lat, lon], radius=2, color="#1565c0", fill=True, fill_opacity=0.8,
                            tooltip=f"t={t:.0f}s").add_to(fg_gps)
    folium.Marker([tr[0, 1], tr[0, 0]], tooltip="Start", icon=folium.Icon(color="green")).add_to(fg_gps)
    folium.Marker([tr[-1, 1], tr[-1, 0]], tooltip="End", icon=folium.Icon(color="red")).add_to(fg_gps)
    for fg in (fg_net, fg_ref, fg_gps):
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    m.save(str(path))


def main():
    rows, all_dt, all_d = [], {"train": [], "student_test": []}, {"train": [], "student_test": []}
    for part, pdir in [("train", C.TRAIN_DIR), ("student_test", C.STUDENT_TEST_DIR)]:
        for tid in list_trips(pdir):
            r, dt, dmin = trip_stats(part, pdir, tid)
            rows.append(r); all_dt[part].append(dt); all_d[part].append(dmin)
            print(part, tid, r["n_points"], f"{r['gps_route_dist_median']:.1f}m")
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "inventory.csv", index=False)
    summ = inv.groupby("partition").agg(
        trips=("trip", "count"), points_total=("n_points", "sum"), points_median=("n_points", "median"),
        points_min=("n_points", "min"), points_max=("n_points", "max"),
        nodes_median=("n_nodes", "median"), arcs_median=("n_arcs", "median"),
        route_arcs_median=("n_route_arcs", "median"), missing_values=("missing_values", "sum"),
        duration_median_s=("duration_s", "median"), frac_dt_1s_mean=("frac_dt_1s", "mean"),
        gaps_gt_10s=("n_gaps_gt_10s", "sum"), isolated_nodes=("isolated_nodes", "sum"),
        weak_components_median=("weak_components", "median"),
        route_continuity_mean=("route_continuity", "mean"),
        gps_route_dist_median=("gps_route_dist_median", "median"))
    summ.T.to_csv(OUT / "partition_summary.csv")
    print(summ.T)

    # ---------------- plots ----------------
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for part, col in [("train", "#1f77b4"), ("student_test", "#ff7f0e")]:
        s = inv[inv.partition == part]
        ax[0, 0].hist(s.n_points, bins=30, alpha=0.6, label=part, color=col)
        ax[0, 1].hist(np.log10(s.n_arcs), bins=30, alpha=0.6, label=part, color=col)
        dt = np.concatenate(all_dt[part])
        ax[0, 2].hist(np.clip(dt, 0, 15), bins=np.arange(0, 16, 0.5), alpha=0.6, label=part, color=col, log=True)
        dd = np.concatenate(all_d[part])
        ax[1, 0].hist(np.clip(dd, 0, 200), bins=80, alpha=0.6, label=part, color=col, log=True, density=True)
        ax[1, 1].scatter(s.path_length_km, s.duration_s / 60, alpha=0.7, label=part, color=col)
    ax[0, 0].set(title="GPS observations per trip", xlabel="points")
    ax[0, 1].set(title="Road-network size (log10 arcs)", xlabel="log10(#arcs)")
    ax[0, 2].set(title="Sampling interval (clipped at 15 s)", xlabel="seconds")
    ax[1, 0].set(title="GPS distance to nearest reference arc (clip 200 m)", xlabel="metres")
    ax[1, 1].set(title="Trip length vs duration", xlabel="km", ylabel="minutes")
    ax[1, 2].scatter(inv.lon_min, inv.lat_min, c=(inv.partition == "train"), cmap="coolwarm", s=25)
    ax[1, 2].set(title="Trip locations (lon/lat)", xlabel="lon", ylabel="lat")
    for a in ax.ravel()[:5]:
        a.legend()
    plt.tight_layout(); plt.savefig(OUT / "eda_overview.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    s = inv.sort_values("frac_gps_gt_50m_from_route", ascending=False).head(15)
    ax[0].barh(s.partition.str[0] + "-" + s.trip, s.frac_gps_gt_50m_from_route, color="#c62828")
    ax[0].set(title="Trips with most GPS points >50 m from reference route", xlabel="fraction")
    ax[0].invert_yaxis()
    ax[1].scatter(inv.n_arcs, inv.weak_components, c=(inv.partition == "train"), cmap="coolwarm")
    ax[1].set(xscale="log", title="Network connectivity", xlabel="#arcs", ylabel="#weakly connected components")
    plt.tight_layout(); plt.savefig(OUT / "eda_quality.png", dpi=130); plt.close()

    # ---------------- maps ----------------
    wm = folium.Map(location=[30, 10], zoom_start=2, tiles="OpenStreetMap")
    for _, r in inv.iterrows():
        folium.CircleMarker([(r.lat_min + r.lat_max) / 2, (r.lon_min + r.lon_max) / 2], radius=5,
                            color="#1565c0" if r.partition == "train" else "#ef6c00", fill=True,
                            tooltip=f"{r.partition} {r.trip}: {r.n_points} pts").add_to(wm)
    wm.save(str(OUT / "map_world.html"))
    for part, pdir, tid in [("train", C.TRAIN_DIR, "00000001"), ("train", C.TRAIN_DIR, "00000054"),
                            ("student_test", C.STUDENT_TEST_DIR, list_trips(C.STUDENT_TEST_DIR)[0])]:
        trip_map(pdir, tid, OUT / f"map_{part}_{tid}.html")

    # ---------------- data-quality notes ----------------
    bad = inv[inv.frac_gps_gt_50m_from_route > 0.3]
    gaps = inv.sort_values("dt_max", ascending=False).head(5)
    with open(OUT / "data_quality_notes.md", "w") as f:
        f.write("# Data-quality notes (generated by scripts/01_eda.py)\n\n")
        f.write(f"- Missing / non-finite values in .track files: {int(inv.missing_values.sum())}\n")
        f.write(f"- Non-increasing timestamps: {int(inv.non_increasing_ts.sum())}\n")
        f.write(f"- Mean fraction of 1 s sampling steps: {inv.frac_dt_1s.mean():.3f}\n")
        f.write(f"- Trips are spread worldwide (lon {inv.lon_min.min():.1f}..{inv.lon_max.max():.1f}, "
                f"lat {inv.lat_min.min():.1f}..{inv.lat_max.max():.1f}) -> per-trip azimuthal-equidistant projection.\n")
        f.write(f"- Every trip has its own road network (arc ids are trip-local).\n")
        f.write(f"- Reference-route continuity (arc[i].to == arc[i+1].from): mean {inv.route_continuity.mean():.4f}, "
                f"min {inv.route_continuity.min():.4f}\n")
        f.write(f"- Isolated nodes total: {int(inv.isolated_nodes.sum())}; zero-length arcs total: {int(inv.zero_length_arcs.sum())}\n")
        f.write("\n## Largest time gaps\n\n" + gaps[["partition", "trip", "dt_max", "n_gaps_gt_10s"]].to_markdown(index=False) + "\n")
        f.write("\n## Trips where GPS disagrees with the reference route (>30% of points >50 m away)\n\n")
        f.write(bad[["partition", "trip", "n_points", "gps_route_dist_median", "frac_gps_gt_50m_from_route"]].to_markdown(index=False) + "\n")
        f.write("\nDecision: labels further than LABEL_MAX_DIST_M from the aligned reference arc are treated as unreliable and "
                "excluded; trips with less than MIN_TRIP_LABEL_FRAC reliable labels are excluded from training.\n")
    print("EDA written to", OUT)


if __name__ == "__main__":
    main()

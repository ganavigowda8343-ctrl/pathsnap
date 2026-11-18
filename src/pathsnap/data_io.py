"""Reading the raw PathSnap files.

Raw format (verified during EDA):
  <id>.track : lon  lat  t_seconds       (whitespace separated, no header)
  <id>.nodes : lon  lat                  (row index = node id, 0-based)
  <id>.arcs  : from_node  to_node        (row index = arc id, 0-based, directed)
  <id>.route : arc_id                    (ground-truth arc sequence, in order)

The prediction pipeline only ever calls `load_trip` (no route).  The route is
loaded separately by `load_route`, which is only used for labelling/evaluation.
"""
from pathlib import Path
import re
import numpy as np

TRIP_ID_RE = re.compile(r"^\d{8}$")


def list_trips(part_dir):
    part_dir = Path(part_dir)
    if not part_dir.exists():
        return []
    return sorted(p.name for p in part_dir.iterdir() if p.is_dir() and TRIP_ID_RE.match(p.name))


def _f(trip_dir, ext):
    trip_dir = Path(trip_dir)
    return trip_dir / f"{trip_dir.name}.{ext}"


def load_trip(trip_dir):
    """Load GPS track + road network (everything the predictor may see)."""
    track = np.loadtxt(_f(trip_dir, "track"), dtype=np.float64, ndmin=2)
    nodes = np.loadtxt(_f(trip_dir, "nodes"), dtype=np.float64, ndmin=2)
    arcs = np.loadtxt(_f(trip_dir, "arcs"), dtype=np.int64, ndmin=2)
    # keep chronological order (timestamps were verified strictly increasing,
    # but sort defensively in case an unseen trip differs)
    order = np.argsort(track[:, 2], kind="stable")
    track = track[order]
    return {"trip_id": Path(trip_dir).name, "track": track, "nodes": nodes, "arcs": arcs}


def load_route(trip_dir):
    """Reference arc sequence. SERVER-SIDE / LABELLING ONLY."""
    return np.loadtxt(_f(trip_dir, "route"), dtype=np.int64, ndmin=1).ravel()

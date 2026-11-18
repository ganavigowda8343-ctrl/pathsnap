"""Road network in a local metric projection + spatial index + routing.

Trips are located all over the world (lon -100..137), so a single UTM zone is
not appropriate.  Each trip is projected with an azimuthal-equidistant
projection centred on the trip's median GPS position: distances from the centre
are exact and distortion over a city-scale area (<~50 km) is negligible.
"""
import heapq
import numpy as np
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree

from . import config as C


class LocalProjection:
    def __init__(self, lon0, lat0):
        crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m")
        self.fwd = Transformer.from_crs(4326, crs, always_xy=True)
        self.inv = Transformer.from_crs(crs, 4326, always_xy=True)

    def to_xy(self, lon, lat):
        x, y = self.fwd.transform(lon, lat)
        return np.column_stack([x, y])


def point_segment(P, A, B):
    """Distance from points P to segments A-B (all (n,2)). Returns dist, t (clamped), t_raw."""
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(-1), 1e-9)
    t_raw = ((P - A) * AB).sum(-1) / L2
    t = np.clip(t_raw, 0.0, 1.0)
    proj = A + t[..., None] * AB
    return np.linalg.norm(P - proj, axis=-1), t, t_raw


class RoadNetwork:
    def __init__(self, nodes_lonlat, arcs, proj: LocalProjection, focus_xy=None):
        self.nodes_lonlat = nodes_lonlat
        self.arcs = arcs
        self.n_nodes = len(nodes_lonlat)
        self.xy = proj.to_xy(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        self.A = self.xy[arcs[:, 0]]
        self.B = self.xy[arcs[:, 1]]
        d = self.B - self.A
        self.length = np.linalg.norm(d, axis=1)
        self.bearing = np.arctan2(d[:, 1], d[:, 0])  # radians, math convention
        self.out_deg = np.bincount(arcs[:, 0], minlength=self.n_nodes)
        self.in_deg = np.bincount(arcs[:, 1], minlength=self.n_nodes)
        key = arcs[:, 0] * self.n_nodes + arcs[:, 1]
        rkey = arcs[:, 1] * self.n_nodes + arcs[:, 0]
        self.has_reverse = np.isin(rkey, key)
        # CSR adjacency for routing (node -> outgoing arcs)
        order = np.argsort(arcs[:, 0], kind="stable")
        self.adj_arc = order
        self.adj_to = arcs[order, 1]
        self.adj_w = self.length[order]
        self.adj_ptr = np.searchsorted(arcs[order, 0], np.arange(self.n_nodes + 1))
        self._build_index(focus_xy)
        self._sp_cache = {}

    # ---------------- spatial index ----------------
    def _build_index(self, focus_xy):
        sel = np.arange(len(self.arcs))
        if focus_xy is not None and len(focus_xy):
            lo = focus_xy.min(0) - (C.CANDIDATE_RADIUS_M + 50)
            hi = focus_xy.max(0) + (C.CANDIDATE_RADIUS_M + 50)
            amin = np.minimum(self.A, self.B)
            amax = np.maximum(self.A, self.B)
            sel = np.where((amax[:, 0] >= lo[0]) & (amin[:, 0] <= hi[0]) &
                           (amax[:, 1] >= lo[1]) & (amin[:, 1] <= hi[1]))[0]
        n_samp = np.maximum(np.ceil(self.length[sel] / C.DENSIFY_STEP_M).astype(int), 1) + 1
        arc_of = np.repeat(sel, n_samp)
        # fraction along arc for each sample
        starts = np.cumsum(n_samp) - n_samp
        frac = (np.arange(n_samp.sum()) - np.repeat(starts, n_samp)) / np.repeat(n_samp - 1, n_samp)
        pts = self.A[arc_of] + frac[:, None] * (self.B[arc_of] - self.A[arc_of])
        self.sample_arc = arc_of
        self.tree = cKDTree(pts) if len(pts) else None

    def candidates(self, P, radius=C.CANDIDATE_RADIUS_M, k=C.MAX_CANDIDATES):
        """For each point in P (n,2) return (n,k) arrays of arc ids (-1 = none) and distances."""
        n = len(P)
        if self.tree is None or n == 0:
            return np.full((n, k), -1), np.full((n, k), np.inf)
        kq = min(C.KD_NEIGHBOURS, len(self.sample_arc))
        dd, ii = self.tree.query(P, k=kq, distance_upper_bound=radius + C.DENSIFY_STEP_M)
        dd = dd.reshape(n, -1); ii = ii.reshape(n, -1)
        valid = np.isfinite(dd)
        arc = np.where(valid, self.sample_arc[np.minimum(ii, len(self.sample_arc) - 1)], -1)
        # exact point-to-segment distance
        safe = np.maximum(arc, 0)
        dist, _, _ = point_segment(P[:, None, :], self.A[safe], self.B[safe])
        dist = np.where(valid, dist, np.inf)
        # remove duplicate arcs within a row
        o = np.argsort(arc, axis=1, kind="stable")
        arc_s = np.take_along_axis(arc, o, 1)
        dup = np.zeros_like(arc_s, dtype=bool)
        dup[:, 1:] = arc_s[:, 1:] == arc_s[:, :-1]
        dist_s = np.take_along_axis(dist, o, 1)
        dist_s[dup | (arc_s < 0)] = np.inf
        dist_s[dist_s > radius] = np.inf
        o2 = np.argsort(dist_s, axis=1, kind="stable")[:, :k]
        arc_k = np.take_along_axis(arc_s, o2, 1)
        dist_k = np.take_along_axis(dist_s, o2, 1)
        arc_k[~np.isfinite(dist_k)] = -1
        if arc_k.shape[1] < k:  # pad
            pad = k - arc_k.shape[1]
            arc_k = np.pad(arc_k, ((0, 0), (0, pad)), constant_values=-1)
            dist_k = np.pad(dist_k, ((0, 0), (0, pad)), constant_values=np.inf)
        return arc_k, dist_k

    # ---------------- routing ----------------
    def shortest_from(self, src, cutoff):
        """Bounded Dijkstra from node `src`. Returns dict node -> distance (<= cutoff). Cached."""
        hit = self._sp_cache.get(src)
        if hit is not None and hit[0] >= cutoff:
            return hit[1]
        dist = {src: 0.0}
        heap = [(0.0, src)]
        ptr, to, w = self.adj_ptr, self.adj_to, self.adj_w
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf):
                continue
            for e in range(ptr[u], ptr[u + 1]):
                nd = d + w[e]
                if nd <= cutoff:
                    v = int(to[e])
                    if nd < dist.get(v, np.inf):
                        dist[v] = nd
                        heapq.heappush(heap, (nd, v))
        if len(self._sp_cache) > 50000:
            self._sp_cache.clear()
        self._sp_cache[src] = (cutoff, dist)
        return dist

    def path_arcs(self, src, dst, cutoff=3000.0):
        """Arc ids of the shortest directed path src -> dst (None if not found within cutoff)."""
        if src == dst:
            return []
        dist = {src: 0.0}; pred = {}
        heap = [(0.0, src)]
        ptr, to, w, aid = self.adj_ptr, self.adj_to, self.adj_w, self.adj_arc
        while heap:
            d, u = heapq.heappop(heap)
            if u == dst:
                break
            if d > dist.get(u, np.inf):
                continue
            for e in range(ptr[u], ptr[u + 1]):
                nd = d + w[e]
                v = int(to[e])
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd; pred[v] = int(aid[e])
                    heapq.heappush(heap, (nd, v))
        if dst not in pred:
            return None
        path, n = [], dst
        while n != src:
            a = pred[n]; path.append(a); n = int(self.arcs[a, 0])
        return path[::-1]

    def arcs_connected(self, a, b):
        """True when arc b can directly follow arc a (or it is the same arc)."""
        return a == b or self.arcs[a, 1] == self.arcs[b, 0]

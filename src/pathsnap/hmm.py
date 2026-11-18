"""HMM map-matching with Viterbi decoding.

Hidden states   : at time step i, the candidate arcs of GPS observation i
                  (each state keeps its arc id, endpoints and projected position
                  `along` the arc -> candidate-to-segment correspondence is preserved).
Emission        : log P(obs_i | arc) := w_em * log( p_i(arc) / sum_c p_i(c) )
                  where p is the selected classifier's probability.
Transition      : Newson & Krumm style. For candidates a (at i) and b (at i+1)
                    d_route = network distance from proj(a) to proj(b)
                    d_gc    = straight-line distance between GPS points i and i+1
                    log T   = -|d_route - d_gc| / beta
                  d_route uses bounded Dijkstra on the directed road graph, so
                  a transition is only possible if b is reachable from a.
Viterbi         : delta_{i+1}(b) = log E_{i+1}(b) + max_a [delta_i(a) + log T(a,b)]
                  with back-pointers; solved in log-space.
Breaks / gaps   : if the time gap or jump between observations is too large, or no
                  candidate of i+1 is reachable from any candidate of i (all
                  transitions -inf), the chain is cut: the current segment is
                  back-tracked and a new chain starts from the emissions alone.
                  Observations with no candidates get no prediction (-1) and also
                  cut the chain.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class HMMConfig:
    beta: float = 10.0           # metres; tolerance of route-vs-straight-line distance mismatch
    emission_weight: float = 1.0
    max_gap_s: float = 60.0      # cut chain when dt exceeds this
    max_jump_m: float = 1500.0   # cut chain when consecutive GPS points are further apart
    search_slack_m: float = 150.0  # Dijkstra cutoff = 2*d_gc + slack
    same_arc_back_tol_m: float = 15.0  # allow small backwards jitter on the same arc

    def to_dict(self):
        return asdict(self)


def _groups(obs):
    """Start/end indices of each observation's candidate block (rows sorted by obs)."""
    change = np.flatnonzero(np.diff(obs)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(obs)]
    return obs[starts], starts, ends


def viterbi_decode(cand, prob, P, t, net, cfg: HMMConfig):
    """
    cand : DataFrame with columns obs, arc, u, v, along_m (sorted by obs)
    prob : classifier probability per candidate row
    P, t : projected GPS points (n,2) and timestamps (n,)
    net  : RoadNetwork (for lengths and shortest paths)
    Returns predicted arc per observation (-1 if none) and number of chain breaks.
    """
    n = len(P)
    pred = np.full(n, -1, dtype=np.int64)
    obs = cand["obs"].to_numpy()
    if len(obs) == 0:
        return pred, 0
    arcs_all = cand["arc"].to_numpy(); u_all = cand["u"].to_numpy(); v_all = cand["v"].to_numpy()
    along_all = cand["along_m"].to_numpy()
    obs_ids, starts, ends = _groups(obs)
    length = net.length

    # emission (normalised per observation)
    logE = []
    for s, e in zip(starts, ends):
        p = np.clip(prob[s:e], 1e-9, None)
        logE.append(cfg.emission_weight * np.log(p / p.sum()))

    breaks = 0
    seg_obs, seg_bp = [], []  # current chain: observation ids and back-pointers
    delta = None

    def flush(final_delta):
        nonlocal seg_obs, seg_bp
        if not seg_obs:
            return
        k = int(np.argmax(final_delta))
        for j in range(len(seg_obs) - 1, -1, -1):
            g = seg_obs[j]
            pred[obs_ids[g]] = arcs_all[starts[g] + k]
            if j > 0:
                k = seg_bp[j][k]
        seg_obs, seg_bp = [], []

    for g in range(len(obs_ids)):
        if delta is None:
            delta = logE[g].copy(); seg_obs = [g]; seg_bp = [None]
            continue
        gp = seg_obs[-1]
        i0, i1 = obs_ids[gp], obs_ids[g]
        dt = t[i1] - t[i0]
        dgc = float(np.linalg.norm(P[i1] - P[i0]))
        if dt > cfg.max_gap_s or dgc > cfg.max_jump_m:
            flush(delta); breaks += 1
            delta = logE[g].copy(); seg_obs = [g]; seg_bp = [None]
            continue
        sa, ea = starts[gp], ends[gp]
        sb, eb = starts[g], ends[g]
        cutoff = 2.0 * dgc + cfg.search_slack_m
        logT = np.full((ea - sa, eb - sb), -np.inf)
        for ia in range(ea - sa):
            ra = sa + ia
            a = arcs_all[ra]
            rem_a = length[a] - along_all[ra]
            sp = None
            for ib in range(eb - sb):
                rb = sb + ib
                b = arcs_all[rb]
                if a == b and along_all[rb] >= along_all[ra] - cfg.same_arc_back_tol_m:
                    d_route = abs(along_all[rb] - along_all[ra])
                else:
                    if sp is None:
                        sp = net.shortest_from(int(v_all[ra]), cutoff)
                    dn = sp.get(int(u_all[rb]))
                    if dn is None:
                        continue
                    d_route = rem_a + dn + along_all[rb]
                logT[ia, ib] = -abs(d_route - dgc) / cfg.beta
        scores = delta[:, None] + logT
        best_prev = np.argmax(scores, axis=0)
        best = scores[best_prev, np.arange(scores.shape[1])]
        if not np.isfinite(best).any():
            flush(delta); breaks += 1
            delta = logE[g].copy(); seg_obs = [g]; seg_bp = [None]
            continue
        delta = best + logE[g]
        delta = delta - np.max(delta[np.isfinite(delta)])  # numerical stability
        seg_obs.append(g); seg_bp.append(best_prev)
    flush(delta)
    return pred, breaks

"""Deriving one reference arc per GPS observation from the .route file.

The .route file gives the ORDERED arcs the vehicle traversed, but not which GPS
point belongs to which arc (there are ~5x more points than route arcs).  We
solve a monotone alignment with dynamic programming:

    each point i is assigned a route position j(i), with j non-decreasing in i,
    minimising  sum_i  min(dist(p_i, arc_route[j(i)]), CAP)

Monotonicity respects the temporal order of the route (so loops / revisits of
the same road are handled correctly), and skipping short route arcs that
received no GPS point is free.  Recurrence:

    D[i, j] = c[i, j] + min_{k <= j} D[i-1, k]

computed row by row with a cumulative minimum (O(n*m) time, O(n*m) int16 memory
for back-pointers).
"""
import numpy as np
from . import config as C
from .network import point_segment


def align_points_to_route(P, route, net):
    n, m = len(P), len(route)
    A, B = net.A[route], net.B[route]
    idx_dtype = np.int16 if m < 32767 else np.int32
    bp = np.empty((n, m), dtype=idx_dtype)
    ar = np.arange(m)
    prev = None
    for i in range(n):
        c, _, _ = point_segment(P[i][None, :], A, B)
        c = np.minimum(c, C.ALIGN_COST_CAP_M)
        if prev is None:
            cur = c
            bp[i] = ar
        else:
            run = np.minimum.accumulate(prev)
            arg = np.maximum.accumulate(np.where(prev == run, ar, 0))
            cur = c + run
            bp[i] = arg
        prev = cur
    j = np.empty(n, dtype=np.int64)
    j[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        j[i - 1] = bp[i, j[i]]
    label_arc = route[j]
    dist, _, _ = point_segment(P, net.A[label_arc], net.B[label_arc])
    return label_arc, j, dist


def derive_labels(P, route, net):
    """Return label arc per point (-1 where the reference is unreliable) and the distance."""
    label_arc, pos, dist = align_points_to_route(P, route, net)
    label = np.where(dist <= C.LABEL_MAX_DIST_M, label_arc, -1)
    return label, dist

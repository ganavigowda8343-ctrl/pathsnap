"""Central configuration: paths and pipeline constants.

Every constant here is a design decision that the team should be able to justify
(see README / report). Values were chosen from the exploratory analysis of the
TRAINING partition only.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA_DIR", ROOT / "data"))
TRAIN_DIR = DATA_DIR / "train"
STUDENT_TEST_DIR = DATA_DIR / "student_test"
# Evaluator trips are only placed on the API laptop at evaluation time.
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", DATA_DIR / "evaluator_test"))

CACHE_DIR = ROOT / "data_cache"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
CONFIGS_DIR = ROOT / "configs"

SEED = 42

# --- Candidate generation -------------------------------------------------
# Median GPS-to-route distance on training trips is ~5 m and the 99th
# percentile of "good" trips is well below 60 m, so a 60 m search radius
# captures almost all true segments while keeping candidate sets small.
CANDIDATE_RADIUS_M = 60.0
MAX_CANDIDATES = 10
DENSIFY_STEP_M = 10.0          # spacing of sample points along arcs for the KD-tree
KD_NEIGHBOURS = 64             # sample points looked up per GPS observation

# --- Label derivation -----------------------------------------------------
# Points whose aligned reference arc is further than this are treated as
# unreliable labels (reference route / GPS disagreement) and excluded from
# training and from accuracy computation.
LABEL_MAX_DIST_M = 60.0
ALIGN_COST_CAP_M = 200.0
# Trips with fewer reliable labels than this are excluded from TRAINING
# (EDA: train 00000054 and 00000033 - GPS mostly >100 m from their route).
MIN_TRIP_LABEL_FRAC = 0.5

# --- Feature engineering --------------------------------------------------
HEADING_HALF_WINDOW = 2        # heading from p[i-2] -> p[i+2]
CONTEXT_OFFSETS = (-3, -2, -1, 1, 2, 3)
MIN_HEADING_DISPLACEMENT_M = 3.0

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cand",
    "t_frac", "beyond_m", "log_arc_len",
    "heading_valid", "cos_heading", "abs_heading_deg",
    "speed_mps", "dt_prev", "dt_next",
    "ctx_mean_dist", "ctx_max_dist", "ctx_mean_minus_min", "ctx_rank",
    "has_reverse", "out_deg", "in_deg",
]

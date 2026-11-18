# PathSnap – Viva Preparation Guide

Every team member must be able to answer these in their own words. Each answer
points to the file that implements it. Open the code while you study.

---

## 0. The 60-second pitch (practise saying it aloud)

> GPS points are noisy, so they drift off the road. PathSnap snaps each GPS point
> to the road segment the car was really on.
>
> 1. **Candidates.** For every GPS point we take up to 10 road segments within 60 m.
> 2. **Classifier.** We compute features (distance, heading agreement, how close the
>    neighbouring points are to the same road, …). An XGBoost classifier then gives
>    each candidate a probability of being the true road.
> 3. **HMM/Viterbi.** Picking the best candidate for each point on its own can make
>    the car jump between unconnected roads. An HMM uses those probabilities as
>    emissions and road-network distances as transitions. Viterbi then finds the
>    most likely *connected* sequence.
> 4. **Gap filling.** Gaps between consecutive matched roads are filled with
>    shortest paths.
>
> On the 10 student-test trips, Top-1 accuracy rises from 89.7 % (classifier
> alone) to 93.6 % with the HMM, and route F1 is 96.7 %. It is deployed as a
> FastAPI backend on Laptop 1 and a browser UI on Laptop 2.

---

## 1. Dataset (`scripts/01_eda.py`, `src/pathsnap/data_io.py`)

**Q: What files does each trip have?**
- `.track`: longitude, latitude, time in seconds (about 1 Hz).
- `.nodes`: longitude and latitude of road nodes; the row number is the node ID.
- `.arcs`: from_node, to_node. These are **directed** road segments, and the row number is the arc ID.
- `.route`: the ordered list of arc IDs the car really drove (the ground truth).

**Q: How many trips and points?**
80 train trips (208,417 points), 10 student-test trips (19,160 points) and 10 evaluator
trips (unseen). The median trip has about 1,500 points and lasts about 30 min.

**Q: What surprised you in the data?**
1. **Trips are all over the world** (longitude −100 to 137, latitude −35 to 62). So you can't use one UTM zone.
2. **Every trip has its own road network.** Arc IDs only mean something inside one trip.
3. **`.route` has no per-point labels.** There are about 6.7× more GPS points than route arcs.
4. Two training trips have unreliable labels:
   - **54:** the car spends about 4.7 h moving inside a ~1 km area that has no mapped roads (87 m from any road).
   - **33:** the GPS is on mapped roads that are *missing from the reference route*.
5. Data quality: no missing values, no out-of-order timestamps, 94 % of steps exactly 1 s, but a few huge gaps (up to 153,064 s ≈ 42 h in train trip 90).
6. 92 % of arcs have a reverse twin (two-way streets), so **direction matters**.

**Q: Why not use Euclidean distance on longitude/latitude?**
Degrees aren't metres. A degree of longitude shrinks with latitude: about 111 km at
the equator and about 55 km at 60°. We project each trip with an **azimuthal-equidistant
projection centred on that trip's median GPS point** (`network.py: LocalProjection`).
Distances from the centre are exact, and the distortion over a city-sized area is
negligible.

---

## 2. Labels (`src/pathsnap/labels.py`)

**Q: How did you get a label for every GPS point?**
With a **monotone alignment by dynamic programming**. Each point *i* gets a position
*j(i)* in the route, and *j* may never go backwards in time. We minimise the total
point-to-arc distance (capped at 200 m):

`D[i, j] = cost[i, j] + min over k ≤ j of D[i−1, k]`

The `min over k ≤ j` is a running minimum (`np.minimum.accumulate`), so each row costs O(m).

**Q: Why not just take the nearest route arc?**
Routes revisit the same roads and cross themselves. "Nearest" would give a point to
the wrong pass over that road. Forcing the order to follow time avoids this.

**Q: What if the label is bad?**
- If the aligned arc is more than **60 m** away, the label is marked unreliable (−1) and is not used for training or scoring.
- Trips with fewer than 50 % reliable labels are excluded from training (trips 33 and 54).

---

## 3. Candidates and features (`src/pathsnap/features.py`, `network.py`)

**Q: Why binary classification and not "predict the arc ID"?**
Arc IDs are local to each trip. Arc 2641 in trip 1 has nothing to do with arc 2641 in
trip 2, so a model can't learn arc IDs. Instead we ask, for each (point, candidate arc)
pair: *is this the true arc?* The features don't depend on the trip, so the model
generalises to new cities.

**Q: How are candidates found?**
1. Each arc is sampled every 10 m into points, and the samples go into a KD-tree.
2. For each GPS point we query the 64 nearest samples and compute the exact distance from the point to each segment.
3. We keep up to **10 distinct arcs within 60 m**.

Why 60 m: the median GPS error is about 5 m, and 60 m covers almost all true arcs.
Candidate miss rate: 0.16 % on student-test.

**Q: Explain your features.** (most important first, by XGBoost gain)

| Feature | Meaning | Why |
|---|---|---|
| `dist_minus_min` (45 % of gain) | distance minus the smallest candidate distance for this point | relative closeness matters more than absolute closeness |
| `abs_heading_deg`, `cos_heading` | angle between the GPS direction of travel and the arc direction | separates the two directions of a two-way street and parallel roads |
| `dist_rank`, `dist`, `dist_ratio` | rank and size of the distance | basic geometry |
| `ctx_mean_dist`, `ctx_mean_minus_min`, `ctx_rank` | how close points i−3 … i+3 are to this arc | the car stays on a road for several seconds, which smooths out noise |
| `beyond_m`, `t_frac` | how far past the end of the segment the projection falls | the point may belong to the next arc |
| `speed_mps`, `dt_prev`, `dt_next` | kinematics and sampling gaps | |
| `has_reverse`, `in_deg`, `out_deg`, `n_cand`, `log_arc_len` | topology and context | |

---

## 4. Models (`scripts/03_train_models.py`)

**Q: Which models did you try and why?**
We tried 7 models from different families:
- Logistic Regression (linear)
- Gaussian Naive Bayes (generative)
- KNN (instance-based)
- Random Forest (bagging)
- LightGBM and XGBoost (boosting)
- MLP (neural network)

**Q: How did you validate?**
- **5-fold GroupKFold by trip** on the 78 usable training trips: no trip appears in both the training and validation sides.
- Then each model is refit on all training trips and scored on the 10 student-test trips.
- Evaluator trips were never touched.

**Q: Why group by trip?**
Consecutive GPS points are almost identical. Splitting at random would put
neighbouring seconds of the same drive on both sides, which leaks information and
inflates the scores.

**Q: Which metric picks the model, and why?**
**CV observation-level Top-1 accuracy**: is the highest-scoring candidate the true
arc? That's exactly what the decoder relies on. Tie-break: lower **log loss**,
because the HMM uses the probabilities themselves.

**Q: Results?** (CV Top-1 / student-test Top-1 / test log loss)

| Model | CV Top-1 | Test Top-1 | Test log loss |
|---|---|---|---|
| XGBoost | 0.892 | 0.898 | 0.075 (best) |
| Random Forest | 0.892 | 0.893 | 0.078 |
| LightGBM | 0.890 | 0.897 | 0.080 |
| MLP | 0.886 | 0.884 | 0.084 |
| Logistic Regression | 0.879 | 0.871 | 0.104 |
| KNN | 0.870 | 0.868 | 0.118 |
| Gaussian NB | 0.862 | 0.858 | 0.83 |

**Q: Why XGBoost over Random Forest when their CV Top-1 is almost equal?**
- XGBoost has the best CV log loss (better probabilities for the HMM).
- Its predictions are **4× faster** (0.18 s vs 0.75 s on student-test).
- It trains **7× faster** (16 s vs 107 s).

**Q: Why is Gaussian NB so bad at log loss?**
It assumes the features are independent and normally distributed. Our features are
strongly correlated (`dist`, `dist_rank` and `dist_minus_min` all measure distance)
and heavily skewed. So it counts the same evidence several times and becomes
over-confident.

**Q: Overfitting?** See `results/models/train_vs_cv_vs_test.csv`.
- Train and CV scores are almost equal for Logistic Regression (gap 0.0002) and Naive Bayes (0.0000). These are underfitting / high-bias models.
- Tree ensembles and KNN show a larger train-vs-CV gap: RF 0.031, LightGBM 0.037, XGBoost 0.035, KNN 0.017 (Top-1). That is mild overfitting, but CV and student-test agree, so they still generalise.
- The MLP used early stopping. Its loss curve is in `mlp_training_history.png`.

---

## 5. Bayesian optimisation (`scripts/04_bayes_opt.py`)

**Q: What did you tune and how?**
We used Optuna's **TPE** sampler (Bayesian: it builds probability models of good and
bad regions of the search space) for 30 trials. The objective was the mean 3-fold
GroupKFold Top-1 accuracy on training trips only.

Search space:
- `n_estimators` 100–800
- `learning_rate` 0.01–0.2 (log scale)
- `max_depth` 3–12
- `min_child_weight` 0.5–20 (log)
- `subsample` 0.5–1
- `colsample_bytree` 0.4–1
- `reg_lambda` 0.001–10 (log)

**Q: Did it help?**
Only marginally:
- CV Top-1 rose from 0.8902 to 0.8905, CV Macro F1 from 0.9349 to 0.9352, and test log loss fell from 0.0745 to 0.0739.
- Student-test Top-1 moved from 0.8984 to 0.8969, which is within noise.

The honest conclusion: the default configuration was already near the optimum, and
accuracy is limited by the **features and the label noise**, not the hyperparameters.
The best trial used a small learning rate (0.01) with more trees (600), which is the
classic low-learning-rate trade-off.

**Q: Why 3 folds instead of 5?**
Cost: 30 trials × 5 folds of a 1.2-million-row training set would take more than an
hour. The tuned and untuned models were then compared with 5 folds.

---

## 6. HMM and Viterbi (`src/pathsnap/hmm.py`)

**Q: Hidden states?**
The candidate arcs of each GPS point. Each state remembers its arc ID, its endpoints,
and where the point projects along the arc.

**Q: Emission probability?**
`log E = w · log(p / Σp)`, where p is the XGBoost probability normalised over that
point's candidates. The tuned weight is w = 2.

**Q: Transition probability?** (Newson & Krumm 2009)
- `d_route` is the road-network distance from where point *i* projects onto arc *a* to where point *i+1* projects onto arc *b*. It is computed with a **bounded Dijkstra** on the directed graph.
- `d_gc` is the straight-line distance between the two GPS points.
- `log T = −|d_route − d_gc| / β`, with β = 2 m after tuning.

The idea: in reality, the distance driven between two fixes is close to the straight
line between them. A path that needs a big detour is unlikely.

**Q: Viterbi recurrence?**
`δ_{i+1}(b) = log E_{i+1}(b) + max_a [ δ_i(a) + log T(a, b) ]`. We store the argmax as a
back-pointer and trace back from the best final state. Everything is done in log space
so the numbers don't underflow.

**Q: Missing candidates, gaps, disconnected roads?**
- A point with no candidates gets no prediction (−1).
- If the time gap is over 60 s or the jump is over 1.5 km, the chain is **cut** and a new chain starts from the emissions alone.
- If no candidate at *i+1* can be reached from any candidate at *i* (all transitions −∞), the chain is also cut. This is the "disconnected segments" case.
- Finally, `fill_route` inserts the shortest path between consecutive matched arcs that aren't directly connected.

**Q: How did you tune β and w without leakage?**
Grid search (β ∈ {1, 2, 3, 6, 12, 25, 50}, w ∈ {0.25, 0.5, 1, 2}) on 16 *training*
trips using **out-of-fold** probabilities, i.e. predictions from models that never saw
those trips. Student-test was not used for tuning.

**Q: Did the HMM help?**
Yes, on all 10 student-test trips:
- Top-1 rose from 0.897 to 0.936 (pooled).
- Continuity rose from 0.953 to 0.979, and to 0.999 with gap filling.
- Route F1 rose from 0.899 to 0.967.
- The biggest gain was trip 77: 0.776 → 0.853.

**Q: Where did it not help?**
- Gap filling slightly *lowers* route precision (0.959 → 0.945) because a shortest path isn't always the road actually driven. On trip 00000000, route F1 fell from 0.943 to 0.937.
- Trip 77 is still the weakest (85 %): the classifier flips between neighbouring arcs, and the HMM fixes only part of it.
- When the classifier is confidently wrong for a long stretch, the HMM can't fix it.

---

## 7. API, UI and deployment (`api/main.py`, `ui/index.html`)

**Q: Endpoints?**
- `GET /health`
- `GET /trips` (trip IDs only)
- `POST /predict {"trip_id"}` returns the GPS points, the predicted route polylines, metrics and timings.

**Q: How do you guarantee the ground truth never reaches the model or UI?**
- `PathSnapModel.predict_trip` only calls `load_trip`, which reads `.track`, `.nodes` and `.arcs`.
- `.route` is read only in `evaluate_prediction`, **after** the prediction is finished.
- The response contains only aggregate numbers, never the reference geometry.
- Trip IDs are checked against the folder listing, which also prevents path tricks like `../`.

**Q: Hardcoding?**
None. The API works for any trip folder you point `PATHSNAP_EVAL_DIR` at, and nothing
in the code depends on a specific trip ID.

**Q: Why two laptops?**
It separates the model server from the client, as a real deployment would. The
evaluator can see from the API logs on Laptop 1 that every prediction really comes
from the API.

---

## 8. Tricky follow-ups to rehearse
- *"What would you do with more time?"*
  - Use the turn angle at junctions in the transitions.
  - Train a learning-to-rank model (e.g. LGBMRanker) that ranks candidates within each point.
  - Calibrate the probabilities.
  - Use a K-shortest-paths fill.
- *"Is 94 % good?"* Compare with the 89.7 % baseline. And remember the labels themselves are noisy: the alignment is approximate near junctions.
- *"Why is precision lower after gap filling?"* See section 6.
- *"Why exclude trips 33/54? Isn't that cherry-picking?"* It was decided from label-quality evidence on the training data, before any model results. It is documented in `results/eda/data_quality_notes.md` and `bad_trip_00000054_zoom.png`. Test trips were never filtered.

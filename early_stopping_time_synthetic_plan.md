# Early stopping on time-ordered data: synthetic mechanisms

## Why

On 24 real time-ordered datasets, tuning the number of trees on time splits and refitting (T) was slightly ahead of early stopping on a random 10 % (ER, H4: +0.0035 [+0.0007, +0.0064] average precision) and of the freshest-set pipeline (D_own, H3: +0.0039 [+0.0003, +0.0075]; 99.375 % intervals, Bonferroni over four). Whether the stopping set is random or freshest made no detectable difference (H1), nor did holding out the freshest 10 % (H2). In an exploratory split, chosen after H3 was seen, H3 is larger in the ten low-signal datasets: +0.0069 [+0.0023, +0.0116] against +0.0017 [−0.0008, +0.0043] in the other 14 (95 % intervals over datasets; at 99.375 % the low-signal interval is [−0.0004, +0.0142]). H4 shows no such gradient (+0.0042 against +0.0031), and H1 leans negative in the low-signal datasets (−0.0039 against +0.0002). Real data cannot say why.

Two published results bear on this. Koster and Sigrist (arXiv 2602.05786) found on random splits, without time order, that putting the tree count in the search was as good as or better than early stopping on a loss for classification, when selection used another metric; that is H3's and H4's direction with no drift at all. Cai and Ye (ICML 2025, arXiv 2502.20260) found that validating on the freshest slice before the test did worse than a random split, mostly through training lag: their models are not refitted on the validation rows. Every D arm here refits on all rows of the window, so that lag is removed by construction, and only H2 (EF keeps the lag) and the stopping-set choice (H1) remain of their comparison; a null H1 does not contradict them.

This study builds datasets where signal strength and the kind, strength and timing of drift are set by hand, and runs the same arms on them. It asks:

- **Q1.** Does a generator without drift produce H3's pattern, T ahead of D_own and more so at weak signal? If it does, drift is not needed to produce it. H3 and H4 are split into a count part (same candidate and rows, different tree count), a part for the candidate and its own count and, for H4, a row part, to show where a difference comes from.
- **Q2.** Which drift mechanisms move H1 to H4, measured against the same data without drift, including one where the freshest rows mislead?

It is a separate study. Its results are not pooled with the real datasets and carry no label of the real-data plan. A generator that reproduces a real effect shows that its mechanism is sufficient, not that it operates in the real datasets. A replicate here is one data draw with one model seed; a real dataset value is a mean over five seeds, so only means, not spreads, are compared with real data.

Directions below were set after reduced pilot fits on seeds below 1000 (single candidates or four candidates through the registered fitting functions); no unit with the run's seeds has been fitted.

## Unchanged from the real-data study

The registered `scripts/es_time.py` and `scripts/es_time_prep.py` of SmallDataBenchmarks (sha256 `4cf0cdf3...` and `4dd30a12...`, as pinned in the extension plan) are used unchanged: the 60 random LightGBM candidates (learning rate 0.01 to 0.3), patience 20 on log loss, the cap of 500 trees, the 12 final arms, the geometry of three cuts with three inner validation blocks each, the metric (average precision of the minority class). `run_unit` is called with population "syn", so only the E2 scheme runs. Each dataset has 10,000 rows with time equal to row position; time is passed to the runner and is not a feature. The E2 final windows are rows 2,400-6,000, 3,733-7,333 and 5,067-8,667, their freshest 10 % (the EF stopping sets) start at 5,640, 6,973 and 8,307, and the test blocks are 6,000-7,333, 7,333-8,667 and 8,667-10,000. Every arm ranks its candidates by the mean of its own inner fits' average precision over the three inner validation blocks of the cut (rows 3,600-6,000, 4,933-7,333 and 6,267-8,667); only the final tree count differs in where it comes from.

Contrasts per cut on the E2 final fits: H1 D_T − DR_T, H2 D − EF, H3 T − D_own, H4 T − ER. H3 splits into T − D_T (count: T's candidate and rows, tuned count against the freshest-set count) plus D_T − D_own (candidate and its own count). H4 splits into T − DR_T (count, random-set count), DR_T − DR (candidate and its own count) and DR − ER (rows: all rows against 90 % at the same count). ER − EF, the closest analogue of Cai and Ye's comparison, is descriptive.

## Base generator

Eight numeric features x1..x8, independent standard normal, and two categorical features c1, c2 with eight equiprobable levels (pandas categoricals with categories fixed on all rows). The latent score is

f(x) = x1 + 0.8 x2 + 0.6 x3 + 0.4 x4 + 0.7 x5 x6 + 0.8 · 1[x7 > 0.5] + e(c1),

with level effects e(c1) from N(0, 0.5²); x8 and c2 carry no signal in the base. f is centred and scaled to unit standard deviation on a reference sample of 10⁶ rows. The label is Bernoulli with probability σ(b0 + s f). Rows are independent given time, except through the entities of family H; a no-drift result here is a property of the stopping rules, not of dependent rows.

Signal s and intercept b0 are set jointly, per signal level and prevalence π, on the reference sample: the expected ROC AUC of the true probabilities is 0.64 (weak), 0.80 (medium) or 0.90 (strong), and the mean probability is π. The weak level puts the real study's low-signal measure (below) at about 0.57 (π 0.1) and 0.59 (π 0.3), inside the real low-signal datasets' range of 0.51 to 0.65, with every pilot replicate (seeds 0-39) below 0.65 in all three cuts; the run's share is a diagnostic. The other 14 real datasets have a median of 0.76 on that measure, about the medium level, so the gradient is tested as weak against medium.

Random streams: `numpy.random.SeedSequence(20261121)` is spawned, in this order, into the reference sample, e(c1), e_new(c1) (families E and B) and the per-row term used to calibrate family H. Replicate r (r = 1000..1039) draws x, c and the label uniforms from `default_rng(r)` in every cell. Family H draws the slot of each row from `default_rng([r, 2])`, then standard normal z per entity (`default_rng([r, 2, 1])`) or per row (`default_rng([r, 2, 2])`), with u = τ z, so the two τ levels share their z. Cells of the same replicate share their base draws, and a drift cell differs from its reference only by the drift; U and X cells have the labels of the medium A cell at π 0.3 exactly.

## Families and cells

Unless stated, signal is medium and π = 0.3, and the reference for a drift cell is the family-A cell with the same signal and π 0.3 in the same replicate (families E and B: and the same cut).

**A, stationary.** Nothing drifts. Signal weak, medium, strong × π 0.1, 0.3 (6 cells).

**H, entity turnover.** Ten entities are active at any time. Each of ten slots runs a chain of entities, each active for L consecutive rows, with slot k's chain shifted by k · L / 10 rows; every row is assigned to a slot uniformly at random and belongs to that slot's current entity. At L = 500 an entity has about 50 rows, a file about 209 entities (34 at L = 4,000) and a 3,600-row window about 80. Each entity adds u to the logit, and its id is a categorical feature under LightGBM's defaults (min_data_per_group 100, cat_smooth 10, max_cat_threshold 32): a split groups several entities, and levels unseen in training are routed as missing. τ 1.5 is a strong entity effect (median odds ratio about 4), τ 0.75 a moderate one (about 2). In the "row" control the same id column is present but u is drawn per row, so the id carries no information, though a fitted model still splits on it. For each τ, s and b0 are set on the reference sample with u per row so that the ROC AUC of f alone is 0.80 and the prevalence 0.3, the same in both variants. The two variants match in true-probability AUC, not in learnable signal: in pilots a default model scores +0.015 test AUC higher in the entity cell. Measured at L = 500 (relative to the final window): 36 to 39 % of the EF stopping set's rows belong to entities unseen in the rows EF trains on, none of the random stopping set's, and 81 to 83 % of the test block's rows to entities unseen in the window; an unseen entity's effect cannot be predicted from any feature. Cells: {entity, row} × τ 0.75, 1.5 at L = 500, and one slow-turnover cell (entity, τ 1.5, L = 4,000: 78 to 85 % of test rows from known entities) (5 cells). The reference for H is its row control at the same τ; the slow cell minus the entity cell at L = 500 is the turnover-only comparison, descriptive.

**E, concept switch.** From row ρ on, the label follows f_new: the linear weights of x1..x4 in reverse order (0.4, 0.6, 0.8, 1, all positive), the interaction moved to x5 x8, the threshold to x6 > 0.5, c1 effects from e_new; scaled like f. The correlation of f and f_new on the reference sample is 0.617; prevalence and AUC are unchanged by the switch. It is a switch of the whole labelling function with P(x) fixed, so per-feature drift AUC does not see it; dated real events are usually partial (one feature group or one subpopulation) and visible in P(x). ρ 5,800, 7,150, 8,450 × signal weak, medium (6 cells). Each (ρ, cut) pair falls into one position:

| Position | (ρ, cut) pairs | post-switch rows |
| --- | --- | --- |
| stop | (5,800, 1), (7,150, 2), (8,450, 3) | 56 %, 51 %, 60 % of the stopping set; 7.6 to 9 % of the 2,400 rows that rank the candidates |
| deep | (5,800, 2), (7,150, 3) | 43 % and 42 % of the final window, all of the stopping set, about 64 % of the ranking rows |
| late | (5,800, 3) | 80 % of the final window; descriptive only |
| test | (7,150, 1), (8,450, 2) | none in training; 14 % and 16 % of the test block; descriptive only |
| after | (8,450, 1) | none: identical data and fits to the A reference, asserted in code |

**B, blip.** The family-E generator at medium signal with f_new on 360 rows only, which then revert to f: a short-lived concept (Webb et al. 2016, "blip"; MOA's recurrent concept streams). Cell B_ck puts the blip exactly on the EF stopping set of cut k (rows 5,640-6,000, 6,973-7,333, 8,307-8,667) (3 cells); about 15 % of the blip rows' labels differ from the reference. The registered position "blip" is the mean over the pairs (B_ck, cut k): the freshest-set count is set on rows of a concept the test block does not have, D trains on them and EF does not, and the blip is 15 % of the ranking rows. It is the regime in which the freshest rows should lose. In pilots over the three blip pairs (seeds 200-219 and 300-329, 4 candidates) H1 was about −0.009, H2 −0.003 to −0.004 and H3 +0.010 against the reference, the same sign on every cut, and the EF_T count about 90 against ER_T's 180 in the same fits. P9 and P11 read this one too-short count from two sides (H3's blip effect is almost all its count part); P10 is the only test of the training rows. The other pairs are descriptive: "window" (the blip inside the final window but not in the stopping set), "test" (the blip in the test block), and (B_c3, cut 1), which sees no blip row and is asserted identical to its reference.

**C, intercept (base-rate) drift.** Signal medium; the intercept varies with time so that the mean true probability rises linearly from 0.15 at row 0 to 0.45 at row 10,000 (the intercept is interpolated on a grid of prevalences solved on the reference sample). P(x) is fixed and the ordering of P(y|x) at any one time is unchanged; the change of P(y|x) is the one label shift would give, and clinical models show it as calibration drift. The AUC of the true probabilities falls from 0.81 to 0.80 as the prevalence rises. Average precision rises with prevalence, so C's AP levels are not comparable with other cells'; only differences between arms are.

**U, usage inflation (segment-specific measurement scale).** x1 is not observed. Instead the model sees usage = exp(x1 + γ_g t / n), where γ_g = γ for c2 levels 0-3 and 0 for levels 4-7: the same usage means less in an inflating segment later, and the ordering across segments shifts between the window and the test block. The motivation is product adoption, but the mechanism is a rescaling of one measured feature, not a changing mix of adopter cohorts. In the "normalised" cell, usage is replaced by usage divided by the median usage of the rows with the same c2 level among the 500 preceding rows of the file (among rows 0-499 for the first 500 rows), a label-free feature computed from the past except for the first 500 rows. γ ln 8, ln 32 × {raw, normalised} (4 cells). The reference for a raw cell is the normalised cell at the same γ, since the raw feature also carries less information in-window than x1; in pilots raw usage costs a default model about 0.024 AP against the normalised cell at γ ln 32.

**X, extrapolation.** x2 is observed as x2 + κ t / n and x3 as x3 − (4/3) κ t / n, and the label depends on the observed values with the base weights, so the two shifts cancel in f, the prevalence stays at 0.3 and P(y|x) is stable: as with the logarithm of a nominal amount that inflates and that of the currency's purchasing power that falls, while the label depends on the real amount. κ 3 is the realistic dose (about 5 % of test rows beyond the window's 99th percentile of x2 and 7 % below the 1st percentile of x3); κ 8 is a stress dose (18 % and 27 %) (2 cells). The drift runs along a direction in which f is flat, so the harm comes from trees that split jointly on x2 and x3 and extrapolate through corner leaves, not from a level error: in pilots a default model loses 0.03 AP at κ 8 on the next block and nothing in-window.

27 cells.

## Replicates and compute

40 replicates per cell (r = 1000..1039), 1,080 units of 2,196 fits each, about 230 thread-seconds per unit, about 69 thread-hours in all. The candidates of replicate r are those the registered draw gives for the name "syn" and seed r in every cell; LightGBM's random state and the random stopping split use seed r.

## Analysis

Per unit and contrast, the value is the mean over the three cuts. For families E and B, the unit value of a position is the mean over its (cell, cut) pairs of cell minus reference at that cut; family E pools weak and medium signal with equal weight. For a drift contrast, the replicate-level value is the cell's value minus its reference's in the same replicate. Where a contrast pools cells ("weak" and "medium" pool both π), the replicate-level value is the mean over those cells.

**Check, before the contrasts.** H1 in family A at medium and strong signal, pooled over the four cells, should be near zero: without drift the random and freshest stopping sets differ only in that the random split is stratified by label, which matters least when the signal is clear (at weak signal it is R4). Its 90 % interval is compared with [−0.003, +0.003] (the real study's δ; two one-sided tests at 5 %). If the interval is not contained in it, the check is flagged, and all contrasts are still reported; nothing is withdrawn. Cut 1 of the "after" cells (E at ρ 8,450, B_c3) must equal its reference exactly; the check mode and the analysis assert this.

Registered contrasts are of two kinds: **P**, with a predicted direction, and **R**, a registered question without one. Each is read from a two-sided t-interval over the 40 replicate-level values at level 1 − 0.05/23 (23 contrasts; multiplier 3.283 at 39 df). A P is **supported** when its interval lies on the predicted side of zero, **contradicted** on the other side, **unresolved** otherwise. An R **moves** (up or down) when its interval excludes zero, otherwise **not resolved**. None is dropped or added after the run.

| # | Family | Contrast (replicate-level) | Predicted | Reason |
| --- | --- | --- | --- | --- |
| P1 | A | H3, weak | > 0 | the real low-signal pattern and the published no-drift result; the reproduction test, the mechanism is open |
| P2 | A | H3 weak − H3 medium | > 0 | as P1; medium is where the other real datasets sit |
| P3 | A | H4, weak | > 0 | the real pattern (H4 > 0 in both signal groups) |
| R1 | A | H4 weak − H4 medium | none | the real data show no gradient |
| R2 | A | T − D_T, weak | none | count part of H3; pilots disagree on whether log-loss stops at weak signal come before the AP optimum (fast candidates stop at a few trees) or near it (slow candidates), so the sign is open |
| R3 | A | T − DR_T, weak | none | count part of H4, as R2 |
| P4 | A | DR − ER, weak | > 0 | row part of H4: 10 % more rows at the same count; small (real +0.0011), may stay unresolved |
| R4 | A | H1, weak | none | the random stopping split is stratified by label and the freshest set is not; pilots give a freshest-set count about 0.87 times the random one and a small negative H1, the sign of the real H1 |
| P5 | H | H2 entity − H2 row, τ 1.5 | > 0 | D trains on the freshest rows, whose entities continue into the test block; EF does not |
| P6 | H | H1 entity − H1 row, τ 1.5 | < 0 | the freshest set holds unseen entities and stops earlier, the random set's entities are all known and it stops later; the longer count helps, since a fifth of test rows continue known entities |
| R5 | H | H4 entity − H4 row, τ 1.5 | none | ER's longer count and its 10 % fewer rows pull in opposite directions |
| P7 | E | H2 stop − reference | > 0 | EF trains on no new-concept row; D refits on the window, whose last 183 to 217 rows follow the new concept |
| P8 | E | H2 deep − reference | > 0 | EF drops the 360 freshest rows, all new concept, of a window that is 42 % new concept; D keeps them |
| R6 | E | H1 stop − reference | none | EF_T's count is set on a set that is half new concept, ER_T's on the window's mix |
| R7 | E | H1 deep − reference | none | as R6, the EF_T set being all new concept |
| P9 | B | H1 blip − reference | < 0 | log loss on the blip rows stops improving early, so D_T's count falls to about half of DR_T's, which is set on the window's mix; at medium signal the short count costs average precision |
| P10 | B | H2 blip − reference | < 0 | D trains on the 360 blip rows, whose labels follow a concept the test block does not have; EF does not, at the same count |
| P11 | B | H3 blip − reference | > 0 | D_own takes the short blip count; T keeps a count tuned on validation blocks that are 85 % blip-free |
| R8 | C | H1 − reference | none | the freshest set has a higher prevalence than the rows trained on; in pilots class-weighted candidates stop earlier and unweighted ones barely move, with an AP effect near −0.0007 |
| R9 | U | H2 raw − H2 normalised, γ ln 32 | none | the freshest rows carry the latest usage scale, but they are 360 rows of a 2,467-row lag; an oracle bound puts the effect near +0.002 |
| R10 | U | H4 raw − H4 normalised, γ ln 32 | none | ER's count is set inside the window, T's through candidates scored on later blocks; more trees fit stale segment offsets better or worse |
| R11 | U | H1 raw − H1 normalised, γ ln 32 | none | the freshest set sees the latest usage levels; the effect on its count is open |
| P12 | X | H2 κ 8 − reference | > 0 | the freshest rows are nearest the test range; D trains on them and EF does not, and joint splits on x2 and x3 place corner leaves nearer the test rows |

**Q1 decision.** P1 gives the sign: supported, the generator reproduces H3's sign; contradicted, it gives the opposite sign; unresolved, it does not resolve it. P2 gives the gradient, stated in every case: supported, a gradient in the direction of the exploratory real split; contradicted, the opposite gradient; unresolved, no resolved gradient. The real gradient is itself barely separated from zero (low-signal minus the other 14: +0.0052, Welch 95 % [+0.0001, +0.0102]), so a supported P2 matches a direction, not an established fact. Size draws no verdict: the weak-signal estimate with its 95 % and 99.375 % intervals is printed beside the real low-signal estimate and its intervals at the same levels, and the P2 estimate beside the real difference. R2 and the descriptive candidate part D_T − D_own say where an H3 difference comes from.

Everything else, including the per-cell means of all contrasts with 95 % intervals, the candidate parts, ER − EF, the slow-turnover cell, the "late", "test" and "window" positions, the τ 0.75 and γ ln 8 cells, the row controls and the normalised U cells against the A reference, and all contrasts of each family, is descriptive, labelled so, and draws no claim.

## Diagnostics, per cell, descriptive

- Prevalence per test block and in the stopping sets, and the stopping-set prevalence minus that of the rows trained on; expected ROC AUC of the true probabilities (family H: also of f alone).
- The real study's low-signal measure on every replicate: ROC AUC on each test block of a default LightGBM (`fit_auc` of the preparation script) fitted on the E2 final window; the share of replicates below 0.65 in all three cuts.
- Drift AUC (`screen` of the preparation script, E2 cut 1, replicate 1000): the largest over the features other than the entity id, which feature it is, and the entity id's separately. For family H also the runner's unseen-level shares and the known-entity shares of the test block relative to the window and of the stopping set relative to the rows trained on. For family X the shares of test rows beyond the window's 99th percentile of x2 and below its 1st percentile of x3.
- Oracle staleness per test block: the cost, in average precision and in log loss, of the probabilities the generator gives at the middle of the final window against the true ones, positive when stale is worse (families E and B: the concept in force there; C: that row's intercept, which moves log loss but not average precision; U raw: the latent read from usage with that row's inflation; not computed for normalised U cells). It is zero by construction in A, H and X, and in B outside the "test" pairs. The same cost in log loss on the freshest stopping set, which is what EF's stop reads, is reported beside it and is what sees a blip. Drift AUC sees changes of P(x), oracle staleness changes of P(y|x); neither sees E or B through P(x).
- Per cell, and for the registered E positions and B's blip pairs alone (a cell mixes positions across its cuts): median tree counts per arm; capped share and share at five trees or fewer of the early-stopping fits; for ER_T and EF_T the median of the AP-best iteration minus the stopping iteration and the share stopped before it; the median log ratio of the EF_T to the ER_T count, for class-weighted and unweighted candidates; the replicate standard deviation of the AP of T, ER, EF, D and DR; the share of (replicate, cut) pairs in which two arms pick the same candidate, including T and D_own, whose candidate part is zero when they agree.

## Code and run

`scripts/es_time_syn.py` holds the generator and the runner, in this repository; it is copied into a SmallDataBenchmarks checkout and run from there, as the extension was. It asserts the sha256 of the two registered scripts and that this plan and the scripts are committed and pushed, calls `run_unit("syn", "syn", r, data)` and writes the cell and the sha256 of the generator into every unit; it patches nothing. Units are saved one file per cell and replicate, resumable; the runner stops at a unit with a failed fit, and the analysis asserts one generator version across units. A run log records the time, the sha256 of this plan and the scripts, the package versions and the threads.

Check mode, before the run: on every replicate and cell, without fitting, that each blip is exactly its cut's EF stopping set, the class counts of the E2 geometry (at least 5 minority rows in every freshest stopping set, both classes in every window and block), the "after" data identity, and two label-free generator targets: the mean true probability against π within 0.02 (family C: per tenth of the file within 0.03; family H with entity effects within 0.08 at L = 500, more than 6 standard deviations of a file's mean, and 0.13 at L = 4,000, about 4) and the expected ROC AUC of the true probabilities against the AUC level within 0.02 (families E and B outside the new concept; family H of f alone, within 0.05 and 0.07 with entity effects; not family C). One reduced unit per cell (4 candidates, three cuts) on seed 999, with no failed fit, and the "after" cells' cut-1 fits identical to their reference's. It prints no difference between arms. A failed assert is a generator defect: the run does not start, and no replicate is dropped or replaced.

`scripts/es_time_syn_analyse.py` writes `results/es_time/syn/analysis_syn.txt`, `cells_syn.csv` and `contrasts_syn.csv`, beside `diagnostics_syn.csv` of the runner; it asserts no failed fit and the "after" identity, is checked on fits with planted effects before the run, and runs twice with byte-identical outputs.

Order: this plan and the scripts are committed and pushed; then the check; then the run, the diagnostics and the analysis.

## Out of scope

These can follow in a second wave planned on the results of this one:

- label maturation delay: a row is in a test block for one cut and in a window for the next, so one label column cannot be mature for one and immature for the other; it needs a new runner;
- an arm that stops on the oldest 10 % of the window (Cai and Ye's proposal) or ranks candidates by random folds: new arms in the runner;
- a dated, partial measurement step (a feature zeroed or recoded for one subpopulation at a date), a partial-scope concept switch, and recurring concepts;
- adoption as a changing mix of cohorts with different usage-to-label relations along an S-curve;
- an in-support control for X with the same density shift, without which P12 cannot separate leaving the support from moving inside it;
- entities with observed attributes, returning entities, and known and new entities scored separately in H;
- other families (gradual rotation of the concept, decaying shortcut, growing interaction, sensor noise, seasonality, selection-policy drift), an oracle tree count per cell, E1, other learners, sizes other than 10,000 rows.

## Limits

- The generator is one functional form with ten features; real datasets differ in many ways at once.
- The weak cells are matched to the real low-signal datasets by one measure, not by their structure.
- With 40 replicates the half-width of a contrast is about 0.52 times its replicate-level standard deviation. In reduced pilot fits on seeds below 1000 (single candidates, family A cells and cell-minus-reference pairs) the standard deviations were 0.003 to 0.008 for single cells and 0.007 to 0.016 for differences from a reference, a resolvable size of about 0.0015 to 0.004 and 0.0035 to 0.008. P4, P10, P12, R1, R8 and the family-U contrasts may stay unresolved (P10's pilot effect, −0.003 to −0.004 with a replicate standard deviation of 0.003 to 0.006, sits at the edge), and an unresolved P is not evidence against its mechanism.
- Ten concurrent entities give family H's entity cells a random prior shift: the stopping-set prevalence minus that of the rows trained on has a standard deviation of about 0.06 (row control 0.025), as large as family C's built-in gap. H's contrasts therefore mix unseen entities with a calibration effect like R8's, and with the entity cell's larger learnable signal; the per-cut gap is reported beside the known-entity shares.
- Family E positions differ in post-switch share and in cut; the switch replaces the whole labelling function, though the new concept correlates 0.617 with the old, a moderate change. Family B has one blip length, one signal level and one alignment, exactly the freshest stopping set, the placement least favourable to that set; shorter or earlier blips are diluted and not tested.
- Patience, the stopping metric (log loss) and the 10 % share are those of the registered runner; the study says nothing about other settings.

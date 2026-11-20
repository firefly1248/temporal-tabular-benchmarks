# Early stopping on time-ordered data: extension plan

Committed before any arm is fitted on a new dataset. Preparation, pre-run checks and the counts below come from `scripts/es_time_prep_ext.py` (committed with this plan). Candidates and licences are in `results/es_time/ext/candidates.md`, the candidate pool in `pool.md` and the web-search queries in `search_queries.md` of the same folder. Anything this plan does not specify is exploratory.

## History

The first run is specified in `early_stopping_time_plan.md` and its dated deviations in the SmallDataBenchmarks repository (commit `f2d89f2`); its published results are in `results/es_time/public/` there. On the eight published thin datasets the 98.75 % intervals of all four hypotheses contain zero, with pooled half-widths of 0.006 to 0.018 average precision (AP). One dataset of the first run is not part of the extension. These results are known, and the extension was decided after seeing them: it adds datasets because the first look's intervals were too wide for most labels. That decision is data-dependent and no correction is made for it.

The extension is the second and last look. No dataset is added after it, and the plan changes only by dated entries in a deviations section at the end. No arm has been fitted on any new dataset; the only fits on them so far are the default-LightGBM fits of check 4 and the single-feature fits of the check 5 screens.

**Pinned from the first run.** These files, at SmallDataBenchmarks `f2d89f2`, are not edited. The new code is in new files that import from them: `scripts/es_time_prep_ext.py`, `scripts/es_time_ext.py` and `scripts/es_time_ext_analyse.py`.

| File | sha256 |
|---|---|
| `early_stopping_time_plan.md` (as committed) | `9a5d08ee641733cecdf10d87ebef4a223c1926dd121fd9eb6b57e176e4ea03da` |
| `scripts/es_time.py` | `4cf0cdf382c3875eb51c55d8e667b9f9b59bcc719581beafce58c772327080e9` |
| `scripts/es_time_prep.py` | `4dd30a122f79b7989d2b377b3707f23d57ba3b47eb3e0e5768f842401fbd03e7` |
| `scripts/es_time_analyse.py` | `770f685e504d0df65127274e0af2127e1f243da6fcd25ae3aa8e1aef75d6ec28` |
| `results/es_time/delta.json` | `3d10b820d52ed3ac59de19b417f1ca87513b5e9bbf3ca453e18f687cba2eb293` |
| `results/es_time/prepared/_manifest.json` (a local file of the first run, not tracked in the repository; its hash is recorded in the first run's `run_log.jsonl`) | `687216575424d7e39723161a8d7bc5932d06112459363b6eaf81bf4e50fc9cd4` |

**Pinned for the extension.** The runner asserts these two local files of the preparation step, which are not tracked in the repository.

| File | sha256 |
|---|---|
| `_manifest_ext.json` | `bf4bb759b045471a85037e9169d9c4db7db7d5e70d955bec9d04c1f66a0ea9ce` |
| `_summary_thin_ext.csv` | `6da2fa6fc360587cd8849ec1f61afdad411c4c3fec073d4e0172eb44e7ad4ab9` |

The stored fit files of the eight published datasets are identified by the hashes in the `inputs` of `delta.json`.

## The first plan in brief

Four hypotheses, each a difference of two arms on thin E2, pooled over datasets as the mean of the per-dataset means over three cuts and five seeds:

| | Contrast | What it isolates | Expected |
|---|---|---|---|
| H1 | D_T − DR_T | random vs freshest stopping set, through the tree count only | D_T ahead or equal |
| H2 | D − EF | the cost of holding out the freshest 10 % | D ahead or equal |
| H3 | T − D_own | two full pipelines, each selected under the rule it is deployed with | none |
| H4 | T − ER | tuned `n_estimators` vs early stopping on a random 10 % | none |

Labels, with δ printed beside each: **first fit ahead** (the interval lies above 0), **second fit ahead** (below 0), **no difference found** (it contains 0); **beyond δ** is added to a direction when the interval lies outside (−δ, +δ); **equivalent within δ** when it lies inside. δ is a precision limit estimated from the run's own T results, not a materiality judgement and not an error-controlled claim. Next to every label go the numbers of datasets ahead, behind and tied (below 1e-4) and a robustness flag (removing one dataset changes the sign or crosses ±δ).

## Question

The same as in the first plan: on data ordered by time, what does early stopping cost compared with tuning the number of trees on time splits and refitting on the freshest rows? The extension asks it of more datasets.

More datasets narrow the intervals and make direction labels reachable, but not "equivalent within δ". δ is recomputed at the same level on the same datasets and narrows in step, so equivalence needs the spread of an arm difference across datasets to be below the spread of a seed-pair difference of T, whatever the number of datasets. On the first look's per-dataset values (`results/es_time/public/per_dataset_points.csv`) the standard deviations across the eight datasets are 0.0078 (H1), 0.0151 (H2), 0.0061 (H3) and 0.0047 (H4), against 0.0062 for the seed-pair noise of T. On those spreads H1 and H2 cannot be labelled equivalent at any number of datasets, H3 only with a pooled difference of essentially zero, and H4 only with one below about 0.001 (the first look's estimate is 0.0026). Equivalence is reachable only if the new datasets differ in spread from the first look's.

## Unchanged from the first plan

Arms (T, ER, EF, DR, D, D_own and the T-based refits), the search space (learning rate 0.01 to 0.3), 60 random candidates per dataset and seed, patience 20, the tree cap of 500, native categoricals, the metric (average precision of the minority class), splits (three test blocks, three inner validation blocks, E2 with L = 36 % as primary, E1 as sensitivity), seeds 0 to 4 and the runner assertions. The pre-run checks 1 to 5 apply with their original thresholds, and the original rules for absolute time columns (check 5) and for labels known after the cut (check 3). The thin variant only: 10,000 rows drawn at random with seed 0, indices sorted. There is no dense variant.

**Additions and departures in preparation:**
- Sources too large to read whole (DVSA MOT, NYC TLC, Citi Bike, Lichess) are first reduced by a seeded Bernoulli sample (2 %, 1 %, 1 %, 1 %; seed 0), and the 10,000 rows are drawn from it. A uniform subset of a uniform sample is a uniform subset of the full file, so the thin variant keeps its definition; only the full-data row count is reduced. NYC 311 uses the requests whose `unique_key` ends in 00, a fixed 1 % of ids that carry no information about the request (not a seeded draw).
- A categorical level with fewer than 100 rows in the prepared data is recoded to "other" in the columns named in each builder of `es_time_prep_ext.py`: austin breed and color; kickstarter category; chicago-food facility and inspection type; seattle-stops initial call type, squad, beat, sector and precinct; london-fire property type, station ground, ward and postcode district; dvsa-mot make, model, colour and postcode area; nyc-collisions vehicle types 1 and 2; citibike start station; lichess time control and event; dota2 game mode and cluster; nyc-311 agency, complaint type, descriptor, location type and zip. The count is over the sample for sampled sources and includes the rows of later test blocks; for a source of a million rows a level of 100 rows is about one thin row, so the recode mostly affects sources of a few hundred thousand rows or fewer. Other high-cardinality columns are not recoded (for example the TLC pickup and drop-off zones, 172 of 239 and 145 of 257 levels under 100 rows in the full sample), and the eight published datasets were not recoded.
- The single-feature screens of check 5 are run on cuts 2 and 3 as well as on cut 1 (the first plan screens cut 1 only), with the same 0.8 limits, so that a column whose coding changes late is seen. The eight published datasets keep their first-plan screens.
- No calendar column is derived except the hour of day (nypd-shooting, stats19, london-fire, nyc-collisions, tlc-yellow, citibike). This departs from the first plan, which keeps cyclic columns (weekday, and month where windows exceed a year): weekday and month are not used in the new datasets (the STATS19 day of week is dropped), unlike the first run's wildlife, which keeps `incident_month`. Absolute time is in `t`.

**Column rules, applied to every new dataset.** Attributes of the event itself are kept even where they are recorded after it (descriptions of a crash, incident, stop, trip or match: vehicle types, road conditions, property category and type, address qualifier, trip distance and fare). Consequences of the event and fields set by its processing are dropped (outcomes, injury counts, police attendance, the stop resolution), as are the label, its records and its near copies (the nflverse dropback flag). Columns set by the process that prepares the label are dropped (austin: neutered or intact status and the name flag, both set by the shelter before the outcome). Absolute-time columns are removed or turned into ages (austin date of birth to age, DVSA first use date to vehicle age, Seattle officer year of birth to officer age, Kickstarter launch date to duration). Outputs of fitted models are dropped when the model was fitted on seasons that include later test blocks (nflverse expected points and win probability; MoneyPuck expected-goal columns). Two datasets keep event attributes recorded after the time column: london-fire (property category, property type and address qualifier, recorded after attendance) and tlc-yellow (the label is the tip, paid at the end of the trip, so trip distance, fare, tolls, extra charge and the drop-off zone are kept).

**Label dates.** A label date is modelled where the label can change more than a day after the time column: dota2 (end of the match, tens of minutes), nyc-311 (three days, by definition of the label) and nypd-shooting (the murder flag can change when a victim dies later; 30 days assumed, the source gives no date). Elsewhere the label is determined within hours of the time column and no lag is modelled: tlc-yellow (the tip, at the end of the trip), london-fire (the false-alarm classification, after attendance), lichess (the end of the game; 0.26 % of the sampled games are correspondence games, which last days). In stats19 a casualty classed slight can be reclassified within 30 days; that is rare and no lag is modelled.

## Population

The population is the **eight published thin datasets** (hotels, wildlife, f1-dnf, nba, elec, nebraska-weather, nycflights, bank-full), whose prepared files and stored fits are reused as they are, plus the **sixteen new datasets** below: **24 datasets**. Before it writes anything, the prep script asserts that the first run's manifest and the thin files of the eight still have the sha256 values the first run recorded. Its own manifest records the input hashes, the download dates (file creation dates on the machine that downloaded them), the file modification dates, the hashes of the seeded sample files and the package versions (numpy 2.4.6, pandas 2.3.3, lightgbm 4.6.0, scipy 1.16.3, scikit-learn 1.7.2).

**Included, new datasets** (thin variant, in the format of the first plan; the share is that of the thin file, "rows" are the rows of the prepared source after the sampling above, and the class scored is the minority class):

| Dataset | Source | Rows (used) | Span | Class scored, share | Minority per test block | Min. per inner block | Min. per fit window | Min. per stopping set, EF / ER | Persistence | ROC AUC time, cuts 1 / 2 / 3 | ROC AUC random, cut 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| austin | Austin open data, Animal Center outcomes | 173,775 (10,000) | 2013-10-01 to 2025-05-04 | adopted, 48.8 % | 677/777/813 | 367 | 1,491 | 149 / 149 | 1.10 | 0.722/0.757/0.806 | 0.730 |
| kickstarter | Kaggle kemical, Kickstarter projects (2018-01 snapshot) | 331,675 (10,000) | 2009-05-29 to 2018-01-02 | successful, 40.7 % | 443/537/578 | 221 | 1,173 | 96 / 117 | 1.04 | 0.702/0.685/0.730 | 0.685 |
| chicago-food | Chicago Data Portal, Food Inspections | 273,488 (10,000) | 2010-01-04 to 2026-11-17 | fail, 22.5 % | 309/339/279 | 144 | 721 | 66 / 72 | 1.01 | 0.604/0.594/0.590 | 0.621 |
| nypd-shooting | NYC Open Data, NYPD shooting incidents (historic) | 29,744 (10,000) | 2006-01-01 to 2024-12-31 | murder, 19.8 % | 258/274/271 | 140 | 667 | 57 / 66 | 1.02 | 0.514/0.551/0.508 | 0.544 |
| seattle-stops | Seattle open data, Terry stops | 35,435 (10,000) | 2019-05-08 to 2026-11-18 | arrest, 23.3 % | 268/263/229 | 134 | 770 | 53 / 77 | 1.01 | 0.663/0.677/0.618 | 0.662 |
| stats19 | UK Department for Transport, STATS19 collisions | 513,801 (10,000) | 2021-01-01 to 2025-12-31 | fatal or serious, 24.6 % | 345/332/357 | 179 | 862 | 78 / 86 | 1.00 | 0.598/0.580/0.585 | 0.606 |
| london-fire | London Datastore, London Fire Brigade incident records | 988,279 (10,000) | 2009-01-01 to 2017-12-31 | false alarm, 48.6 % | 628/639/652 | 373 | 1,729 | 160 / 172 | 1.00 | 0.783/0.772/0.764 | 0.802 |
| dvsa-mot | DVSA anonymised MOT tests 2023 (2 % sample of 32.8M) | 656,492 (10,000) | 2023-01-02 to 2023-12-30 | fail, 23.7 % | 327/302/322 | 177 | 833 | 73 / 83 | 0.99 | 0.676/0.621/0.676 | 0.653 |
| nyc-collisions | NYC Open Data, motor vehicle collisions | 1,158,623 (10,000) | 2012-07-01 to 2026-11-16 | injury or death, 25.6 % | 321/510/585 | 147 | 668 | 54 / 66 | 1.03 | 0.615/0.590/0.614 | 0.643 |
| tlc-yellow | NYC TLC yellow taxi 2023, card trips (1 % sample of 29.9M) | 298,681 (10,000) | 2023-01-01 to 2023-12-31 | no tip, 4.8 % | 74/68/52 | 33 | 162 | 15 / 16 | 1.00 | 0.650/0.647/0.609 | 0.610 |
| nflverse | nflverse play-by-play 2015-2024 | 342,945 (10,000) | 2015-09-10 to 2025-02-09 | run, 41.8 % | 573/554/574 | 325 | 1,450 | 136 / 145 | 1.00 | 0.749/0.766/0.760 | 0.767 |
| moneypuck | MoneyPuck NHL shots 2020-2024 | 564,450 (10,000) | row order, game ids 2020-21 to 2024-25 | goal, 7.2 % | 85/104/83 | 46 | 242 | 12 / 24 | 1.00 | 0.693/0.708/0.668 | 0.669 |
| citibike | Citi Bike trips 2023 (1 % sample of 35.1M) | 350,506 (10,000) | 2023-01-01 to 2023-12-31 | casual rider, 19.2 % | 304/273/221 | 160 | 600 | 65 / 60 | 1.01 | 0.570/0.556/0.606 | 0.568 |
| dota2 | Kaggle devinanzelmo, Dota 2 matches | 50,000 (10,000) | 2015-11-05 to 2015-11-18 | dire win, 48.7 % | 682/624/649 | 370 | 1,748 | 159 / 174 | 1.01 | 0.591/0.555/0.579 | 0.594 |
| lichess | Lichess rated games 2013 (1 % sample of 3.39M) | 33,701 (10,000) | 2012-12-31 to 2013-12-31 | no white win, 48.9 % | 650/659/639 | 379 | 1,723 | 169 / 172 | 1.00 | 0.664/0.687/0.663 | 0.677 |
| nyc-311 | NYC Open Data, 311 service requests 2023 (1 % of ids) | 32,180 (10,000) | 2023-01-01 to 2023-12-31 | slower or open, 30.0 % | 412/433/390 | 230 | 996 | 84 / 99 | 1.03 | 0.938/0.922/0.927 | 0.935 |

All sixteen pass checks 1 to 4.
- **Low signal** (time-split ROC AUC below 0.65 on all three cuts, range 0.51 to 0.65): chicago-food, nypd-shooting, stats19, nyc-collisions, tlc-yellow (cut-1 value 0.6498), citibike and dota2. With f1-dnf, nycflights and bank-full of the first run that is ten of the 24 datasets. As in the first plan, the pooled estimates are also shown without them (14 datasets, 13 degrees of freedom, δ held at its full-data value). This subset is registered, not exploratory.
- **Check 3.** Three datasets carry a label date. The share of a fitting window and of an EF stopping set with a label unknown at the start of the block it feeds is 2.4 % and 23.6 % for dota2, 3.6 % and 36.4 % for nyc-311, and 1.7 % and 17.2 % for nypd-shooting (30 days assumed). All are under the 5 % limit for fitting windows. The stopping-set shares exceed 5 %; the first plan reports that share without limiting it and shows H1 also without such datasets. The H1 subset here therefore excludes dota2, nyc-311, nypd-shooting and the first run's nycflights (EF share 10.3 %): 20 of the 24 datasets.
- **Check 5, absolute time.** No retained numeric feature has a Spearman correlation with row position above 0.13 in absolute value (nflverse `temp` -0.13; kickstarter `duration_days` -0.11; moneypuck `shotAngleAdjusted` 0.11). Coded columns are categorical and outside this check; see the screens below.
- **Short spans.** Six datasets span one year or less (tlc-yellow, citibike, dvsa-mot, nyc-311, lichess: one year each; dota2: 13 days, but 9,945 of its 10,000 thin rows fall on 12 to 18 November 2015, so it is effectively a six-day stream).
- **Dates in November 2026** (chicago-food, nyc-collisions, seattle-stops) are the last records before the download, not a freeze. Those portals update daily, so the prepared files are identified by the sha256 of the downloaded file and its download date.

**Single-feature screens above 0.8, reviewed by hand** (thin variant, all three cuts):
- austin `Breed`, window against test 0.81 on cut 1 (label 0.58): the breeds taken in change over a decade, as the drivers and teams of an era in f1-dnf. Kept; H1 on austin is read against the unseen-level diagnostic. austin `age_days` has label AUC 0.80 on cut 3 (0.69 and 0.74 on cuts 1 and 2): age at outcome is known at the outcome and is a legitimate predictor. Kept.
- seattle-stops `Officer Squad`, window against test 0.82 (label 0.57): squads are reorganised over time. Kept, as above.
- nyc-311 `agency`, `complaint_type`, `descriptor`, `location_type`, label 0.81 to 0.94 on all cuts, drift 0.51 to 0.60: all set by the caller or routed at creation, before any handling, and closing speed depends on them through each agency's service rules; none records the outcome. Kept. nyc-311 has the highest time-split ROC AUC of the population (0.92 to 0.94), so it is also shown without it.
- nypd-shooting `LOC_OF_OCCUR_DESC`, `LOC_CLASSFCTN_DESC` and `LOCATION_DESC`: drift 0.99, 0.98 and 0.92 on cut 3 (the first two missing in every row before 2022, the third in 59 %, all filled afterwards; values in `results/es_time/ext/screens_dropped_columns.csv`). Dropped, as the first plan dropped bank-full `contact`.

Two columns change coding without crossing 0.8 and are kept: stats19 `special_conditions_at_site` (the share of code 0 falls from 0.96 to 0.13 across the fifths of the thin file as the missing code rises from 0 to 0.87; drift 0.79, 0.74 and 0.70 on the three cuts) and nyc-collisions `VEHICLE TYPE CODE 1` ("PASSENGER VEHICLE" at 0.54 and 0.33 of rows in the first two fifths and 0 afterwards, "Sedan" the reverse; drift 0.66, 0.52 and 0.58). Their per-cut drift AUCs are printed beside H1 for these two datasets, because the unseen-level diagnostic cannot see a change in the shares of levels that were already seen. london-fire `PropertyType` (label 0.77 to 0.79), `PropertyCategory` (0.75) and `AddressQualifier` (0.64 to 0.66) are recorded by the crews after attendance, while `t` is the call; they stay below the limit and are kept as event attributes.

**Preparation per dataset.** Time column, label and the decisions that bear on leakage are below. The full lists of dropped columns, each with its reason, are in the comments of `es_time_prep_ext.py`; identifiers, location text and duplicates of kept columns are dropped everywhere.
- austin: time = outcome date and time. Label: adopted. Dropped: date of birth (age at outcome kept), the outcome type and subtype (the label), the neutered or intact status and the name flag (set by the shelter during the stay).
- kickstarter: time = deadline, when the label is known. Label: successful among projects that ended successful or failed. Dropped: the funding columns (pledged, backers: the outcome) and the launch date (kept as the duration in days); rows with launch dates before 2009 (placeholders) are dropped.
- chicago-food: time = inspection date. Label: fail among pass, pass with conditions and fail. Dropped: the violations text (the record of the outcome).
- nypd-shooting: time = occurrence date and time; label date assumed 30 days later. Label: the statistical murder flag. Dropped: perpetrator columns (filled as the investigation proceeds), the three location-description columns (see the screens), the date (time of day kept as `hour`).
- seattle-stops: time = occurred date, rows from 2019-05-08, the date of the first arrest flag. The flag holds no arrest in the 34,159 source rows before that date (2003 to 7 May 2019) and 23 % of the rows from it are arrests. The stop resolution field does record arrests earlier, but it is dropped as the record of the label, so the flag is the label and the cut follows it. Label: arrest. Dropped: stop resolution, weapon type, final call type and frisk flag (all fixed during or after the stop); the officer's year of birth is kept as the officer's age at the stop.
- stats19: collisions of the last five published years, time = collision date and time (hour kept). Label: fatal or serious against slight. Dropped: day of week, the four local authority codes (the district code is -1 in 99.96 % of rows, the others identifier-like), the four columns ending `_historic` (older codings kept beside the current ones), the number of casualties and whether the police attended (after the collision), the derived severity columns.
- london-fire: time = date and time of call (hour kept). Label: false alarm. Dropped: the special service type (filled only for special services, so it answers the label), the stop code description, every attendance, pump and cost column, the number of calls (accumulates after the first call).
- dvsa-mot: 2023 results, time = test date, normal tests with a pass or fail result. Label: fail. Dropped: first use date (the age at test in years kept).
- nyc-collisions: time = crash date, with the hour (hour kept); rows whose date does not parse (a few malformed lines) are dropped. Label: at least one person injured or killed. Dropped: every injury and death count, contributing factors (judged after the event), vehicle types 3 to 5.
- tlc-yellow: 2023, time = pickup, credit-card trips only (cash tips are not recorded), fare above zero (hour kept). Label: no tip, determined at the end of the trip, so the trip's own attributes are kept. Dropped: drop-off time, tip and total amount (the total contains the tip), payment type (constant).
- nflverse: scrimmage plays that were a pass or a run, time = game date (many plays per date; ties broken at random). Label: run. Dropped: every outcome of the play, the dropback flag (it records what the quarterback did after the snap and nearly copies the label), the expected-points and win-probability columns. The pre-game betting lines are kept.
- moneypuck: time = season and game id. The files carry no date; the id order is taken as game order within a season, an assumption (rescheduled games may keep their original ids); ties within a game are broken at random. Label: goal. A whitelist of pre-shot columns is kept; dropped: score columns, every post-shot field, all expected-goal model outputs and the event type (it contains the label).
- citibike: 2023, time = ride start (hour kept). Label: casual rider against member, a status held before the ride. Dropped: end time, end station and end coordinates (known at the end of the ride), station names (the ids stay), rides without a start station.
- dota2: public matches, time = match start, label known at the end of the match. Label: radiant win (dire win is the minority and is scored). Features known before the match only: the ten hero picks (one column per hero: +1 radiant, -1 dire, 0 not picked), game mode and server cluster. Dropped: duration, first blood, tower and barracks status, votes, every per-player statistic. The E2 windows span about two days (one nine days, reaching the sparse head) and the EF stopping sets 3 to 8 hours.
- nyc-311: requests created in 2023, time = creation, label known three days after. Label: closed within three days (the table scores the minority class, slower or open; the horizon was fixed before the data were read; requests still open count as not closed). Features: agency, complaint type, descriptor, location type, zip, borough, channel, latitude and longitude. Nothing set during handling is requested from the source.
- lichess: 2013, time = game start, label known at the end of the game. Label: white win (the table scores the minority class, no white win). Features known at pairing: both ratings, the time control and the event type. Dropped: moves, opening, termination and rating changes (from the played game).

## Selection and exclusions

**Selection.** The candidate order was fixed before any download: nine named datasets first (weatherAUS, no-shows, Chicago crimes, Austin, Kickstarter ordered by deadline, Criteo, Avazu, occupancy, TabReD homesite), then datasets from the pool of `pool.md`, which holds 136 datasets from web searches and catalogue reading. The finalists beyond the fixed nine were chosen from the pool by judgment (an outcome known within days, an open licence, a source process not yet in the set) and examined in an order that was not recorded. The stopping rule, set before any download, was to stop when sixteen datasets had passed every gate. After fifteen had passed, CFPB was tried for a sixteenth place and failed check 3. Kickstarter, whose order the candidate list specifies, then passed under that order, which made sixteen. No further candidate was examined and the pool was not exhausted. SFPD passed the formal checks and was rejected on review, so seventeen datasets passed the formal gates and sixteen are included.

**Licence rule.** A dataset is taken when the owner's licence or use statement allows analysis and publication of aggregate results. It is not taken when the terms give no grant and the metadata or terms warn against the use, forbid copying or publication, or reserve all rights. Chicago crimes and Chicago food inspections share the City's general terms (no grant, revocable); crimes is not taken because its metadata also says the data "should not be used for comparison purposes over time", and food inspections is kept. Cases involving a repackager were decided one by one, not by a single rule. Kickstarter (kemical) and the Kaggle no-shows were taken on the uploader's CC BY-NC-SA 4.0 although the original owner's terms were not opened and no restriction was found on any page read. Dota 2 was taken on the Kaggle repackager's CC BY-SA 4.0 although the Valve terms were not read; the Valve terms that led to rejecting the Steam reviews dataset also cover the match interface, so this is the weakest licence position of the included sets. KKBox (a depositor's CC BY 4.0 on a third-party re-upload, with a site-wide "all rights reserved" notice and unreadable competition rules) and Hong Kong racing (the Hong Kong Jockey Club claims copyright over the source) were not taken, because a page read names a rights holder who claims the rights. The line is a judgment.

**Excluded after the gate, the checks or review**, by stage:

| Dataset | Stage | Reason |
|---|---|---|
| weatherAUS | licence gate | Bureau of Meteorology default terms; the CC BY applies only to Climate Data Online |
| Avazu, kick (Carvana) | licence gate | competition rules unreadable |
| TabReD homesite, TabReD Ecom Offers | licence gate | competition-bound data, rules unreadable; the repository's Apache-2.0 covers code only |
| Steam reviews | licence gate | Valve terms restrict copying and publishing |
| Philadelphia shootings | licence gate | the City reserves all rights |
| Montgomery County traffic violations | licence gate | no explicit grant |
| Hong Kong horse racing, KKBox (two copies), Kickstarter (Web Robots) | licence gate | see the licence rule; the Web Robots files are "All Rights Reserved" |
| Coveo SIGIR eCom | licence gate | access through a form that requires accepting terms |
| Chicago crimes | licence gate | see the licence rule |
| Criteo (display advertising) | not obtained | passes the gate (CC BY-NC-SA 4.0) and the training rows are documented as chronological; the owner's download link returned 404 on 2026-11-20 and the Kaggle copy needs acceptance of the competition rules, which was not given |
| SF Fire dispatch calls | not examined | passes the gate; no outcome column that is not a proxy of the call type |
| Stanford Open Policing Project | not examined | passes the gate; the same source family as Seattle Terry stops (police stops) and overlapping it |
| Lending Club, Home Credit | skipped without testing | outcome horizon of months to years |
| IEEE-CIS | skipped without testing | chargeback lag |
| OpenML airlines | skipped without testing | the same process as nycflights |
| launches, ozone-eighthr, ozone-onehr, eeg-eye-state, cholera, dengue, biotech | skipped without testing | excluded in the first plan |
| occupancy (UCI 357) | checks 2 and 4 | the smallest inner block holds no minority row; time-split ROC AUC 0.997 and a Light-sensor single-feature label AUC of 1.00, the room's light is the occupancy |
| LAPD crime data 2020-2024 | before the checks | the arrest label matures over months; about 80 % of the cases were still under investigation |
| SFPD incident reports | review | passes the formal checks; `Category` (label AUC 0.88) and `Descript` (0.93) record the arrest itself (the arrest share is 0.97 for warrant arrests and 0.98 for traffic violations) and four weak features remain without them. Its thin file is written by the prep script and listed in the extension manifest; it is not in the dataset list of the runner |
| no-shows (Brazil 2016), Olist | check 3 | time = booking and purchase, the moment the features exist, fixed before the check; the labels mature after the appointment and the delivery (Olist also fails check 2: 3 minority rows in the smallest EF stopping set). The orderings by appointment day and delivery date were not tried |
| CFPB Consumer Complaint Database | check 3 | public data "freely available for anyone to use, analyze, and build on"; time = date received; label = closed with relief, taken as known 60 days after the date sent to the company (an assumption); largest share of rows with an unknown label in an E2 fit window 0.18 against the limit of 0.05 |

Thresholds, time columns and assumptions were not changed after a result, with one exception: Kickstarter is ordered by deadline, as the candidate list fixed. A trial ordering by launch date failed check 3 and was replaced by the fixed ordering; no arm was fitted on either.

**Source origin.** Six of the 24 datasets come from New York City sources (the new collisions, 311, NYPD shooting, TLC and Citi Bike, and the first run's nycflights); team sports are nflverse, moneypuck and the first run's nba. The pooled mean weights every dataset equally.

## Primary analysis

H1 to H4 are those of the first plan, on thin E2, pooled over the **24 datasets**: the mean of the per-dataset means (over three cuts and five seeds), with a two-sided t-interval on the 24 per-dataset means (23 degrees of freedom, multiplier 3.0095) at **99.375 %**. The first look's intervals were at 98.75 %. Each look controls its own family of four hypotheses (5 % at the first look, 2.5 % at the extension), so the two looks together spend up to 7.5 %, and the extension is not a 5 % procedure across both looks. "Equivalent within δ" is not error-controlled in either look.

The interval claims the mean over these 24 datasets, treated as a sample of open, binary, time-ordered tables with enough minority rows, with the dataset-to-dataset spread as the only source of variation. It ignores dependence among datasets: the eight published datasets are in both looks (all of the first and a third of the second), and six datasets are New York sources, three of them (tlc-yellow, citibike, nyc-311) from the same year. The test rows of each dataset are one sample.

**δ** is recomputed by the first plan's A/A procedure on the extended population in its own step, before the analysis. For each of the 15 splits of seeds 0 to 4 into two disjoint pairs, take the half-width of the pooled t-interval (24 datasets, 99.375 %) of the difference between T's seed-pair means on thin E2; δ is the root mean square of the 15 half-widths. The step reads only T's rows of the 24 datasets and writes `results/es_time/ext/delta_ext.json` with the hashes of its inputs. The analysis refuses to run without that file and never recomputes it. δ comes from T's test scores, which also enter H3 and H4, and it is held at its extended-population value in every subset analysis. The first look's `delta.json` is not used. Beside each label, the label under the first look's A/A δ of the eight published datasets (0.00727, computed by the same step from the eight stored published fits at the first look's 98.75 % level and written to `delta_ext.json`) is printed as a companion without a verdict.

Labels, the robustness flag, the low-signal subset, the EF-stopping-set subset for H1, the capped-fit rule and the per-dataset tables are those of the first plan. The four E2 thin labels on the extended population are the primary result; each is printed with δ, the half-width, the numbers of datasets ahead, behind and tied, the robustness flag and the estimate without the low-signal datasets.

## Secondary, no labels

- The 16 new datasets alone: the same four estimates and t-intervals at 99.375 %, 15 degrees of freedom (multiplier 3.1771), and the difference between the mean of the eight published datasets and the mean of the sixteen new ones with a Welch t-interval at 99.375 %. No replication verdict is drawn.
- The three New York 2023 datasets (tlc-yellow, citibike, nyc-311) averaged into one unit before pooling: 22 units, 21 degrees of freedom, multiplier 3.0382 at 99.375 %.
- E1 on the 24 datasets, as in the first plan: the same estimates and intervals. E1 shares data, blocks and candidates with E2, so agreement in sign between them is not counted.

## Exploratory, labelled so

No labels; none changes a registered number. On the 24 datasets: the tree-count curve (test AP at every iteration of T's pick, indexed by iteration, with a count tuned on the three inner validation blocks), the inner refit comparisons, the EF stops at 1 to 5 trees (counts, share, and the effect on D and H2), pooled estimates without nyc-311, and the descriptive diagnostics of the first plan. Scatter plots mark the new datasets only to tell them from the published ones.

## Runner

`scripts/es_time_ext.py` imports `scripts/es_time.py` and replaces the dataset list (the 16 new datasets), the paths, the manifest name and the functions that hard-code them: `load` and `run_log`. It reuses the stored fits of the eight published datasets and asserts the hashes of their prepared files, of the first run's manifest, of the registered files in the table above (so the checkout used must contain them unmodified) and of the stored fit files the analysis reads. Its other assertions are the first plan's, applied to the new datasets. The extension scripts are run from a checkout of SmallDataBenchmarks at `f2d89f2` with the extension scripts copied into its `scripts/` folder.

`scripts/es_time_ext_analyse.py` contains the δ step and the analysis. It holds its own copies of the registered functions it needs (interval, pooling, A/A splits, labels), with the level, the datasets and the paths as arguments, and imports nothing that writes to the first run's outputs: the registered `es_time_analyse.delta()` would overwrite `delta.json`, and `half_width` binds 98.75 % as its default argument. `delta_ext.json` records the level 0.99375, and the analysis asserts the level and a t multiplier of 3.0095 at 23 degrees of freedom. The script is committed and pushed, with a self-test on synthetic data, before the full run, and it is not changed after the first test score is read except by a dated deviation. The run log records the sha256 of this plan, of `es_time_prep_ext.py`, of the runner, of the analysis script, of the extension manifest, of the registered files and of the stored fit files.

## Compute

Per dataset: 4,392 fits per seed over E2 and E1, so 21,960 over five seeds and 351,360 over the 16 new datasets. The first run's thin fits on the eight published datasets took 0.34 to 0.77 thread-hours each (mean 0.50, total 4.0), so the new datasets are estimated at 8.0 thread-hours (5.5 to 12.3; dota2 has 112 hero columns and may fall outside). There is no compute budget: the run uses E2 and E1, five seeds and every arm. A smoke run on two named datasets (nyc-311 and dota2: one wide in categoricals, one wide in numeric columns) checks the runner end to end and prints timings and capped shares only, never test scores.

## Out of scope

As in the first plan; also a dense variant, further datasets after this look, and any change to the eight published datasets' fits.

## Limits

- Ten of the 24 datasets are low-signal, so arms can hardly differ there. That pulls pooled estimates towards zero and favours "no difference" and "equivalent".
- The extension is a second look at the same hypotheses, decided after the first, and the pooled estimate over the 24 is not independent of the first.
- Large sources enter through seeded samples or fixed ids, so the full-data row counts in the table are those of the sample. The rare-level recode therefore changes little for them, and the unseen-level diagnostic of the first plan is shown separately for the recoded and the unrecoded datasets.
- Dota 2 is a short stream, so drift inside it is small; it tests early stopping on a short stream rather than across an era. NYC 311 has the highest time-split ROC AUC of the population, set by the complaint fields at creation.
- Class shares move inside several datasets. By fifths of the thin file, the share of the scored class goes from 0.20 to 0.42 in nyc-collisions (0.28 in the fourth fifth), from 0.39 to 0.59 in austin (it becomes the majority in the last fifth), from 0.15 to 0.23 and back to 0.18 in citibike, and from 0.27 to 0.18 in seattle-stops; in kickstarter it runs 0.47, 0.48, 0.31, 0.35, 0.43. The first plan flagged log-loss stopping under such shifts: stopping on a set with a different minority share stops for calibration as well as ranking reasons.
- The 30-day label date of nypd-shooting is an assumption; a rare late reclassification in stats19 and a long game in lichess are not modelled.
- Unread or weaker licences: MoneyPuck (non-commercial use with credit), Citi Bike (non-commercial published analyses), Chicago Food Inspections (no grant, revocable), nflverse (upstream NFL terms not read), Dota 2, Kickstarter and no-shows (see the licence rule), and NYC Open Data (the terms of use page itself could not be read; the city's FAQ and Administrative Code 23-502(d) allow use). The details are in `candidates.md`.
- The 24 datasets are a judgment sample, not a random draw from open time-ordered tables: the finalists were chosen from the pool by judgment and the order in which they were examined was not recorded.
- The first plan's other limits apply: three correlated cuts, patience 20 on small stopping sets, the tree count carried from 90 % to 100 % unscaled, learning rates below 0.01 and ensembles above 500 trees not covered.

## Run conditions

Order of steps: this plan and the prep script are committed and pushed first; then the runner and the analysis script with its self-test; then the run (check mode, the smoke run, the full run); then the results. No run starts before its runner and analysis script are committed and pushed. The run log's start time is compared with their commit times before the results are read, and a run log that predates either commit invalidates the run, which is then repeated. The analysis step runs twice with byte-identical outputs and writes the output hashes to a manifest. A facts sheet ties every quoted number to an output line and is checked by a script.

## Deviations

2026-11-20, before the run.

- Check mode fits a reduced unit (two candidates, three cuts, all arms) of every new dataset and requires that no fit fails. It prints no test score and none has been read. Reduced units of the same kind were fitted on all 16 new datasets in the development of the runner, before the scripts were committed, so the statement in the History section that no arm has been fitted on a new dataset holds only up to those development runs.
- The runner stops at the first unit with a failed fit instead of saving it. Units of the smoke run (seed 0 of nyc-311 and dota2) are units of the full run, so a run-log entry whose inputs differ from the stored units' is refused when the run starts and again by the analysis. A log from which no unit was saved is archived under a new name and not deleted.
- The runner also pins the two preparation files in the table above.
- The analysis script prints, beside H1, the drift AUC and H1 per cut of the two columns that change coding. It prints the unseen-level shares per dataset and for three groups (the published datasets, the new datasets with recoded levels, the other new datasets). Its `forest.csv` carries labels for the 24-dataset block only. Pooled estimates without nyc-311 appear in `analysis_ext.txt` under an exploratory heading. A `reproduce` command checks its functions at the first look's level against the first look's public values.
- The remaining exploratory items listed above are done in a separate script, `scripts/es_time_ext_explore.py`, and are labelled exploratory.

2026-11-21.

- Before the run, the commit that first published this plan was replaced by one in which the companion δ is the A/A δ of the eight published datasets (98.75 %, 7 df) and the Runner paragraph names the two functions the runner replaces. Both changes precede every fit.
- The SmallDataBenchmarks checkout used for the run had uncommitted edits in three files the runner does not read: the first plan, `scripts/es_time_explore.py` and `scripts/es_time_publish.py`. The first plan is checked as committed at `f2d89f2`; the run log lists the three files.
- Run record (UTC): runner and analysis commit 04:40:36, push 04:41:14, check mode 05:34:25, run-log entries 05:35:32 (smoke) and 05:38:56 (run), last unit 07:11:28, δ step 07:12:57, analysis 07:13:05. The run log postdates the commits and the push.
- After the run, the repository history was squashed into one commit. The run log's `extension_commit` (a979aef) names the pushed commit the run started from, which is not in this history; the sha256 entries of the run log tie the run to the three scripts in this commit.

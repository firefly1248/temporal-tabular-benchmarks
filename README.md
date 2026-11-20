# temporal-tabular-benchmarks

Extension of the early-stopping study on time-ordered tabular data: a second look with 16 new datasets.

- `early_stopping_time_extension_plan.md`: the plan, committed before any arm is fitted on a new dataset.
- `results/es_time/ext/`: licence and candidate table, candidate pool, web-search queries, columns dropped by the screens.
- `scripts/es_time_prep_ext.py`: preparation and pre-run checks. It imports `es_time_prep.py` from SmallDataBenchmarks at commit `f2d89f2`.
- `scripts/es_time_ext.py`, `scripts/es_time_ext_analyse.py`: the runner and the δ step and analysis.

Results in `results/es_time/ext/`:

- `analysis_ext.txt`: primary labels, secondary and sensitivity estimates. The blocks "Descriptive differences" and "Diagnostics" are exploratory; the diagnostic tables pool the 24 datasets, except the last one (thread seconds of the new datasets). Companion labels use the A/A δ of the eight published datasets at 98.75 % (7 df) and carry no verdict.
- `forest.csv`, `per_dataset_points.csv`: pooled and per-dataset values per hypothesis and block; labels only for the 24-dataset block.
- `per_dataset_ext.csv`: per dataset, cut and seed; `cut` 0 is the mean over the three cuts, `lo95` and `hi95` are 95 % intervals over the five seeds.
- `stopping_set_gaps_ext.csv`: H1 and H2 per dataset and cut, with the minority-share gap between the freshest 10 % and the test block (`gap_test`) and the older training rows (`gap_older`).
- `delta_ext.json`: δ, its 15 half-widths and the hashes of the fits it read; `outputs_manifest.json`: hashes of the analysis outputs; `run_log_ext.jsonl`: inputs, versions and start time of the smoke run and the full run.

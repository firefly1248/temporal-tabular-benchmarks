"""es_time_ext: runner of the extension. Plan: early_stopping_time_extension_plan.md.

Imports scripts/es_time.py (the registered runner) and replaces its dataset list, its paths, its manifest name and
the functions that hard-code them (load, run_log). The fits of the eight published datasets are reused as
they are; this runner only asserts their hashes. Every fit, arm, split, candidate list and seed is the first
run's. Run from a checkout of SmallDataBenchmarks at f2d89f2 with the extension scripts copied into scripts/.

Usage: python -m scripts.es_time_ext check              pins, hashes, geometry, candidates, one reduced unit
       python -m scripts.es_time_ext smoke              seed 0 of nyc-311 and dota2; timings and capped shares only
       python -m scripts.es_time_ext run [dataset ...]  resumable per dataset and seed, thin E2 and E1, seeds 0 to 4
Environment: ES_THREADS (default 6) as in es_time; ES_EXT_REPO, ES_EXT_PREPARED and ES_EXT_FITS locate the extension
repository, the prepared files and the new fits.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import optuna
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import log_loss

import scripts.es_time as es

PUBLISHED = ["hotels", "wildlife", "f1-dnf", "nba", "elec", "nebraska-weather", "nycflights", "bank-full"]
NEW = ["austin", "kickstarter", "chicago-food", "nypd-shooting", "seattle-stops", "stats19", "london-fire", "dvsa-mot",
       "nyc-collisions", "tlc-yellow", "nflverse", "moneypuck", "citibike", "dota2", "lichess", "nyc-311"]
SMOKE = ["nyc-311", "dota2"]
SDB = es.ROOT
EXT = Path(os.environ.get("ES_EXT_REPO", Path.home() / "temporal-tabular-benchmarks"))
PREPARED = Path(os.environ.get("ES_EXT_PREPARED", Path.home() / ".sdb-runs/es_time_ext/prepared_ext"))
FITS = Path(os.environ.get("ES_EXT_FITS", Path.home() / ".sdb-runs/es_time_ext/fits"))
OUT = EXT / "results/es_time/ext"
FIRST = SDB / "results/es_time"
# The pinned files of the first run (plan, section History), relative to the SmallDataBenchmarks checkout.
PINS = {"scripts/es_time.py": "4cf0cdf382c3875eb51c55d8e667b9f9b59bcc719581beafce58c772327080e9",
        "scripts/es_time_prep.py": "4dd30a122f79b7989d2b377b3707f23d57ba3b47eb3e0e5768f842401fbd03e7",
        "scripts/es_time_analyse.py": "770f685e504d0df65127274e0af2127e1f243da6fcd25ae3aa8e1aef75d6ec28",
        "results/es_time/delta.json": "3d10b820d52ed3ac59de19b417f1ca87513b5e9bbf3ca453e18f687cba2eb293"}
EXT_PINS = {"_manifest_ext.json": "bf4bb759b045471a85037e9169d9c4db7db7d5e70d955bec9d04c1f66a0ea9ce",
            "_summary_thin_ext.csv": "6da2fa6fc360587cd8849ec1f61afdad411c4c3fec073d4e0172eb44e7ad4ab9"}
FIRST_MANIFEST = "687216575424d7e39723161a8d7bc5932d06112459363b6eaf81bf4e50fc9cd4"
SDB_COMMIT = "f2d89f2"  # the checkout of SmallDataBenchmarks; its first plan is pinned as committed there
FIRST_PLAN = "early_stopping_time_plan.md", "9a5d08ee641733cecdf10d87ebef4a223c1926dd121fd9eb6b57e176e4ea03da"
FIRST_VERSIONS = ("numpy", "pandas", "lightgbm", "scipy", "sklearn", "optuna")
SCRIPTS = ("es_time_prep_ext.py", "es_time_ext.py", "es_time_ext_analyse.py")
PLAN = "early_stopping_time_extension_plan.md"
DEV = os.environ.get("ES_EXT_DEV") == "1"  # development: check mode before the commit; a run refuses to start
first_load = es.load  # the first run's loader, kept for the published datasets


def guard_dev():
    """ES_EXT_DEV skips the commit checks. It is for developing the scripts on a scratch copy of the repository (no .git
    folder); a step that writes results refuses it in the real repository."""
    assert not DEV or not (EXT / ".git").exists(), "ES_EXT_DEV in the real repository"


sha256 = es.sha256


def git(*args, repo=EXT):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def stored_fit_files():
    """The stored fit files of the eight published datasets (those of the ninth, excluded dataset are never read)."""
    return {f"{n}_seed{s}.joblib": FIRST / "thin" / f"{n}_seed{s}.joblib" for n in PUBLISHED for s in es.SEEDS}


def git_blob(path):
    return subprocess.run(["git", "-C", str(SDB), "show", f"{SDB_COMMIT}:{path}"], capture_output=True, check=True).stdout


def package_versions():
    return {m.__name__: m.__version__ for m in (np, pd, lightgbm, scipy, sklearn, optuna, joblib)}


def assert_pins():
    """Registered files, the first run's manifest, the published prepared files and stored fits are unchanged; the
    extension scripts in the checkout are the committed ones."""
    for f, h in PINS.items():
        assert sha256(SDB / f) == h, f"{f} differs from the pinned file"
    assert sha256(FIRST / "prepared/_manifest.json") == FIRST_MANIFEST, "the first run's manifest changed"
    prepared = json.load(open(FIRST / "prepared/_manifest.json"))["prepared"]
    for n in PUBLISHED:
        assert sha256(FIRST / f"prepared/{n}_thin.csv") == prepared[f"{n}_thin.csv"], f"{n}: prepared file changed"
    inputs = json.load(open(FIRST / "delta.json"))["inputs"]
    for name, path in stored_fit_files().items():
        assert sha256(path) == inputs[name], f"{name}: stored fit changed"
    assert git("rev-parse", "HEAD", repo=SDB).startswith(SDB_COMMIT), f"the checkout is not at {SDB_COMMIT}"
    assert hashlib.sha256(git_blob(FIRST_PLAN[0])).hexdigest() == FIRST_PLAN[1], "the first plan as committed changed"
    first_versions = next(e["versions"] for e in map(json.loads, open(FIRST / "run_log.jsonl")) if e["pop"] == "thin")
    assert {k: v for k, v in package_versions().items() if k in FIRST_VERSIONS} == first_versions, \
        "package versions differ from the first run's"
    for f, h in EXT_PINS.items():
        assert sha256(PREPARED / f) == h, f"{f} differs from the pinned file"
    manifest = json.load(open(PREPARED / "_manifest_ext.json"))
    assert manifest["published_manifest_sha256"] == FIRST_MANIFEST
    assert manifest["published"] == {f"{n}_thin.csv": prepared[f"{n}_thin.csv"] for n in PUBLISHED}
    assert all(f"{n}_thin.csv" in manifest["prepared"] for n in NEW)
    if DEV:
        return
    for f in SCRIPTS:
        assert sha256(SDB / "scripts" / f) == sha256(EXT / "scripts" / f), f"scripts/{f} in the checkout is not the extension's file"
    assert git("status", "--porcelain", "--", PLAN, "scripts", "results/es_time/ext/candidates.md") == "", \
        "uncommitted changes in the plan, the scripts or the candidate table"
    assert git("rev-list", "--count", "@{u}..HEAD") == "0", "commits not pushed"


def load(name, pop="thin"):
    """Features, label (1 = minority) and time of a prepared file, with its sha256 checked against its manifest."""
    if name in PUBLISHED:
        return first_load(name, pop)
    assert name in NEW and pop == "thin"
    manifest = json.load(open(PREPARED / "_manifest_ext.json"))["prepared"]
    side = json.load(open(PREPARED / f"{name}.json"))
    file = f"{name}_thin.csv"
    assert sha256(PREPARED / file) == manifest[file] == side["sha256"][file], f"{file} differs from its manifest"
    df = pd.read_csv(PREPARED / file)
    if df.t.dtype == object:
        df["t"] = pd.to_datetime(df.t)
    feats = [c for c in df.columns if c not in ("t", "y", "label_known")]
    assert set(feats) <= set(side["features"]), "a feature outside the sidecar list"
    X = df[feats].copy()
    cats = [c for c in side["categorical"] if c in feats]
    for c in cats:
        X[c] = X[c].astype("category")
    return X, df.y.to_numpy(), df.t.to_numpy(), cats


def run_files():
    """sha256 of everything a run depends on: this plan, the extension's scripts and tables, the prepared inputs, the
    registered files and the stored fits of the published datasets."""
    drift = [f"_features_allcuts_{n}.csv" for n in ("stats19", "nyc-collisions")]
    return {PLAN: sha256(EXT / PLAN), **{f"scripts/{f}": sha256(EXT / "scripts" / f) for f in SCRIPTS},
            "results/es_time/ext/candidates.md": sha256(OUT / "candidates.md"),
            "_manifest_ext.json": sha256(PREPARED / "_manifest_ext.json"),
            "_summary_thin_ext.csv": sha256(PREPARED / "_summary_thin_ext.csv"),
            **{f"{n}.json": sha256(PREPARED / f"{n}.json") for n in NEW},
            **{f: sha256(PREPARED / f) for f in drift},
            **{f: sha256(SDB / f) for f in PINS}, "results/es_time/prepared/_manifest.json": sha256(FIRST / "prepared/_manifest.json"),
            "results/es_time/prepared/_summary_thin.csv": sha256(FIRST / "prepared/_summary_thin.csv"),
            **{f"stored fit {n}": sha256(p) for n, p in stored_fit_files().items()}}


def fit_inputs(files):
    """What the stored fits depend on: everything in the hashed files except the plan and the analysis script, which may
    gain a dated deviation after the run."""
    return {k: v for k, v in files.items() if k not in (PLAN, "scripts/es_time_ext_analyse.py")}


def environment():
    return {k: os.environ.get(k) for k in ("ES_EXT_REPO", "ES_EXT_PREPARED", "ES_EXT_FITS")}


def assert_one_run():
    """Every run-log entry (smoke and run) has the same inputs and package versions, and they are the current ones."""
    log = [json.loads(line) for line in open(OUT / "run_log_ext.jsonl")]
    assert log, "no run log"
    now = fit_inputs(run_files())
    for entry in log:
        assert entry["versions"] == log[0]["versions"] and entry["environment"] == environment(), \
            "the run log has entries of different package versions or locations"
        assert fit_inputs(entry["sha256"]) == now, "inputs or scripts changed since the run started"


def run_log(pop):
    """Start of a run: asserts the pins and appends the hashes of everything the run depends on. Stored units were
    written under the earlier entries, so the inputs must be the same; with no stored unit an earlier log is archived."""
    assert not DEV, "ES_EXT_DEV is for check mode only; a run needs the committed scripts"
    assert_pins()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "run_log_ext.jsonl"
    if path.exists() and not list((FITS / "thin").glob("*.joblib")):
        path.rename(OUT / f"run_log_ext.superseded-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "pop": pop, "threads": es.THREADS,
                            "extension_commit": git("rev-parse", "HEAD"), "versions": package_versions(),
                            "python": sys.version, "machine": platform.machine(), "environment": environment(),
                            "checkout_status": git("status", "--porcelain", "--untracked-files=no", repo=SDB).splitlines(),
                            "sha256": run_files()}) + "\n")
    assert_one_run()


def check():
    """Pins, manifest hashes and geometry asserts for the new datasets, candidate bounds, the log-loss stop, and a
    reduced unit (2 candidates, 3 cuts) of every new dataset that must run without errors. Nothing here prints a score."""
    assert_pins()
    assert es.OUT == FITS and es.DATASETS["thin"] == NEW, "the runner is not patched"
    import scripts.es_time_ext_analyse as analysis
    analysis.precheck()
    for name in NEW:
        X, y, t, cats = load(name)
        g, lay = es.layout(t)
        es.check_geometry(t, g, lay)
        print(f"{name}: {len(X)} rows, {X.shape[1]} features ({len(cats)} categorical), minority {y.mean():.3f}: "
              "sha256 and geometry asserts pass")
    for name in NEW:
        for seed in es.SEEDS:
            es.candidates(name, seed)
    print("candidates: learning rate at least 0.01 for every dataset and seed")
    X, y, t, cats = load("nyc-311")
    g, lay = es.layout(t)
    params = next(c for c in es.candidates("nyc-311", 0) if c["class_weight"] is None)
    ctx = dict(X=X, y=y, cats=cats, seed=0, cands=[params], stop_start=g["stop_start"])
    (a, b), (ta, tb) = lay[("e2", 0)]["final"]
    r = es.stopped(ctx, params, (a, b), (ta, tb), "time")
    s = g["stop_start"](a, b)
    model = es.make({**params, "n_estimators": es.MAX_TREES}, 0).fit(X.iloc[a:s], y[a:s])
    ll = [log_loss(y[s:b], model.predict_proba(X.iloc[s:b], num_iteration=k)[:, 1], labels=[0, 1])
          for k in range(1, r["best"] + es.PATIENCE + 1)]
    assert int(np.argmin(ll)) + 1 == r["best"], "early stopping did not stop on log loss"
    print(f"early stop agrees with a manual log-loss argmin on nyc-311 (best iteration {r['best']})")
    for name in NEW:
        df = es.run_unit(name, "thin", 0, load(name), n_cand=2, cuts=(0, 1, 2))
        assert df.ap.notna().all()
        print(f"reduced unit {name}: {len(df)} fits, arms {len(df.arm.unique())}, no errors")


def smoke():
    """Seed 0 of nyc-311 and dota2, saved like any unit and reused by the full run. Prints wall time, thread seconds and
    capped shares only, never a test score."""
    run_log("thin")
    (FITS / "thin").mkdir(parents=True, exist_ok=True)
    for name in SMOKE:
        path = FITS / "thin" / f"{name}_seed0.joblib"
        if path.exists():
            print(f"{name} seed 0: stored, skipped")
            continue
        start = time.time()
        df = es.run_unit(name, "thin", 0, load(name))
        joblib.dump(df, f"{path}.tmp")
        os.replace(f"{path}.tmp", path)
        capped = df[df.arm.isin(es.ES_ARMS)].groupby("stage").best.apply(lambda b: (b > es.CAPPED).mean())
        print(f"smoke {name} seed 0: wall {time.time() - start:.0f} s, thread s {df.sec.sum():.0f} "
              f"({df.sec.sum() / 3600:.3f} thread hours; times 5 seeds {5 * df.sec.sum() / 3600:.2f}), fits {len(df)}, "
              f"errors {(df.error != '').sum()}, capped ES fits inner {capped.get('inner', np.nan):.2f} "
              f"final {capped.get('final', np.nan):.2f}", flush=True)


def install():
    """Replace the first runner's dataset list, paths and hard-coded functions by the extension's. A unit with a failed
    fit stops the run before it is saved: the registered runner records a fit exception as an error row."""
    es.OUT = FITS
    es.DATASETS = {"thin": NEW}
    es.load, es.run_log = load, run_log
    unguarded = es.run_unit

    def run_unit(name, pop, seed, data, **kw):
        df = unguarded(name, pop, seed, data, **kw)
        assert (df.error == "").all(), f"{name} seed {seed}: failed fits\n{df[df.error != ''].error.value_counts()}"
        return df

    es.run_unit = run_unit


if __name__ == "__main__":
    install()
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "check":
        check()
    elif command == "smoke":
        smoke()
    elif command == "run":
        only = sys.argv[2:]
        assert set(only) <= set(NEW), "datasets outside the extension list"
        es.run("thin", only)
    else:
        print(__doc__)

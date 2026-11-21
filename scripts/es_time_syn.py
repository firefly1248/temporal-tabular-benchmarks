"""es_time_syn: synthetic mechanisms for early stopping on time-ordered data. Plan: early_stopping_time_synthetic_plan.md.

Generator of the seven families and a runner that calls the registered `run_unit` of SmallDataBenchmarks unchanged,
with population "syn" (E2 only) and the name "syn" for every cell, so the candidates depend on the replicate only.

Run from a SmallDataBenchmarks checkout with this file copied into its scripts/ folder:
  python -m scripts.es_time_syn check            data asserts on every cell and replicate, one reduced unit per cell
  python -m scripts.es_time_syn run [cell ...]   one file per cell and replicate, resumable; stops on a failed fit
  python -m scripts.es_time_syn diag             per-cell diagnostics -> results/es_time/syn/diagnostics_syn.csv
Environment: ES_SYN_REPO (this repository), ES_SYN_FITS (fit folder), ES_THREADS (concurrent fits, default 6).
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import optuna
import pandas as pd
import scipy
import sklearn
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.metrics import average_precision_score, log_loss

from scripts import es_time as es
from scripts.es_time_prep import fit_auc, geometry, screen

N = 10_000
REPS = range(1000, 1040)
REF_ROWS = 1_000_000
ROOT_SEED = 20261121
AUC = {"weak": 0.64, "medium": 0.80, "strong": 0.90}
SDB = es.ROOT
EXT = Path(os.environ.get("ES_SYN_REPO", Path.home() / "temporal-tabular-benchmarks"))
FITS = Path(os.environ.get("ES_SYN_FITS", Path.home() / ".sdb-runs/es_time_syn/fits"))
OUT = EXT / "results/es_time/syn"
PLAN = "early_stopping_time_synthetic_plan.md"
SCRIPTS = ("scripts/es_time_syn.py", "scripts/es_time_syn_analyse.py")
PINS = {"scripts/es_time.py": "4cf0cdf382c3875eb51c55d8e667b9f9b59bcc719581beafce58c772327080e9",
        "scripts/es_time_prep.py": "4dd30a122f79b7989d2b377b3707f23d57ba3b47eb3e0e5768f842401fbd03e7"}


def cells():
    out = {}
    for sig in AUC:
        for pi in (0.1, 0.3):
            out[f"A_{sig}_p{round(pi * 100)}"] = dict(family="A", sig=sig, pi=pi)
    for kind in ("entity", "row"):
        for tau in (0.75, 1.5):
            out[f"H_{kind}_t{round(tau * 100)}"] = dict(family="H", kind=kind, tau=tau, chain=500)
    out["H_slow_t150"] = dict(family="H", kind="entity", tau=1.5, chain=4000)
    for rho in (5800, 7150, 8450):
        for sig in ("weak", "medium"):
            out[f"E_r{rho}_{sig}"] = dict(family="E", rho=rho, sig=sig)
    for k, start in enumerate((5640, 6973, 8307), 1):  # the EF stopping set of final cut k
        out[f"B_c{k}"] = dict(family="B", blip=(start, start + 360), sig="medium")
    out["C"] = dict(family="C")
    for kind in ("raw", "norm"):
        for g in (8, 32):
            out[f"U_{kind}_g{g}"] = dict(family="U", kind=kind, g=g, gamma=np.log(g))
    for k in (3, 8):
        out[f"X_k{k}"] = dict(family="X", kappa=k)
    return out


CELLS = cells()
FAMILY_CODE = {"A": 1, "H": 2, "E": 3, "C": 4, "U": 5, "X": 6, "B": 7}


def reference_cell(cell):
    """The no-drift reference of a drift cell: family A with the same signal at pi 0.3; H: its row control;
    U raw: the normalised cell at the same gamma."""
    c = CELLS[cell]
    if c["family"] == "A":
        return None
    if c["family"] == "H":
        return f"H_row_t{round(c['tau'] * 100)}"
    if c["family"] == "U" and c["kind"] == "raw":
        return f"U_norm_g{c['g']}"
    return f"A_{c.get('sig', 'medium')}_p30"


# ---------------------------------------------------------------- latent score and calibration

@lru_cache(None)
def streams():
    ref, e, e_new, u = np.random.SeedSequence(ROOT_SEED).spawn(4)
    return dict(ref=ref, e=np.random.default_rng(e).normal(0, 0.5, 8), e_new=np.random.default_rng(e_new).normal(0, 0.5, 8),
                u=u)


def raw_score(x, c1, new=False):
    s = streams()
    if new:
        return (0.4 * x[:, 0] + 0.6 * x[:, 1] + 0.8 * x[:, 2] + 1.0 * x[:, 3] + 0.7 * x[:, 4] * x[:, 7]
                + 0.8 * (x[:, 5] > 0.5) + s["e_new"][c1])
    return (x[:, 0] + 0.8 * x[:, 1] + 0.6 * x[:, 2] + 0.4 * x[:, 3] + 0.7 * x[:, 4] * x[:, 5]
            + 0.8 * (x[:, 6] > 0.5) + s["e"][c1])


@lru_cache(None)
def reference():
    rng = np.random.default_rng(streams()["ref"])
    x, c1 = rng.standard_normal((REF_ROWS, 8)), rng.integers(0, 8, REF_ROWS)
    f, f_new = raw_score(x, c1), raw_score(x, c1, new=True)
    return dict(x=x, c1=c1, f=f, f_new=f_new, mu=f.mean(), sd=f.std(), mu_new=f_new.mean(), sd_new=f_new.std(),
                fz=(f - f.mean()) / f.std(), corr=float(np.corrcoef(f, f_new)[0, 1]))


def score(x, c1, new=False):
    r = reference()
    return (raw_score(x, c1, new) - r["mu_new"]) / r["sd_new"] if new else (raw_score(x, c1) - r["mu"]) / r["sd"]


def expected_auc(score_, p):
    """ROC AUC of `score_` for labels drawn as Bernoulli(p), in expectation (positives weighted p, negatives 1 - p)."""
    order = np.argsort(score_, kind="stable")
    s, w1, w0 = score_[order], p[order], 1 - p[order]
    below = np.cumsum(w0) - w0
    _, start, counts = np.unique(s, return_index=True, return_counts=True)
    tied0 = np.add.reduceat(w0, start)
    below_tied = np.repeat(below[start], counts) + 0.5 * np.repeat(tied0, counts)
    return float((w1 * below_tied).sum() / (w1.sum() * w0.sum()))


def intercept(s, pi, noise):
    fz = reference()["fz"]
    return brentq(lambda b: expit(b + s * fz + noise).mean() - pi, -20, 20, xtol=1e-10)


@lru_cache(None)
def calibrate(target, pi, tau=0.0):
    """Signal s and intercept b0 with expected ROC AUC of the latent score `target` and prevalence `pi` on the
    reference sample; tau > 0 adds a per-row N(0, tau^2) logit term (family H)."""
    fz = reference()["fz"]
    noise = np.random.default_rng(streams()["u"]).normal(0, tau, REF_ROWS) if tau else 0.0

    def gap(s):
        return expected_auc(fz, expit(intercept(s, pi, noise) + s * fz + noise)) - target
    s = brentq(gap, 0.05, 10, xtol=1e-6)
    return s, intercept(s, pi, noise)


@lru_cache(None)
def prior_path():
    """Family C: intercepts on a grid of prevalences 0.15..0.45 at the medium signal."""
    s, _ = calibrate(AUC["medium"], 0.3)
    grid = np.linspace(0.15, 0.45, 61)
    return s, grid, np.array([intercept(s, q, 0.0) for q in grid])


# ---------------------------------------------------------------- generator

def base(r):
    rng = np.random.default_rng(r)
    return rng.standard_normal((N, 8)), rng.integers(0, 8, (N, 2)), rng.random(N)


def entities(r, chain):
    """Ten slots, each a chain of entities active for `chain` rows, slot k shifted by k * chain / 10 rows."""
    slot = np.random.default_rng([r, FAMILY_CODE["H"]]).integers(0, 10, N)
    j = (np.arange(N) + slot * chain // 10) // chain
    return np.array([f"e{k}_{i}" for k, i in zip(slot, j)])


def usage_norm(usage, group):
    """Usage over the median usage of the same group in the 500 preceding rows (rows 0-499 for the first 500)."""
    out = np.empty(N)
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        lo = np.searchsorted(idx, np.maximum(idx - 500, 0))
        hi = np.searchsorted(idx, np.maximum(idx, 500))
        out[idx] = usage[idx] / np.array([np.median(usage[idx[a:b]]) for a, b in zip(lo, hi)])
    return out


def frame(x, c, cols):
    X = pd.DataFrame(x, columns=cols[:x.shape[1]])
    for name, col in zip(("c1", "c2"), c.T):
        X[name] = pd.Categorical(col, categories=range(8))
    return X


def probabilities(cell, r, t_eff=None):
    """True probabilities of every row with the concept at times `t_eff` (default: each row's own time)."""
    c, (x, cc, _) = CELLS[cell], base(r)
    t = np.arange(N) if t_eff is None else t_eff
    fam = c["family"]
    if fam == "A":
        s, b0 = calibrate(AUC[c["sig"]], c["pi"])
        return expit(b0 + s * score(x, cc[:, 0]))
    if fam == "H":
        s, b0 = calibrate(AUC["medium"], 0.3, c["tau"])
        return expit(b0 + s * score(x, cc[:, 0]) + entity_effect(cell, r))
    if fam in ("E", "B"):
        s, b0 = calibrate(AUC[c["sig"]], 0.3)
        new = t >= c["rho"] if fam == "E" else (t >= c["blip"][0]) & (t < c["blip"][1])
        return expit(b0 + s * np.where(new, score(x, cc[:, 0], new=True), score(x, cc[:, 0])))
    if fam == "C":
        s, grid, b = prior_path()
        return expit(np.interp(0.15 + 0.30 * t / N, grid, b) + s * score(x, cc[:, 0]))
    if fam == "U":
        s, b0 = calibrate(AUC["medium"], 0.3)
        g = np.where(cc[:, 1] < 4, c["gamma"], 0.0)
        usage = np.exp(x[:, 0] + g * np.arange(N) / N)
        x1 = np.log(usage) - g * t / N  # the latent read from usage with the inflation of time t
        return expit(b0 + s * score(np.c_[x1, x[:, 1:]], cc[:, 0]))
    s, b0 = calibrate(AUC["medium"], 0.3)  # X: f of the observed values equals f of the latent ones
    return expit(b0 + s * score(x, cc[:, 0]))


def entity_effect(cell, r):
    c = CELLS[cell]
    rng = np.random.default_rng([r, FAMILY_CODE["H"], 1 if c["kind"] == "entity" else 2])
    if c["kind"] == "row":
        return c["tau"] * rng.standard_normal(N)
    ids = entities(r, c["chain"])
    levels = np.unique(ids)
    return c["tau"] * pd.Series(rng.standard_normal(len(levels)), index=levels)[ids].to_numpy()


def generate(cell, r):
    """Features, label, time and categoricals of one cell and replicate, as `run_unit` takes them."""
    c, (x, cc, uni) = CELLS[cell], base(r)
    t = np.arange(N)
    y = (uni < probabilities(cell, r)).astype(int)
    cols = [f"x{i}" for i in range(1, 9)]
    fam = c["family"]
    if fam == "U":
        g = np.where(cc[:, 1] < 4, c["gamma"], 0.0)
        usage = np.exp(x[:, 0] + g * t / N)
        feat = usage if c["kind"] == "raw" else usage_norm(usage, cc[:, 1])
        X = frame(np.c_[feat, x[:, 1:]], cc, ["usage" if c["kind"] == "raw" else "usage_norm"] + cols[1:])
    elif fam == "X":
        xo = x.copy()
        xo[:, 1] += c["kappa"] * t / N
        xo[:, 2] -= 4 / 3 * c["kappa"] * t / N
        X = frame(xo, cc, cols)
    else:
        X = frame(x, cc, cols)
    cats = ["c1", "c2"]
    if fam == "H":
        ids = entities(r, c["chain"])
        X["entity"] = pd.Categorical(ids, categories=np.unique(ids))
        cats.append("entity")
    return X, y, t, cats


# ---------------------------------------------------------------- checks

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args, repo=EXT):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def assert_pins():
    for f, h in PINS.items():
        assert sha256(SDB / f) == h, f"{f} differs from the pinned file"
    for f in SCRIPTS:
        assert sha256(SDB / f) == sha256(EXT / f), f"{f} in the checkout differs from {EXT / f}"


def assert_committed():
    """The plan and the scripts are committed and pushed in this repository."""
    assert git("status", "--porcelain", "--", PLAN, *SCRIPTS) == "", "plan or scripts not committed"
    assert git("rev-list", "--count", "@{u}..HEAD") == "0", "commits not pushed"


def targets(cell, r, p):
    """Prevalence (mean true probability) and expected ROC AUC of the true probabilities against the cell's targets,
    within 0.02; family C prevalence only, per tenth of the file; families E and B outside the new concept; family H
    for the latent score alone, wider with entity effects (a file holds 209 of them at L 500, 34 at L 4000; the slow
    cell's 0.13 is about 4 sd of a file's mean, 0.08 at L 500 more than 6)."""
    c, fam = CELLS[cell], CELLS[cell]["family"]
    t = np.arange(N)
    if fam == "C":
        for k in range(10):
            m = (t >= k * N // 10) & (t < (k + 1) * N // 10)
            assert abs(p[m].mean() - (0.15 + 0.30 * (k + 0.5) / 10)) < 0.03, f"{cell} r{r} prevalence tenth {k}"
        return
    tol_p, tol_a = 0.02, 0.02
    if fam == "H" and c["kind"] == "entity":
        tol_p, tol_a = (0.13, 0.07) if c["chain"] == 4000 else (0.08, 0.05)
    assert abs(p.mean() - c.get("pi", 0.3)) < tol_p, f"{cell} r{r} prevalence {p.mean():.3f}"
    target = AUC[c.get("sig", "medium")]
    x, cc, _ = base(r)
    m = (t < c["rho"] if fam == "E" else (t < c["blip"][0]) | (t >= c["blip"][1]) if fam == "B"
         else np.ones(N, bool))
    auc = expected_auc(score(x, cc[:, 0]), p) if fam == "H" else expected_auc(p[m], p[m])
    assert abs(auc - target) < tol_a, f"{cell} r{r} AUC {auc:.3f} against {target}"


AFTER = {"E_r8450_weak": ("A_weak_p30", 8450), "E_r8450_medium": ("A_medium_p30", 8450),
         "B_c3": ("A_medium_p30", 8307)}  # cut 1 of these cells sees no new-concept row: same data and fits


def assert_positions():
    """Each blip is exactly the EF stopping set of its final cut; each 'after' cell changes after cut 1's test block."""
    g, lay = es.layout(np.arange(N))
    for k in range(3):
        (a, b), (ta, tb) = lay[("e2", k)]["final"]
        assert CELLS[f"B_c{k + 1}"]["blip"] == (g["stop_start"](a, b), b), f"B_c{k + 1}"
        if k == 0:
            assert all(row >= tb for _, row in AFTER.values()), "an 'after' cell changes before cut 1's test block ends"


def class_counts(y, t):
    g, lay = es.layout(t)
    for cut in (lay[("e2", k)] for k in range(3)):
        for (a, b), (va, vb) in cut["inner"] + [cut["final"]]:
            s = g["stop_start"](a, b)
            assert 5 <= y[s:b].sum() < b - s and 0 < y[a:b].sum() < b - a and 0 < y[va:vb].sum() < vb - va


def check():
    assert_pins()
    assert_positions()
    r0 = reference()
    print(f"reference: corr(f, f_new) {r0['corr']:.3f}")
    for name, c in CELLS.items():
        if c["family"] == "H":
            s, b0 = calibrate(AUC["medium"], 0.3, c["tau"])
        elif c["family"] == "C":
            s, b0 = prior_path()[0], np.nan
        else:
            s, b0 = calibrate(AUC[c.get("sig", "medium")], c.get("pi", 0.3))
        print(f"{name}: s {s:.3f}, b0 {b0:.3f}")
    for r in REPS:
        data = {name: generate(name, r) for name in CELLS}
        for name, (X, y, t, cats) in data.items():
            targets(name, r, probabilities(name, r))
            class_counts(y, t)
        for name, (ref, row) in AFTER.items():
            assert data[name][0].equals(data[ref][0]) and (data[name][1][:row] == data[ref][1][:row]).all(), name
    print(f"data asserts pass on {len(CELLS)} cells x {len(REPS)} replicates, 'after' data identical")
    units = {}
    for name in CELLS:
        df = es.run_unit("syn", "syn", 999, generate(name, 999), n_cand=4)
        assert (df.error == "").all() and df.ap.notna().all(), f"failed fit in {name}"
        units[name] = df
        print(f"reduced unit {name}: {len(df)} fits, no errors")
    cols = ["stage", "block", "arm", "cand", "ap", "trees"]
    for name, (ref, _) in AFTER.items():
        a, b = (units[n].query("cut == 0")[cols].reset_index(drop=True) for n in (name, ref))
        assert a.equals(b), f"{name} cut 1 fits differ from {ref}"
    print("reduced units: 'after' fits identical to their reference on cut 1")


# ---------------------------------------------------------------- run

def run_log():
    OUT.mkdir(parents=True, exist_ok=True)
    entry = {"time": datetime.now(timezone.utc).isoformat(), "threads": es.THREADS,
             "commit": git("rev-parse", "HEAD"), "sdb_commit": git("rev-parse", "HEAD", repo=SDB),
             "versions": {m.__name__: m.__version__ for m in (np, pd, lightgbm, scipy, sklearn, optuna, joblib)},
             "python": sys.version, "environment": {k: os.environ.get(k) for k in ("ES_SYN_REPO", "ES_SYN_FITS")},
             "sha256": {f: sha256(EXT / f) for f in (PLAN, *SCRIPTS)} | {f: sha256(SDB / f) for f in PINS}}
    with open(OUT / "run_log_syn.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def run(only):
    assert_pins()
    assert_committed()
    FITS.mkdir(parents=True, exist_ok=True)
    run_log()
    names = only or list(CELLS)
    units = [(r, name) for r in REPS for name in names]
    for i, (r, name) in enumerate(units, 1):
        path = FITS / f"{name}_r{r}.joblib"
        if path.exists():
            continue
        start = time.time()
        df = es.run_unit("syn", "syn", r, generate(name, r)).assign(dataset=name, syn_sha=sha256(EXT / SCRIPTS[0]))
        errors = int((df.error != "").sum())
        assert errors == 0, f"{name} r{r}: {errors} failed fits; unit not saved"
        joblib.dump(df, f"{path}.tmp")
        os.replace(f"{path}.tmp", path)
        print(f"[{i}/{len(units)}] {name} r{r}: wall {time.time() - start:.0f} s, thread s {df.sec.sum():.0f}, "
              f"fits {len(df)}", flush=True)
    print("Done.")


# ---------------------------------------------------------------- diagnostics

def diagnostics():
    rows = []
    g = geometry(np.arange(N))
    finals = [(cut[3], test) for cut, test in zip(g["e2"], g["test"])]
    for name, c in CELLS.items():
        for r in REPS:
            X, y, t, cats = generate(name, r)
            p = probabilities(name, r)
            row = dict(cell=name, rep=r, prevalence=y.mean())
            for k, ((a, b), (ta, tb)) in enumerate(finals, 1):
                s = g["stop_start"](a, b)
                row |= {f"prev_test_c{k}": y[ta:tb].mean(), f"prev_stop_c{k}": y[s:b].mean(),
                        f"stop_gap_c{k}": y[s:b].mean() - y[a:s].mean(),
                        f"auc_true_c{k}": expected_auc(p[ta:tb], p[ta:tb]),
                        f"low_signal_auc_c{k}": fit_auc(X, y, np.arange(a, b), np.arange(ta, tb))}
                if not (c["family"] == "U" and c["kind"] == "norm"):  # the learner of a normalised cell is not stale
                    stale = probabilities(name, r, np.full(N, (a + b) // 2))
                    row |= {f"stale_ap_c{k}": average_precision_score(y[ta:tb], p[ta:tb])
                            - average_precision_score(y[ta:tb], stale[ta:tb]),
                            f"stale_ll_c{k}": log_loss(y[ta:tb], stale[ta:tb]) - log_loss(y[ta:tb], p[ta:tb]),
                            f"stale_stop_ll_c{k}": log_loss(y[s:b], stale[s:b]) - log_loss(y[s:b], p[s:b])}
                if c["family"] == "H":
                    x, cc, _ = base(r)
                    ids = X.entity.to_numpy()
                    row |= {f"auc_f_c{k}": expected_auc(score(x[ta:tb], cc[ta:tb, 0]), p[ta:tb]),
                            f"known_test_c{k}": float(np.isin(ids[ta:tb], ids[a:b]).mean()),
                            f"known_stop_c{k}": float(np.isin(ids[s:b], ids[a:s]).mean())}
                if c["family"] == "X":
                    row[f"x2_beyond_p99_c{k}"] = float((X.x2.iloc[ta:tb] > np.quantile(X.x2.iloc[a:b], 0.99)).mean())
                    row[f"x3_below_p01_c{k}"] = float((X.x3.iloc[ta:tb] < np.quantile(X.x3.iloc[a:b], 0.01)).mean())
            rows.append(row)
        X, y, t, cats = generate(name, REPS[0])
        drift = screen(X, y, g).set_index("feature").drift_auc
        rows.append(dict(cell=name, rep=-1, **{f"drift_auc_{f}": v for f, v in drift.items()}))
        print(f"diagnostics {name}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(6).to_csv(OUT / "diagnostics_syn.csv", index=False)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "run":
        run(sys.argv[2:])
    else:
        {"check": check, "diag": diagnostics}.get(command, lambda: print(__doc__))()

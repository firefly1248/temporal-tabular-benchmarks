"""es_time_ext_analyse: delta step and analysis of the extension. Plan: early_stopping_time_extension_plan.md.

Holds its own copies of the registered functions it needs (interval, pooling, A/A splits, labels), with the level, the
datasets and the paths as arguments. It imports nothing from es_time_analyse, which binds a 98.75 % level as a default
argument and whose delta() overwrites the first run's delta.json.

delta     T's final E2 fits of the 23 datasets -> results/es_time/ext/delta_ext.json (run once, first)
analyse   primary, secondary, sensitivity, descriptive and diagnostic outputs; needs delta_ext.json, never recomputes it
manifest  sha256 of the outputs -> outputs_manifest.json
check     every number of a facts sheet against the cited line of an output (default: results/es_time/ext/facts_ext.md)
selftest  the analysis on synthetic fits with planted effects, in a temporary folder; reads no results
reproduce the seven published datasets at the first look's level: their A/A delta, per-dataset points and pooled rows

f1-dnf (Ergast terms, non-commercial use only) was dropped after the run; every step reads the other 23 datasets (plan,
Deviations, 2026-11-23). Its fits are hash-checked with the other published fits but never read.

Usage: python -m scripts.es_time_ext_analyse delta | analyse | manifest | check [sheet] | selftest | reproduce
"""
import json
import re
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

import scripts.es_time_ext as ext
from scripts.es_time import CAPPED, SEEDS, layout

LEVEL, TIE = 0.99375, 1e-4
HYP = {"H1": ("d_t", "dr_t"), "H2": ("d", "ef"), "H3": ("t", "d_own"), "H4": ("t", "er")}
DESCRIPTIVE = {"D - DR": ("d", "dr"), "T - EF": ("t", "ef"), "D - D_own": ("d", "d_own"), "DR - ER": ("dr", "er"),
               "D_T9 - D_T": ("d_t9", "d_t")}
PAIRS = [("d_own_es", "ef"), ("er", "ef"), ("t", "ef"), ("t", "er")]
KEY = ["pop", "scheme", "dataset", "seed", "cut"]
DROPPED = ["f1-dnf"]
PUB = [n for n in ext.PUBLISHED if n not in DROPPED]  # the seven published datasets kept
ALL = PUB + ext.NEW
NYC = ["tlc-yellow", "citibike", "nyc-311"]
ARMS = ("t", "er", "ef", "d", "dr", "d_own_es", "d_own", "er_t", "ef_t", "dr_t", "d_t", "d_t9")
FILES = ["analysis_ext.txt", "per_dataset_ext.csv", "stopping_set_gaps_ext.csv", "delta_ext.json", "forest.csv",
         "per_dataset_points.csv"]
ROWS_PER_UNIT = 2 * 3 * 732  # thin: E2 and E1, three cuts, 732 fits per scheme and cut
# The plan's subset lists (Population, check 3 and the low-signal rule); the summary files must reproduce them.
LOW = {"chicago-food", "nypd-shooting", "stats19", "nyc-collisions", "tlc-yellow", "citibike", "dota2", "nycflights",
       "bank-full"}
LAG = {"nycflights", "nypd-shooting", "dota2", "nyc-311"}
# Datasets whose rare levels are recoded to "other" (plan, Unchanged from the first plan).
RECODED = ["austin", "kickstarter", "chicago-food", "seattle-stops", "london-fire", "dvsa-mot", "nyc-collisions", "citibike",
           "lichess", "dota2", "nyc-311"]
DRIFT = {"stats19": "special_conditions_at_site", "nyc-collisions": "VEHICLE TYPE CODE 1"}


def multiplier(n, level):
    return t_dist.ppf(1 - (1 - level) / 2, n - 1)


def half_width(values, level):
    return multiplier(len(values), level) * np.std(values, ddof=1) / np.sqrt(len(values))


# ---------------------------------------------------------------- delta

def split_halves(s, level):
    """Half-width of the pooled interval of T's seed-pair contrast, for the 15 splits of five seeds into two disjoint
    pairs. s: dataset x seed frame of T's mean test score over cuts."""
    out = []
    for left_out in SEEDS:
        rest = [k for k in SEEDS if k != left_out]
        for j in (1, 2, 3):
            a = [rest[0], rest[j]]
            b = [k for k in rest if k not in a]
            out.append(half_width((s[a].mean(axis=1) - s[b].mean(axis=1)).to_numpy(), level))
    return out


def compute_delta(fits, level):
    t = fits[(fits.arm == "t") & (fits.stage == "final") & (fits.scheme == "e2")]
    assert set(t.dataset) == set(ALL) and t.ap.notna().all()
    assert len(t) == len(ALL) * len(SEEDS) * 3 and (t.groupby("dataset").size() == len(SEEDS) * 3).all(), "T rows incomplete"
    halves = split_halves(t.groupby(["dataset", "seed"]).ap.mean().unstack("seed").loc[ALL], level)
    return {"delta": float(np.sqrt(np.mean(np.square(halves)))), "half_widths": halves, "level": level,
            "n_datasets": len(ALL), "multiplier": float(multiplier(len(ALL), level)),
            "rows_read": f"arm t, stage final, scheme e2, {len(t)} rows"}


FIRST_LEVEL = 0.9875  # the first look's level


def published_delta(fits):
    """The first look's A/A procedure on the seven published datasets kept (0.00866), at the first look's level."""
    t = fits[(fits.arm == "t") & (fits.stage == "final") & (fits.scheme == "e2")]
    s = t.groupby(["dataset", "seed"]).ap.mean().unstack("seed").loc[PUB]
    return float(np.sqrt(np.mean(np.square(split_halves(s, FIRST_LEVEL)))))


# ---------------------------------------------------------------- labels and pooling

def label(lo, hi, delta_):
    direction = "first fit ahead" if lo > 0 else "second fit ahead" if hi < 0 else "no difference found"
    if lo > delta_ or hi < -delta_:
        return direction + ", beyond delta"
    if lo > -delta_ and hi < delta_:
        return direction + ", equivalent within delta"
    return direction


def flagged(per, delta_):
    """Datasets whose removal flips the sign of the pooled mean or moves it across +delta or -delta."""
    full = per.mean()
    out = []
    for d in per.index:
        loo = per.drop(d).mean()
        if np.sign(full) != np.sign(loo) or (full > delta_) != (loo > delta_) or (full < -delta_) != (loo < -delta_):
            out.append(d)
    return out


def pooled(per, level):
    """Mean over datasets with its t-interval; NaN bounds for fewer than two datasets."""
    if len(per) < 2:
        return per.mean(), np.nan, np.nan
    h = half_width(per.to_numpy(), level)
    return per.mean(), per.mean() - h, per.mean() + h


def fmt(per, level):
    m, lo, hi = pooled(per, level)
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] (n {len(per)}, {len(per) - 1} df)"


def welch(a, b, level):
    """Difference of the means of two groups of datasets with a Welch t-interval."""
    va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    m, h = np.mean(a) - np.mean(b), t_dist.ppf(1 - (1 - level) / 2, df) * np.sqrt(va + vb)
    return m, m - h, m + h, df


def one_unit(per, names, name):
    """`names` averaged into one unit called `name`."""
    return pd.concat([per.drop(names), pd.Series({name: per.loc[names].mean()})])


# ---------------------------------------------------------------- analysis

def cells(fits):
    """One row per population, scheme, dataset, seed and cut; columns (value, arm) of the final fits."""
    f = fits[fits.stage == "final"]
    return f.set_index(KEY + ["arm"])[["ap", "trees", "best", "cand"]].unstack("arm")


def diff(w, a, b):
    return w["ap"][a] - w["ap"][b]


def per_dataset(d, scheme):
    per = d.xs(("thin", scheme), level=("pop", "scheme")).groupby("dataset").mean().loc[ALL]
    assert per.notna().all(), "a dataset has no value"
    return per


def check_complete(fits, want):
    per_unit = fits.groupby(["dataset", "seed"]).size()
    short = [(k, n) for k, n in per_unit.items() if n != want]
    failed = fits[(fits.error != "") | fits.ap.isna()]
    lines = [f"Before analysis: {len(per_unit)} units (expected {len(ALL) * len(SEEDS)}); units with a wrong row count: "
             f"{short or 'none'}; failed fits or NaN scores: {len(failed)}"]
    for r in failed.head(20).itertuples():
        lines.append(f"  {r.dataset} seed {r.seed} {r.scheme} cut {r.cut} {r.stage} {r.arm} cand {r.cand}: {r.error}")
    assert not short and len(per_unit) == len(ALL) * len(SEEDS) and set(fits.dataset) == set(ALL), "incomplete run"
    assert failed.empty, "failed fits or NaN scores"
    return lines


def report(fits, delta_new, delta_pub, summary, level, drift=None):
    w = cells(fits)
    new, low = ext.NEW, list(summary.index[summary.low_signal])
    lag = list(summary.index[summary.c3_ef_max > 0.05])
    assert set(low) == LOW and set(lag) == LAG, "the summary files do not give the plan's subsets"
    final = fits[(fits.stage == "final") & (fits.scheme == "e2")]
    share = {a: final[final.arm == a].groupby("dataset").best.apply(lambda b: (b > CAPPED).mean()) for a in ("er_t", "ef_t")}
    capped = list(share["er_t"].index[np.maximum(share["er_t"], share["ef_t"]) > 0.25])
    equal = (w["trees"]["d_t"] == w["trees"]["dr_t"]).astype(float)
    L = []
    p = L.append
    p(f"Primary: thin, E2 (differences of test average precision, mean over 3 cuts and 5 seeds per dataset; pooled over "
      f"{len(ALL)} datasets; t-interval at {level:.3%}, {len(ALL) - 1} df, multiplier {multiplier(len(ALL), level):.4f}; "
      f"delta {delta_new:.5f}, delta of the seven published datasets {delta_pub:.5f})")
    for h, (a, b) in HYP.items():
        d = diff(w, a, b)
        per = per_dataset(d, "e2")
        m, lo, hi = pooled(per, level)
        ahead, behind = int((per >= TIE).sum()), int((per <= -TIE).sum())
        bad = flagged(per, delta_new)
        p(f" {h} {a.upper()} - {b.upper()}: {fmt(per, level)}, half-width {(hi - lo) / 2:.4f}")
        p(f"    label: {label(lo, hi, delta_new)} (delta {delta_new:.5f}); ahead {ahead} / behind {behind} / tied "
          f"{len(per) - ahead - behind}; leave-one-out: {'not robust, flagged by ' + ', '.join(bad) if bad else 'robust'}; "
          f"cells without a value: {int(d.xs(('thin', 'e2'), level=('pop', 'scheme')).isna().sum())}")
        p(f"    companion label under the delta of the seven published datasets {delta_pub:.5f} (no verdict): {label(lo, hi, delta_pub)}")
        p(f"    without low-signal datasets ({', '.join(low) or 'none'}): {fmt(per.drop(low), level)}")
        if h == "H1":
            p(f"    without datasets above 5 % EF stopping-set label lag ({', '.join(lag) or 'none'}): "
              f"{fmt(per.drop(lag), level)}")
            p(f"    without datasets with over 25 % capped ER_T or EF_T final fits ({', '.join(capped) or 'none'}): "
              f"{fmt(per.drop(capped), level)}")
            p("    cells with equal tree counts for D_T and DR_T, per dataset: "
              + ", ".join(f"{k} {v:.2f}" for k, v in per_dataset(equal, "e2").items()))
            if drift is not None:
                assert len(drift) == 6, "drift rows"
                p("    columns that change coding, drift AUC and H1 per cut:")
                for r in drift.itertuples():
                    cut = d.xs(("thin", "e2", r.dataset, r.cut - 1), level=("pop", "scheme", "dataset", "cut")).mean()
                    p(f"      {r.dataset} {r.feature} cut {r.cut}: drift AUC {r.drift_auc:.3f}, H1 {cut:+.4f}")
        for k in range(3):
            pc = d.xs(("thin", "e2", k), level=("pop", "scheme", "cut")).groupby("dataset").mean().loc[ALL]
            p(f"    cut {k + 1}: {fmt(pc, level)}")
    p("\nSecondary, no labels (same level)")
    for h, (a, b) in HYP.items():
        per = per_dataset(diff(w, a, b), "e2")
        mw, lw, hw, dfw = welch(per.loc[PUB].to_numpy(), per.loc[new].to_numpy(), level)
        p(f" {h}: the {len(new)} new datasets alone {fmt(per.loc[new], level)}; the seven published datasets minus the new "
          f"ones {mw:+.4f} [{lw:+.4f}, {hw:+.4f}] (Welch, {dfw:.1f} df); New York 2023 datasets as one unit "
          f"{fmt(one_unit(per, NYC, 'new-york-2023'), level)}")
    p("\nSensitivity, no labels: E1 (expanding window)")
    for h, (a, b) in HYP.items():
        p(f"  {h}: {fmt(per_dataset(diff(w, a, b), 'e1'), level)}")
    p("\nExploratory: pooled without nyc-311")
    for h, (a, b) in HYP.items():
        p(f"  {h}: {fmt(per_dataset(diff(w, a, b), 'e2').drop('nyc-311'), level)}")
    p("\nDescriptive differences, thin E2 (no label):")
    for name, (a, b) in DESCRIPTIVE.items():
        p(f"  {name}: {fmt(per_dataset(diff(w, a, b), 'e2'), level)}")
    ratio = (w["best"]["er_t"] / w["best"]["ef_t"]).xs(("thin", "e2"), level=("pop", "scheme")).groupby("dataset").median()
    p("  median ER_T / EF_T best iteration per dataset: " + ", ".join(f"{k} {v:.2f}" for k, v in ratio.items()))
    p("  share of cells where two arms pick the same candidate (mean over datasets):")
    for a, b in PAIRS:
        p(f"    {a} = {b}: {per_dataset((w['cand'][a] == w['cand'][b]).astype(float), 'e2').mean():.2f}")
    p("\nDiagnostics, thin E2 (the last block: new datasets, E2 and E1):")
    f = fits[fits.scheme == "e2"]
    p(" median trees per arm and cut (inner, final):")
    L += f.groupby(["stage", "arm", "cut"]).trees.median().unstack("cut").round(0).to_string().split("\n")
    es = f[f.arm.isin(["er", "ef", "er_t", "ef_t", "d_own_es"])]
    p(f" capped share of ES fits (best iteration above {CAPPED}):")
    L += es.groupby(["stage", "arm"]).best.apply(lambda b: (b > CAPPED).mean()).round(3).to_string().split("\n")
    fin = f[(f.stage == "final") & f.arm.isin(["er", "ef"])].assign(ll_gap=lambda x: x.tgt_ll - x.stop_ll,
                                                                   ap_gap=lambda x: x.ap - x.stop_ap)
    p(" stopping-set optimism, final ER and EF fits (test minus stopping set), unseen-level shares, best iterations:")
    cols = ["ll_gap", "ap_gap", "unseen_stop", "unseen_tgt", "best", "best_ap_iter"]
    agg = lambda x: x.mean() if x.name not in ("best", "best_ap_iter") else x.median()
    L += fin.groupby("arm")[cols].agg(agg).round(4).to_string().split("\n")
    p(" the same, for the published datasets (never recoded), the new datasets with rare levels recoded and the other new ones:")
    group = np.where(fin.dataset.isin(PUB), "published", np.where(fin.dataset.isin(RECODED), "new, recoded", "new, not recoded"))
    L += fin.assign(group=group).groupby(["group", "arm"])[cols].agg(agg).round(4).to_string().split("\n")
    p(" unseen-level shares per dataset (final ER and EF fits, mean): stopping set, test block")
    L += fin.groupby(["dataset", "arm"])[["unseen_stop", "unseen_tgt"]].mean().round(4).to_string().split("\n")
    p(" thread seconds per arm and stage, new datasets, E2 and E1:")
    L += fits[fits.dataset.isin(new)].groupby(["stage", "arm"]).sec.sum().round(0).to_string().split("\n")
    return L


def stopping_set_gaps(w, summary, loader):
    """H1 and H2 per dataset and cut with the minority-share gaps and persistence they are plotted against."""
    rows = []
    for name in ALL:
        _, y, t, _ = loader(name)
        g, lay = layout(t)
        for k in range(3):
            (a, b), (ta, tb) = lay[("e2", k)]["final"]
            s = g["stop_start"](a, b)
            cell = w.xs(("thin", "e2", name, k), level=("pop", "scheme", "dataset", "cut"))
            rows.append(dict(dataset=name, cut=k + 1, h1=(cell["ap"]["d_t"] - cell["ap"]["dr_t"]).mean(),
                             h2=(cell["ap"]["d"] - cell["ap"]["ef"]).mean(),
                             gap_test=y[s:b].mean() - y[ta:tb].mean(), gap_older=y[s:b].mean() - y[a:s].mean(),
                             persistence=summary.persistence[name]))
    return pd.DataFrame(rows)


def per_dataset_table(w):
    """Per dataset: the four differences per cut and seed, mean and 95 % t-interval over the five seeds (descriptive)."""
    rows = []
    for h, (a, b) in HYP.items():
        d = diff(w, a, b).unstack("seed")
        for (pop, scheme, name, cut), v in d.iterrows():
            rows.append((pop, scheme, name, h, cut + 1) + tuple(v))
    out = pd.DataFrame(rows, columns=["pop", "scheme", "dataset", "hyp", "cut"] + [f"seed{s}" for s in SEEDS])
    seeds = [f"seed{s}" for s in SEEDS]
    allcuts = out.groupby(["pop", "scheme", "dataset", "hyp"])[seeds].mean().reset_index().assign(cut=0)
    out = pd.concat([out, allcuts], ignore_index=True)
    v = out[seeds].to_numpy()
    out["mean"] = np.nanmean(v, axis=1)
    out["lo95"] = out["mean"] - t_dist.ppf(0.975, len(SEEDS) - 1) * np.nanstd(v, axis=1, ddof=1) / np.sqrt(len(SEEDS))
    out["hi95"] = 2 * out["mean"] - out["lo95"]
    return out


def forest(w, summary, delta_new, delta_pub, level):
    low = list(summary.index[summary.low_signal])
    blocks = {"23 datasets": ALL, "16 new datasets": ext.NEW, "without the low-signal datasets": [d for d in ALL if d not in low]}
    rows, points = [], []
    for block, names in blocks.items():
        for h, (a, b) in HYP.items():
            per = per_dataset(diff(w, a, b), "e2").loc[names]
            m, lo, hi = pooled(per, level)
            rows.append(dict(block=block, hypothesis=h, first=a, second=b, estimate=m, lower=lo, upper=hi,
                             label=label(lo, hi, delta_new) if block == "23 datasets" else "", delta=delta_new,
                             delta_published_seven=delta_pub,
                             level=level, n=len(names)))
            points += [dict(block=block, hypothesis=h, dataset=d, value=v) for d, v in per.items()]
    return pd.DataFrame(rows), pd.DataFrame(points)


def write_outputs(fits, summary, delta_meta, delta_pub, loader, out, want, drift=None):
    """All analysis outputs into `out`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    level = delta_meta["level"]
    assert level == LEVEL and abs(multiplier(len(ALL), level) - 3.0231) < 5e-5, "level or multiplier is not the plan's"
    lines = check_complete(fits, want) + report(fits, delta_meta["delta"], delta_pub, summary, level, drift)
    (out / "analysis_ext.txt").write_text("\n".join(lines) + "\n")
    w = cells(fits)
    per_dataset_table(w).to_csv(out / "per_dataset_ext.csv", index=False, float_format="%.6f")
    stopping_set_gaps(w, summary, loader).to_csv(out / "stopping_set_gaps_ext.csv", index=False, float_format="%.6f")
    f, pts = forest(w, summary, delta_meta["delta"], delta_pub, level)
    f.to_csv(out / "forest.csv", index=False, float_format="%.6f")
    pts.to_csv(out / "per_dataset_points.csv", index=False, float_format="%.6f")
    return lines


# ---------------------------------------------------------------- steps on the stored fits

def stored():
    """The stored fits: the seven published datasets kept (the hashes of all eight are checked against the delta.json of
    the first look) and the new ones."""
    inputs = json.load(open(ext.FIRST / "delta.json"))["inputs"]
    pub = ext.stored_fit_files()
    assert {n: ext.sha256(p) for n, p in pub.items()} == {n: inputs[n] for n in pub}, "published fits changed"
    pub = {n: p for n, p in pub.items() if n.rsplit("_seed", 1)[0] in PUB}
    want = [f"{n}_seed{s}.joblib" for n in ext.NEW for s in SEEDS]
    new = {f: ext.FITS / "thin" / f for f in want}
    assert all(p.exists() for p in new.values()), "the new fits are incomplete"
    return pub, new


def read(files):
    return pd.concat([joblib.load(f) for f in files.values()], ignore_index=True)


def summary_frame():
    first = pd.read_csv(ext.FIRST / "prepared/_summary_thin.csv").set_index("dataset").loc[PUB]
    new = pd.read_csv(ext.PREPARED / "_summary_thin_ext.csv").set_index("dataset").loc[ext.NEW]
    cols = ["low_signal", "c3_ef_max", "persistence"]
    return pd.concat([first[cols], new[cols]]).loc[ALL]


def drift_frame():
    rows = [pd.read_csv(ext.PREPARED / f"_features_allcuts_{n}.csv").query("feature == @c")[["dataset", "feature", "cut", "drift_auc"]]
            for n, c in DRIFT.items()]
    return pd.concat(rows, ignore_index=True)


def precheck():
    """The summary files give the plan's subsets and the drift file the six rows; nothing is fitted."""
    s = summary_frame()
    assert set(s.index[s.low_signal]) == LOW and set(s.index[s.c3_ef_max > 0.05]) == LAG, "subsets differ from the plan's"
    assert len(drift_frame()) == 6, "drift rows"


def delta_step():
    ext.guard_dev()
    ext.assert_pins()
    ext.assert_one_run()
    assert not (ext.OUT / "delta_ext.json").exists(), "delta_ext.json exists: the delta step runs once"
    pub, new = stored()
    fits = read({**pub, **new})
    meta = compute_delta(fits, LEVEL)
    meta["delta_published_seven"] = published_delta(fits)
    meta["inputs"] = {n: ext.sha256(p) for n, p in new.items()}
    meta["published_inputs"] = {n: ext.sha256(p) for n, p in pub.items()}
    ext.OUT.mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(ext.OUT / "delta_ext.json.tmp", "w"), indent=1)
    (ext.OUT / "delta_ext.json.tmp").replace(ext.OUT / "delta_ext.json")
    h = meta["half_widths"]
    print(f"delta {meta['delta']:.5f} at level {LEVEL} from {len(h)} splits (half-widths {min(h):.5f} to {max(h):.5f})")


def analyse_step():
    ext.guard_dev()
    ext.assert_pins()
    ext.assert_one_run()
    meta = json.load(open(ext.OUT / "delta_ext.json"))
    pub, new = stored()
    assert {n: ext.sha256(p) for n, p in new.items()} == meta["inputs"], "new results changed since delta was computed"
    assert {n: ext.sha256(p) for n, p in pub.items()} == meta["published_inputs"], "published results changed"
    delta_pub = meta["delta_published_seven"]
    lines = write_outputs(read({**pub, **new}), summary_frame(), meta, delta_pub, ext.load, ext.OUT, ROWS_PER_UNIT,
                          drift_frame())
    print("\n".join(lines))
    check_manifest()


def check_manifest():
    """With a manifest from an earlier analysis run, the outputs of this run must be byte-identical to it."""
    path = ext.OUT / "outputs_manifest.json"
    if path.exists():
        assert json.load(open(path)) == {n: ext.sha256(ext.OUT / n) for n in FILES}, "this run is not byte-identical to the last"
        print("outputs byte-identical to the manifest")


def manifest_step():
    """Writes the output hashes of the first analysis run; the second analysis run then compares itself with them."""
    ext.guard_dev()
    path = ext.OUT / "outputs_manifest.json"
    if not path.exists():
        json.dump({n: ext.sha256(ext.OUT / n) for n in FILES}, open(path, "w"), indent=1, sort_keys=True)
    check_manifest()
    print(open(path).read())


def numbers_in(line):
    return [x.replace(",", "").rstrip("%") for x in
            re.findall(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?%?(?![\w.])", line)]


def has_number(token, line):
    """The token is one of the numbers of the line, sign included (not part of a longer number or a name like H1)."""
    return token.replace(",", "").rstrip("%") in numbers_in(line)


def check_step(sheet):
    """A row of the sheet is `| id | statement | numbers separated by ; | source |`; the source is `file:line` of an
    output or `calc:total_hours`. Every number must occur in the cited line."""
    text = {n: (ext.OUT / n).read_text().splitlines() for n in FILES}
    bad, n_numbers, n_rows = [], 0, 0
    for row in Path(sheet).read_text().splitlines():
        cells_ = [c.strip() for c in row.strip().strip("|").split("|")]
        if not row.startswith("|") or cells_[0] in ("id", "") or set(cells_[0]) <= set("-: "):
            continue
        if len(cells_) != 4:
            bad.append(("malformed row", row[:80]))
            continue
        fid, _, numbers, source = cells_
        tokens = [t.strip() for t in numbers.split(";") if t.strip()]
        n_rows += 1
        n_numbers += len(tokens)
        if source == "calc:total_hours":
            _, new = stored()
            have = f"{read(new).sec.sum() / 3600:.2f}"
            if tokens != [have]:
                bad.append((fid, tokens, have))
            continue
        name, _, line = source.partition(":")
        assert tokens and int(line) >= 1, f"{fid}: no numbers or no line"
        line_text = text[name][int(line) - 1]
        bad += [(fid, t, f"{source}: {line_text.strip()[:90]}") for t in tokens if not has_number(t, line_text)]
    print(f"facts sheet: {n_rows} rows, {n_numbers} numbers checked, {len(bad)} not found in the cited line")
    for b in bad:
        print("  MISMATCH", b)
    sys.exit(1 if bad else 0)


# ---------------------------------------------------------------- self test

# E2 offsets of the arms in the synthetic fits: H1 +0.02, H2 +0.015, H3 +0.003, H4 +0.006 before the extras in fake_fits.
OFFSET = {"d_t": 0.02, "d": 0.015, "t": 0.004, "d_own": 0.001, "er": -0.002}
TRIO = ["tlc-yellow", "citibike", "nyc-311"]


def fake_fits(rng):
    """Synthetic fits with the real schema. E2 carries the offsets above; on D_T the new datasets get +0.01, tlc-yellow
    +0.05 and later cuts +0.002 per cut; on D the new datasets get +0.01; austin has H1 exactly 0 (a tie). E1 has a
    different effect (D_T -0.03); capped fits: nyc-311 ER_T in every cell (share 1), lichess ER_T and dvsa-mot EF_T in two
    seeds of five (0.4), moneypuck EF_T in one (0.2, under the 25 % rule); hotels has unequal tree counts for D_T and DR_T."""
    rows = []
    for name in ALL:
        for seed in SEEDS:
            for scheme in ("e2", "e1"):
                for cut in range(3):
                    base = 0.3 + 0.05 * rng.standard_normal()
                    for arm in ARMS:
                        es = arm in ("er", "ef", "er_t", "ef_t", "d_own_es")
                        off = OFFSET.get(arm, 0.0) if scheme == "e2" else (-0.03 if arm == "d_t" else 0.0)
                        if scheme == "e2" and arm == "d_t":
                            off += (0.01 if name in ext.NEW else 0.0) + (0.05 if name == "tlc-yellow" else 0.0) + 0.002 * cut
                        if scheme == "e2" and arm == "d" and name in ext.NEW:
                            off += 0.01
                        noise = 0.0 if name == "austin" and arm in ("d_t", "dr_t") else 0.003 * rng.standard_normal()
                        if name == "austin" and scheme == "e2" and arm == "d_t":
                            off = 0.0
                        capped = es and ((name == "nyc-311" and arm == "er_t") or (name == "lichess" and arm == "er_t" and seed < 2)
                                         or (name == "dvsa-mot" and arm == "ef_t" and seed < 2)
                                         or (name == "moneypuck" and arm == "ef_t" and seed == 0))
                        trees = 120 if (name == "hotels" and arm == "dr_t") else 90 if arm == "d" else 100
                        rows.append(dict(pop="thin", scheme=scheme, dataset=name, seed=seed, cut=cut, stage="final", arm=arm,
                                         block=-1, cand=int(rng.integers(3)), ap=base + noise + off, trees=trees,
                                         best=(490 if capped else 100) if es else np.nan,
                                         best_ap_iter=80 if es else np.nan, stop_ll=0.5, stop_ap=0.4, tgt_ll=0.55,
                                         unseen_stop=0.0, unseen_tgt=0.0, sec=1.0, error=""))
    return pd.DataFrame(rows)


def independent_delta(fits, level):
    """Delta by enumeration of the 15 unordered pairs of disjoint seed pairs, with scipy's interval."""
    from itertools import combinations
    from scipy.stats import ttest_1samp
    s = fits[(fits.arm == "t") & (fits.scheme == "e2")].groupby(["dataset", "seed"]).ap.mean().unstack("seed").loc[ALL]
    halves = []
    for a in combinations(SEEDS, 2):
        for b in combinations([k for k in SEEDS if k not in a], 2):
            if a < b:
                x = (s[list(a)].mean(axis=1) - s[list(b)].mean(axis=1)).to_numpy()
                ci = ttest_1samp(x, 0).confidence_interval(level)
                halves.append((ci.high - ci.low) / 2)
    assert len(halves) == 15
    return float(np.sqrt(np.mean(np.square(halves))))


class Truth:
    """The analysis recomputed from the raw fits with pandas and scipy, sharing no code with the script."""

    def __init__(self, fits, level):
        self.fits, self.level = fits, level

    def per(self, a, b, scheme="e2", cut=None):
        f = self.fits[(self.fits.scheme == scheme) & (self.fits.stage == "final")]
        if cut is not None:
            f = f[f.cut == cut]
        m = f.pivot_table(index="dataset", columns="arm", values="ap", aggfunc="mean")
        return (m[a] - m[b]).loc[ALL]

    def interval(self, v):
        from scipy.stats import ttest_1samp
        ci = ttest_1samp(np.asarray(v, dtype=float), 0).confidence_interval(self.level)
        return float(np.mean(v)), float(ci.low), float(ci.high)

    def fmt(self, v):
        m, lo, hi = self.interval(v)
        return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] (n {len(v)}, {len(v) - 1} df)"

    def welch(self, a, b):
        from scipy.stats import ttest_ind
        r = ttest_ind(a, b, equal_var=False)
        ci = r.confidence_interval(self.level)
        return float(np.mean(a) - np.mean(b)), float(ci.low), float(ci.high), float(r.df)


def selftest():
    from scipy.stats import ttest_1samp
    assert "scripts.es_time_analyse" not in sys.modules, "the registered analysis module must not be imported"
    assert abs(multiplier(23, LEVEL) - 3.0231) < 5e-5 and abs(multiplier(16, LEVEL) - 3.1771) < 5e-5
    assert abs(multiplier(22, LEVEL) - 3.0382) < 5e-5 and abs(half_width(np.array([1.0, -1.0] * 12), LEVEL)
                                                              - 3.0095 * np.std([1.0, -1.0] * 12, ddof=1) / np.sqrt(24)) < 1e-4
    assert label(0.001, 0.003, 0.005) == "first fit ahead, equivalent within delta"
    assert label(-0.002, 0.003, 0.005) == "no difference found, equivalent within delta"
    assert label(0.006, 0.01, 0.005) == "first fit ahead, beyond delta"
    assert label(-0.02, -0.008, 0.005) == "second fit ahead, beyond delta"
    assert label(-0.02, 0.02, 0.005) == "no difference found"
    assert label(-0.001, 0.02, 0.005) == "no difference found"
    per = pd.Series([0.01, 0.011, 0.012, 0.0, 0.2], index=list("abcde"))
    assert flagged(per, 0.01) == ["e"]
    a, b = np.arange(8, dtype=float), np.arange(16, dtype=float) + 1.0
    m, lo, hi, df = welch(a, b, LEVEL)
    assert m < 0 and hi < 0 and abs(m - (a.mean() - b.mean())) < 1e-12 and lo < m < hi and 8 <= df <= 22
    unit = one_unit(pd.Series(np.arange(23.0), index=ALL), NYC, "x")
    assert len(unit) == 21 and abs(unit["x"] - np.mean([ALL.index(n) for n in TRIO])) < 1e-12 and NYC == TRIO
    assert has_number("0.0020", "-0.0020 [-0.0066, +0.0026] (n 24, 23 df)") is False
    assert has_number("-0.0020", "-0.0020 [-0.0066, +0.0026] (n 24, 23 df)") and not has_number("0.01", "0.012")
    assert not has_number("1", "H1 seed0") and not has_number("-5.3", "-5.3e-05") and has_number("99.375", "at 99.375%")
    assert has_number("351,360", "fits 351,360 in all")
    rng = np.random.default_rng(0)
    fits = fake_fits(rng)
    truth = Truth(fits, LEVEL)
    meta = compute_delta(fits, LEVEL)
    assert meta["level"] == LEVEL and len(meta["half_widths"]) == 15 and abs(meta["multiplier"] - 3.0231) < 5e-5
    assert abs(meta["delta"] - independent_delta(fits, LEVEL)) < 1e-9, "delta differs from the independent computation"
    assert meta["rows_read"].endswith(f"{len(ALL) * len(SEEDS) * 3} rows")
    summary = pd.DataFrame({"low_signal": [n in LOW for n in ALL], "c3_ef_max": [0.1 if n in LAG else 0.01 for n in ALL],
                            "persistence": np.arange(len(ALL)) / 10}, index=ALL)
    drift = pd.DataFrame({"dataset": ["stats19"] * 3 + ["nyc-collisions"] * 3,
                          "feature": [DRIFT["stats19"]] * 3 + [DRIFT["nyc-collisions"]] * 3, "cut": [1, 2, 3] * 2,
                          "drift_auc": [0.79, 0.74, 0.70, 0.66, 0.52, 0.58]})
    h1_mean = float(truth.per("d_t", "dr_t").mean())
    delta_pub = 1.02 * h1_mean  # unlike delta, and close enough to H1 that removing austin (a tie) crosses it
    texts = []
    for _ in range(2):
        state = np.random.default_rng(1)

        def loader(name, state=state):
            n = 10_000
            return None, (state.random(n) < 0.3).astype(int), np.arange(n), None

        with tempfile.TemporaryDirectory() as tmp:
            files = {}
            for name, g in fits.groupby("dataset"):  # the fits go through files, as the steps read them
                files[name] = Path(tmp) / f"{name}.joblib"
                joblib.dump(g, files[name])
            write_outputs(read(files), summary, meta, delta_pub, loader, tmp, want=len(ARMS) * 2 * 3, drift=drift)
            texts.append({n: (Path(tmp) / n).read_bytes() for n in FILES if n != "delta_ext.json"})
    assert texts[0] == texts[1], "two runs of the analysis are not byte-identical"
    out = texts[0]["analysis_ext.txt"].decode()
    d, new = meta["delta"], ext.NEW
    names = {"H1": ("D_T", "DR_T", "d_t", "dr_t"), "H2": ("D", "EF", "d", "ef"), "H3": ("T", "D_OWN", "t", "d_own"),
             "H4": ("T", "ER", "t", "er")}
    low = [n for n in ALL if n in LOW]
    lag = [n for n in ALL if n in LAG]
    for h, (A, B, a, b) in names.items():
        v = truth.per(a, b)
        m, lo, hi = truth.interval(v)
        assert f" {h} {A} - {B}: {truth.fmt(v)}, half-width {(hi - lo) / 2:.4f}\n" in out, h
        ahead, behind = int((v >= 1e-4).sum()), int((v <= -1e-4).sum())
        loo = [n for n in ALL if (np.sign(v.mean()) != np.sign(v.drop(n).mean()) or (v.mean() > d) != (v.drop(n).mean() > d)
                                  or (v.mean() < -d) != (v.drop(n).mean() < -d))]
        assert (f"    label: {label(lo, hi, d)} (delta {d:.5f}); ahead {ahead} / behind {behind} / tied "
                f"{len(v) - ahead - behind}; leave-one-out: {'not robust, flagged by ' + ', '.join(loo) if loo else 'robust'}; ") in out, h
        assert f"(no verdict): {label(lo, hi, delta_pub)}\n" in out, h
        assert f"without low-signal datasets ({', '.join(low)}): {truth.fmt(v.drop(low))}\n" in out, h
        for k in range(3):
            assert f"    cut {k + 1}: {truth.fmt(truth.per(a, b, cut=k))}\n" in out, (h, k)
        assert f"  {h}: {truth.fmt(truth.per(a, b, 'e1'))}\n" in out, h
        assert f"  {h}: {truth.fmt(v.drop('nyc-311'))}\n" in out, h
        mw, lw, hw, dfw = truth.welch(v.loc[PUB].to_numpy(), v.loc[new].to_numpy())
        unit = pd.concat([v.drop(TRIO), pd.Series({"u": v.loc[TRIO].mean()})])
        assert (f" {h}: the 16 new datasets alone {truth.fmt(v.loc[new])}; the seven published datasets minus the new ones "
                f"{mw:+.4f} [{lw:+.4f}, {hw:+.4f}] (Welch, {dfw:.1f} df); New York 2023 datasets as one unit "
                f"{truth.fmt(unit)}\n") in out, h
        if h == "H1":
            assert f"without datasets above 5 % EF stopping-set label lag ({', '.join(lag)}): {truth.fmt(v.drop(lag))}" in out
            cap = ["dvsa-mot", "lichess", "nyc-311"]
            assert f"capped ER_T or EF_T final fits ({', '.join(cap)}): {truth.fmt(v.drop(cap))}" in out
            assert ", ".join(f"{n} {0.0 if n == 'hotels' else 1.0:.2f}" for n in ALL) in out, "equal tree counts"
            assert "tlc-yellow 0.0" not in out
            assert abs(float(v["austin"])) < 1e-4 and "tied 1;" in out, "the planted tie"
            for r in drift.itertuples():
                c = truth.per(a, b, cut=r.cut - 1)
                assert f"{r.dataset} {r.feature} cut {r.cut}: drift AUC {r.drift_auc:.3f}, H1 {c.loc[r.dataset]:+.4f}\n" in out
    for name, (a, b) in {"D - DR": ("d", "dr"), "T - EF": ("t", "ef"), "D - D_own": ("d", "d_own"),
                         "DR - ER": ("dr", "er"), "D_T9 - D_T": ("d_t9", "d_t")}.items():
        assert f"  {name}: {truth.fmt(truth.per(a, b))}\n" in out, name
    assert "multiplier 3.0231" in out and "99.375%" in out and "f1-dnf" not in out
    f2 = fits[(fits.scheme == "e2") & (fits.stage == "final") & fits.arm.isin(["er_t", "ef_t"])].pivot_table(
        index=["dataset", "seed", "cut"], columns="arm", values="best")
    ratio = (f2.er_t / f2.ef_t).groupby("dataset").median()
    assert "median ER_T / EF_T best iteration per dataset: " + ", ".join(f"{n} {r:.2f}" for n, r in ratio.items()) in out
    assert RECODED == ["austin", "kickstarter", "chicago-food", "seattle-stops", "london-fire", "dvsa-mot", "nyc-collisions",
                       "citibike", "lichess", "dota2", "nyc-311"]
    # forest.csv, per_dataset_ext.csv and the gaps file against the same truth
    import io
    forest_rows = pd.read_csv(io.BytesIO(texts[0]["forest.csv"]), keep_default_na=False)
    assert len(forest_rows) == 12
    for r in forest_rows.itertuples():
        a, b = names[r.hypothesis][2:]
        v = truth.per(a, b)
        v = {"23 datasets": v, "16 new datasets": v.loc[new], "without the low-signal datasets": v.drop(low)}[r.block]
        m, lo, hi = truth.interval(v)
        assert (r.first, r.second, r.n) == (a, b, len(v)) and abs(r.estimate - m) < 2e-6 and abs(r.lower - lo) < 2e-6
        assert abs(r.upper - hi) < 2e-6 and abs(r.delta - d) < 2e-6 and abs(r.delta_published_seven - delta_pub) < 2e-6
        assert r.label == (label(lo, hi, d) if r.block == "23 datasets" else ""), r
    pts = pd.read_csv(io.BytesIO(texts[0]["per_dataset_points.csv"]))
    assert len(pts) == 4 * (23 + 16 + 14)
    table = pd.read_csv(io.BytesIO(texts[0]["per_dataset_ext.csv"]))
    row = table[(table.scheme == "e2") & (table.dataset == "hotels") & (table.hyp == "H2") & (table.cut == 0)].iloc[0]
    x = truth.fits[(truth.fits.scheme == "e2") & (truth.fits.dataset == "hotels")].pivot_table(
        index=["seed", "cut"], columns="arm", values="ap")
    seeds = (x["d"] - x["ef"]).groupby("seed").mean().to_numpy()
    ci = ttest_1samp(seeds, 0).confidence_interval(0.95)
    assert abs(row["mean"] - seeds.mean()) < 2e-6 and abs(row.lo95 - ci.low) < 2e-6 and abs(row.hi95 - ci.high) < 2e-6
    gaps = pd.read_csv(io.BytesIO(texts[0]["stopping_set_gaps_ext.csv"]))
    g = gaps[(gaps.dataset == "citibike") & (gaps.cut == 2)].iloc[0]
    assert abs(g.h1 - truth.per("d_t", "dr_t", cut=1)["citibike"]) < 2e-6 and abs(g.h2 - truth.per("d", "ef", cut=1)["citibike"]) < 2e-6
    assert abs(g.persistence - 0.1 * ALL.index("citibike")) < 2e-6
    # negative cases: each must raise
    bad_fits = fits.copy()
    bad_fits.loc[bad_fits.index[7], "ap"] = np.nan
    for what, call in {
            "a NaN score": lambda: check_complete(bad_fits, len(ARMS) * 2 * 3),
            "a failed fit": lambda: check_complete(fits.assign(error=np.where(fits.index == 3, "x", "")), len(ARMS) * 2 * 3),
            "a short unit": lambda: check_complete(fits.iloc[:-1], len(ARMS) * 2 * 3),
            "a dataset without a value": lambda: per_dataset(diff(cells(bad_fits.assign(ap=np.where(
                bad_fits.dataset == "austin", np.nan, bad_fits.ap))), "d_t", "dr_t"), "e2"),
            "subsets that differ from the plan": lambda: report(fits, d, delta_pub, summary.assign(
                low_signal=~summary.low_signal), LEVEL),
            "T rows incomplete": lambda: compute_delta(fits.drop(fits[(fits.arm == "t") & (fits.dataset == "elec")].index[:1]), LEVEL)}.items():
        try:
            call()
        except (AssertionError, KeyError):
            continue
        raise AssertionError(f"no error for {what}")
    print("selftest passed: level 99.375 %, multiplier 3.0231, every printed interval, label, count and csv cell equals an "
          "independent computation, delta equals an enumeration, two runs byte-identical, bad inputs raise")


def reproduce():
    """The extension's functions at the first look's level (98.75 %) on the seven stored published fits kept give the
    first look's A/A delta of the seven datasets (0.00866), its per-dataset points and its pooled rows (estimate, interval
    and label under that delta) in public/per_dataset_points.csv and public/forest.csv."""
    ext.assert_pins()
    fits = read(stored()[0])
    delta_7 = published_delta(fits)
    assert round(delta_7, 5) == 0.00866, delta_7
    w = cells(fits)
    pub = pd.read_csv(ext.FIRST / "public/per_dataset_points.csv")
    forest_ref = pd.read_csv(ext.FIRST / "public/forest.csv").set_index("hypothesis")
    for h, (a, b) in HYP.items():
        mine = diff(w, a, b).xs(("thin", "e2"), level=("pop", "scheme")).groupby("dataset").mean().loc[PUB]
        ref = pub[pub.hypothesis == h].set_index("dataset").value.loc[PUB]
        assert (mine.round(6) - ref).abs().max() < 2e-6, h
        m, lo, hi = pooled(mine, FIRST_LEVEL)
        r = forest_ref.loc[h]
        assert abs(m - r.estimate) < 2e-6 and abs(lo - r.lower) < 2e-6 and abs(hi - r.upper) < 2e-6, h
        assert label(lo, hi, r.delta_recomputed) == r.label and abs(r.delta_recomputed - delta_7) < 2e-6, h
        assert (r["first"], r["second"], r.n) == (a, b, 7), h
    print(f"reproduce: delta of the seven published datasets {delta_7:.5f}; per-dataset H1 to H4 and the pooled rows equal "
          "the public files")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "check":
        check_step(sys.argv[2] if len(sys.argv) > 2 else ext.OUT / "facts_ext.md")
    else:
        {"delta": delta_step, "analyse": analyse_step, "manifest": manifest_step, "selftest": selftest,
         "reproduce": reproduce}.get(
            command, lambda: print(__doc__))()

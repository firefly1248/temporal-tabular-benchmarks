"""es_time_syn_analyse: analysis of the synthetic study. Plan: early_stopping_time_synthetic_plan.md.

analyse    the check, registered contrasts P1-P12 and R1-R11, the Q1 decision, descriptive cell tables and diagnostics
           -> results/es_time/syn/{analysis_syn.txt, contrasts_syn.csv, cells_syn.csv, outputs_manifest_syn.json}
selftest   the analysis on fake fits with planted effects, against an independent computation; reads no results
"""
import hashlib
import json
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from scripts import es_time_syn as syn

K = 23
LEVEL = 1 - 0.05 / K
CONTRASTS = {"H1": ("d_t", "dr_t"), "H2": ("d", "ef"), "H3": ("t", "d_own"), "H4": ("t", "er"),
             "T-D_T": ("t", "d_t"), "D_T-D_own": ("d_t", "d_own"), "T-DR_T": ("t", "dr_t"),
             "DR_T-DR": ("dr_t", "dr"), "DR-ER": ("dr", "er"), "T-EF": ("t", "ef"), "ER-EF": ("er", "ef")}
E_POSITIONS = {"stop": [(5800, 0), (7150, 1), (8450, 2)], "deep": [(5800, 1), (7150, 2)], "late": [(5800, 2)],
               "test": [(7150, 0), (8450, 1)]}
B_POSITIONS = {"blip": [("B_c1", 0), ("B_c2", 1), ("B_c3", 2)], "window": [("B_c1", 1), ("B_c1", 2), ("B_c2", 2)],
               "test": [("B_c2", 0), ("B_c3", 1)]}
EXTRA_PAIRS = [("H_slow_t150", "H_entity_t150"), ("H_row_t75", "A_medium_p30"), ("H_row_t150", "A_medium_p30")]
REAL_LOW_H3 = "+0.0069, 95 % [+0.0023, +0.0116], 99.375 % [-0.0004, +0.0142]"
REAL_GRADIENT = "+0.0052, Welch 95 % [+0.0001, +0.0102]"  # low-signal minus the other 14, exploratory split
FILES = ("analysis_syn.txt", "contrasts_syn.csv", "cells_syn.csv")


def per_cut(fits):
    """Contrast values per cell, replicate and cut from the E2 final fits."""
    f = fits[fits.stage == "final"].pivot_table(index=["dataset", "seed", "cut"], columns="arm", values="ap")
    return pd.DataFrame({k: f[a] - f[b] for k, (a, b) in CONTRASTS.items()})


def unit(v, cell):
    """Replicate-level values of one cell: the mean over the three cuts."""
    return v.xs(cell, level="dataset").groupby("seed").mean()


def pairs(v, cell_cuts, ref):
    """Per replicate, the mean over (cell, cut) pairs of the cell minus `ref` at that cut."""
    at = lambda c, k: v.xs((c, k), level=("dataset", "cut"))
    return sum(at(c, k) - at(ref, k) for c, k in cell_cuts) / len(cell_cuts)


def e_position(v, pos):
    """Family E: weak and medium signal pooled with equal weight."""
    return sum(pairs(v, [(f"E_r{rho}_{sig}", k) for rho, k in E_POSITIONS[pos]], f"A_{sig}_p30")
               for sig in ("weak", "medium")) / 2


def b_position(v, pos):
    return pairs(v, B_POSITIONS[pos], "A_medium_p30")


def registered(v):
    """The 23 registered contrasts as replicate-level series."""
    u = lambda c: unit(v, c)
    weak, medium = ((u(f"A_{s}_p10") + u(f"A_{s}_p30")) / 2 for s in ("weak", "medium"))
    ref = u("A_medium_p30")
    e = {pos: e_position(v, pos) for pos in ("stop", "deep")}
    b = b_position(v, "blip")
    h = u("H_entity_t150") - u("H_row_t150")
    us = u("U_raw_g32") - u("U_norm_g32")
    return [
        ("P1", "A", "H3, weak", ">", weak.H3),
        ("P2", "A", "H3 weak - H3 medium", ">", weak.H3 - medium.H3),
        ("P3", "A", "H4, weak", ">", weak.H4),
        ("R1", "A", "H4 weak - H4 medium", "", weak.H4 - medium.H4),
        ("R2", "A", "T - D_T, weak", "", weak["T-D_T"]),
        ("R3", "A", "T - DR_T, weak", "", weak["T-DR_T"]),
        ("P4", "A", "DR - ER, weak", ">", weak["DR-ER"]),
        ("R4", "A", "H1, weak", "", weak.H1),
        ("P5", "H", "H2 entity - H2 row, tau 1.5", ">", h.H2),
        ("P6", "H", "H1 entity - H1 row, tau 1.5", "<", h.H1),
        ("R5", "H", "H4 entity - H4 row, tau 1.5", "", h.H4),
        ("P7", "E", "H2 stop - reference", ">", e["stop"].H2),
        ("P8", "E", "H2 deep - reference", ">", e["deep"].H2),
        ("R6", "E", "H1 stop - reference", "", e["stop"].H1),
        ("R7", "E", "H1 deep - reference", "", e["deep"].H1),
        ("P9", "B", "H1 blip - reference", "<", b.H1),
        ("P10", "B", "H2 blip - reference", "<", b.H2),
        ("P11", "B", "H3 blip - reference", ">", b.H3),
        ("R8", "C", "H1 - reference", "", u("C").H1 - ref.H1),
        ("R9", "U", "H2 raw - H2 normalised, gamma ln 32", "", us.H2),
        ("R10", "U", "H4 raw - H4 normalised, gamma ln 32", "", us.H4),
        ("R11", "U", "H1 raw - H1 normalised, gamma ln 32", "", us.H1),
        ("P12", "X", "H2 kappa 8 - reference", ">", u("X_k8").H2 - ref.H2),
    ]


def interval(x, level):
    x = np.asarray(x, float)
    m, hw = x.mean(), stats.t.ppf(1 - (1 - level) / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - hw, m + hw


def verdict(pred, lo, hi):
    if not pred:
        return "moves up" if lo > 0 else "moves down" if hi < 0 else "not resolved"
    above, below = lo > 0, hi < 0
    if pred == ">":
        return "supported" if above else "contradicted" if below else "unresolved"
    return "supported" if below else "contradicted" if above else "unresolved"


def q1_decision(p1, p2):
    sign = {"supported": "reproduces H3's sign", "contradicted": "gives H3 the opposite sign",
            "unresolved": "does not resolve H3's sign"}[p1]
    gradient = {"supported": "a gradient in the direction of the exploratory real split",
                "contradicted": "the opposite gradient", "unresolved": "no resolved gradient"}[p2]
    return f"{sign}, with {gradient}"


def fmt(m, lo, hi):
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def assert_after(fits):
    """Cut 1 of each 'after' cell has the same final fits as its A reference."""
    f = fits[(fits.stage == "final") & (fits.cut == 0)]
    cols = ["seed", "arm", "cand", "ap", "trees"]
    for cell, (ref, _) in syn.AFTER.items():
        a, b = (f[f.dataset == c][cols].sort_values(["seed", "arm"]).reset_index(drop=True) for c in (cell, ref))
        assert a.equals(b), f"{cell} cut 1 differs from {ref}"


@lru_cache(None)
def balanced(seed):
    return [c["class_weight"] == "balanced" for c in syn.es.candidates("syn", seed)]


def diag_row(g, inner):
    """Fit diagnostics of final fits `g` and inner ER/EF fits `inner` (one cell, or the pairs of one position)."""
    es = g[g.arm.isin(["er", "ef", "er_t", "ef_t", "d_own_es"])]
    idx = ["dataset", "seed", "cut"]
    picks = g.pivot_table(index=idx, columns="arm", values="cand")
    row = {f"trees_{a}": g[g.arm == a].trees.median() for a in ("t", "er", "ef", "d_own", "d_t", "dr_t")}
    row |= {"capped_final": (es.best > 480).mean(), "capped_inner": (inner.best > 480).mean(),
            "le5_final": (es.best <= 5).mean(), "le5_inner": (inner.best <= 5).mean()}
    row |= {f"same_{a}_{b}": (picks[a] == picks[b]).mean()
            for a, b in (("t", "er"), ("t", "ef"), ("er", "ef"), ("t", "d_own"), ("ef", "d_own"))}
    row |= {f"ap_sd_{a}": g[g.arm == a].groupby("seed").ap.mean().std() for a in ("er", "ef", "d", "dr", "t")}
    for arm in ("er_t", "ef_t"):
        r = g[g.arm == arm]
        row |= {f"ap_iter_minus_best_{arm}": (r.best_ap_iter - r.best).median(),
                f"early_share_{arm}": (r.best < r.best_ap_iter).mean()}
    ratio = g[g.arm.isin(["er_t", "ef_t"])].pivot_table(index=idx, columns="arm", values=["best", "cand"])
    bal = np.array([balanced(s)[int(c)] for (_, s, _), c in ratio[("cand", "ef_t")].items()])
    lr = np.log(ratio[("best", "ef_t")] / ratio[("best", "er_t")]).to_numpy()
    row |= {"log_ef_er_count_balanced": np.median(lr[bal]) if bal.any() else np.nan,
            "log_ef_er_count_unweighted": np.median(lr[~bal]) if (~bal).any() else np.nan}
    return row


def fit_diagnostics(fits):
    final = fits[fits.stage == "final"]
    inner = fits[(fits.stage == "inner") & fits.arm.isin(["er", "ef"])]
    out = {}
    for cell, g in final.groupby("dataset"):
        row = diag_row(g, inner[inner.dataset == cell])
        if cell.startswith("H_"):
            for arm in ("er", "ef"):
                r = g[g.arm == arm]
                row |= {f"unseen_stop_{arm}": r.unseen_stop.mean(), f"unseen_tgt_{arm}": r.unseen_tgt.mean()}
        out[cell] = row
    groups = {f"E {pos}": [(f"E_r{rho}_{s}", k) for rho, k in E_POSITIONS[pos] for s in ("weak", "medium")]
              for pos in ("stop", "deep")} | {"B blip": B_POSITIONS["blip"]}
    at = lambda df, p: df[pd.MultiIndex.from_arrays([df.dataset, df.cut]).isin(p)]
    for label, p in groups.items():
        out[label] = diag_row(at(final, p), at(inner, p))
    return pd.DataFrame(out).T


def report(fits, diag=None):
    assert (fits.error == "").all() and fits.ap.notna().all(), "failed fits"
    assert_after(fits)
    v = per_cut(fits)
    reps = sorted(v.index.get_level_values("seed").unique())
    n = len(reps)
    L = [f"Synthetic study: {v.index.get_level_values('dataset').nunique()} cells x {n} replicates; registered "
         f"contrasts at level 1 - 0.05/{K} = {LEVEL:.5f}, two-sided t-interval over {n} replicate-level values "
         f"({n - 1} df, multiplier {stats.t.ppf(1 - (1 - LEVEL) / 2, n - 1):.3f})", ""]
    a_cells = [c for c in syn.CELLS if c.startswith(("A_medium", "A_strong"))]
    m, lo, hi = interval(sum(unit(v, c).H1 for c in a_cells) / len(a_cells), 0.90)
    L.append(f"Check: H1 in family A, medium and strong signal, four cells pooled: {m:+.4f}, 90 % interval "
             f"[{lo:+.4f}, {hi:+.4f}]; " + ("equivalent to zero within 0.003 (two one-sided tests at 5 %)"
                                            if -0.003 < lo and hi < 0.003 else
                                            "FLAGGED: not shown equivalent to zero within 0.003; reported, "
                                            "no contrast withdrawn"))
    L += ["Check: cut 1 of the 'after' cells equals its A reference (asserted)", ""]
    rows = []
    for pid, fam, name, pred, x in registered(v):
        assert list(x.index) == reps
        m, lo, hi = interval(x, LEVEL)
        rows.append(dict(id=pid, family=fam, contrast=name, predicted=pred or "none", estimate=m, lower=lo, upper=hi,
                         verdict=verdict(pred, lo, hi), n=n))
    reg = pd.DataFrame(rows)
    L.append("Registered contrasts")
    for r in reg.itertuples():
        L.append(f" {r.id:<4}{r.family}  {r.contrast:<40} predicted {r.predicted:<5} {fmt(r.estimate, r.lower, r.upper)}"
                 f"  {r.verdict}")
    p1, p2 = (reg.set_index("id").loc[k] for k in ("P1", "P2"))
    weak = registered(v)[0][4]
    L += ["", f"Q1 decision: the generator {q1_decision(p1.verdict, p2.verdict)} (P1 {p1.verdict}, P2 {p2.verdict})",
          f"Q1 size, descriptive: weak-signal H3 {p1.estimate:+.4f}, 95 % [{interval(weak, 0.95)[1]:+.4f}, "
          f"{interval(weak, 0.95)[2]:+.4f}], 99.375 % [{interval(weak, 0.99375)[1]:+.4f}, "
          f"{interval(weak, 0.99375)[2]:+.4f}]; real low-signal H3 {REAL_LOW_H3}",
          f"Q1 gradient, descriptive: weak minus medium H3 {p2.estimate:+.4f}; real low-signal minus the other 14 "
          f"{REAL_GRADIENT}"]

    L += ["", "Descriptive, unadjusted 95 % intervals, no claim: per cell, mean over cuts and replicates"]
    cells = []
    for cell in syn.CELLS:
        u = unit(v, cell)
        for k in CONTRASTS:
            m, lo, hi = interval(u[k], 0.95)
            cells.append(dict(cell=cell, contrast=k, estimate=m, lower=lo, upper=hi, n=len(u)))
    ct = pd.DataFrame(cells)
    wide = ct.assign(txt=[fmt(r.estimate, r.lower, r.upper) for r in ct.itertuples()]).pivot(
        index="cell", columns="contrast", values="txt").loc[list(syn.CELLS), list(CONTRASTS)]
    L.append(wide.to_string())
    four = lambda x: "; ".join(f"{k} {fmt(*interval(x[k], 0.95))}" for k in ("H1", "H2", "H3", "H4"))
    L += ["", "Descriptive: family E by position, cell minus A reference, weak and medium pooled (95 %)"]
    L += [f" {pos:<6} {four(e_position(v, pos))}" for pos in E_POSITIONS]
    L += ["", "Descriptive: family B by position, cell minus A_medium_p30 (95 %; per-cell rows above mix positions)"]
    L += [f" {pos:<6} {four(b_position(v, pos))}" for pos in B_POSITIONS]
    L += ["", "Descriptive: cells minus their reference (95 %)"]
    for cell, ref in [(c, syn.reference_cell(c)) for c in syn.CELLS
                      if syn.CELLS[c]["family"] in ("H", "C", "U", "X")] + EXTRA_PAIRS:
        if cell != ref:
            L.append(f" {cell:<14} - {ref:<14} {four(unit(v, cell) - unit(v, ref))}")
    L += ["", "Diagnostics from the fits, per cell and for the registered E and B positions (medians of tree counts; "
          "shares over early-stopping fits or (replicate, cut); ap_sd: replicate sd of the arm's AP)",
          fit_diagnostics(fits).round(3).to_string()]
    if diag is not None:
        L += ["", "Diagnostics of the data (means over replicates; drift AUC on replicate 1000, cut 1)",
              diag.round(3).to_string()]
    return L, reg, ct


def data_diagnostics(path):
    d = pd.read_csv(path)
    reps, drift = d[d.rep >= 0], d[d.rep < 0].set_index("cell").filter(like="drift_auc_")
    low = reps.filter(like="low_signal_auc_c")
    reps = reps.assign(low_signal_share=(low < 0.65).all(axis=1).astype(float))
    cols = ["prevalence", "low_signal_share"] + [c for c in reps.columns if c.startswith(
        ("prev_test", "prev_stop", "stop_gap", "auc_true", "auc_f", "known_", "low_signal_auc", "stale_", "x2_beyond",
         "x3_below"))]
    feat = drift.drop(columns="drift_auc_entity")
    top = pd.DataFrame({"drift_auc_max": feat.max(axis=1), "drift_top": feat.idxmax(axis=1).str.removeprefix("drift_auc_"),
                        "drift_auc_entity": drift.drift_auc_entity})
    return reps.groupby("cell")[cols].mean().join(top).loc[list(syn.CELLS)]


def write(fits, diag, out):
    lines, reg, ct = report(fits, diag)
    out = Path(out)
    (out / "analysis_syn.txt").write_text("\n".join(lines) + "\n")
    reg.round(6).to_csv(out / "contrasts_syn.csv", index=False)
    ct.round(6).to_csv(out / "cells_syn.csv", index=False)
    return lines


def analyse():
    syn.assert_pins()
    files = sorted(syn.FITS.glob("*.joblib"))
    assert len(files) == len(syn.CELLS) * len(syn.REPS), f"{len(files)} units, expected {len(syn.CELLS) * len(syn.REPS)}"
    fits = pd.concat([joblib.load(f) for f in files], ignore_index=True)
    assert fits.syn_sha.nunique() == 1, "units come from more than one generator version"
    lines = write(fits, data_diagnostics(syn.OUT / "diagnostics_syn.csv"), syn.OUT)
    print("\n".join(lines))
    path = syn.OUT / "outputs_manifest_syn.json"
    now = {n: hashlib.sha256((syn.OUT / n).read_bytes()).hexdigest() for n in FILES}
    if path.exists():
        assert json.load(open(path)) == now, "this run is not byte-identical to the last"
        print("outputs byte-identical to the manifest")
    else:
        json.dump(now, open(path, "w"), indent=1)


# ---------------------------------------------------------------- self-test

def fake_fits(effects, reps=range(12), noise=0.002):
    """Final and inner rows for every cell; arm AP = 0.5 + replicate effect + planted effects + noise. `effects` maps
    cell -> {arm: offset on every cut, (arm, cut): offset on that cut}; cut 1 of each 'after' cell copies its reference."""
    rng = np.random.default_rng(0)
    arms = ["t", "er", "ef", "d", "dr", "d_own_es", "d_own", "er_t", "ef_t", "dr_t", "d_t", "d_t9"]
    rows = []
    for r in reps:
        base = 0.05 * rng.standard_normal()
        for cell in syn.CELLS:
            for k in range(3):
                e = effects.get(cell, {})
                for i, arm in enumerate(arms):
                    rows.append(dict(dataset=cell, seed=r, cut=k, stage="final", arm=arm, cand=i % 3,
                                     ap=0.5 + base + e.get(arm, 0.0) + e.get((arm, k), 0.0) + noise * rng.standard_normal(),
                                     trees=50 + i, best=50 + i, best_ap_iter=60, error="", unseen_stop=0.1,
                                     unseen_tgt=0.2))
    f = pd.DataFrame(rows)
    for cell, (ref, _) in syn.AFTER.items():
        src = f[(f.dataset == ref) & (f.cut == 0)].assign(dataset=cell)
        f = pd.concat([f[~((f.dataset == cell) & (f.cut == 0))], src], ignore_index=True)
    inner = f[f.arm.isin(["t", "er", "ef", "d_own"])].assign(stage="inner")
    return pd.concat([f, inner], ignore_index=True)


def independent(f, cells_a, cells_b, a, b):
    """Mean over replicates of (cells_a mean - cells_b mean) of arm a minus arm b, computed directly."""
    g = f[(f.stage == "final")]
    val = lambda cells: g[g.dataset.isin(cells)].pivot_table(index=["seed", "dataset", "cut"], columns="arm",
                                                             values="ap").pipe(lambda w: (w[a] - w[b])).groupby(
        ["seed", "dataset"]).mean().groupby("seed").mean()
    x = val(cells_a) - (val(cells_b) if cells_b else 0)
    return x.mean()


def lines_of(f):
    with tempfile.TemporaryDirectory() as tmp:
        write(f, None, tmp)
        return Path(tmp, "analysis_syn.txt").read_text()


def selftest():
    eff = {"A_weak_p10": {"t": 0.006, "dr": 0.002, "d_t": -0.004}, "A_weak_p30": {"t": 0.006, "dr": 0.002, "d_t": -0.004},
           "A_strong_p10": {"t": 0.001}, "A_strong_p30": {"t": 0.001},
           "H_entity_t150": {"t": 0.01, "d_t": -0.02, "ef": -0.03}, "C": {"d_t": -0.01},
           "U_raw_g32": {"t": 0.02, "ef": -0.02}, "U_norm_g32": {"t": 0.005}, "X_k8": {"ef": -0.015}}
    # families E and B: effects only on the registered (cell, cut) pairs, so a wrong position table shows
    for pos, ef in (("stop", -0.01), ("deep", -0.02)):
        for rho, k in E_POSITIONS[pos]:
            for sig in ("weak", "medium"):
                eff.setdefault(f"E_r{rho}_{sig}", {}).update({("ef", k): ef, ("d_t", k): 0.015})
    for k in range(3):
        eff[f"B_c{k + 1}"] = {("d_t", k): -0.01, ("d", k): -0.01, ("d_own", k): -0.02}
    f = fake_fits(eff)
    with tempfile.TemporaryDirectory() as tmp:
        texts = [write(f, None, tmp) and Path(tmp, "analysis_syn.txt").read_bytes() for _ in range(2)]
        reg = pd.read_csv(Path(tmp) / "contrasts_syn.csv").set_index("id")
    assert texts[0] == texts[1], "two runs differ"
    assert len(reg) == K and sorted(reg.index) == sorted([f"P{i}" for i in range(1, 13)] + [f"R{i}" for i in range(1, 12)])
    weak, medium = ["A_weak_p10", "A_weak_p30"], ["A_medium_p10", "A_medium_p30"]
    expect = {"P1": independent(f, weak, None, "t", "d_own"), "P3": independent(f, weak, None, "t", "er"),
              "R2": independent(f, weak, None, "t", "d_t"), "R3": independent(f, weak, None, "t", "dr_t"),
              "P4": independent(f, weak, None, "dr", "er"), "R4": independent(f, weak, None, "d_t", "dr_t"),
              "P2": independent(f, weak, None, "t", "d_own") - independent(f, medium, None, "t", "d_own"),
              "R1": independent(f, weak, None, "t", "er") - independent(f, medium, None, "t", "er"),
              "P5": independent(f, ["H_entity_t150"], ["H_row_t150"], "d", "ef"),
              "P6": independent(f, ["H_entity_t150"], ["H_row_t150"], "d_t", "dr_t"),
              "R5": independent(f, ["H_entity_t150"], ["H_row_t150"], "t", "er"),
              "R8": independent(f, ["C"], ["A_medium_p30"], "d_t", "dr_t"),
              "R9": independent(f, ["U_raw_g32"], ["U_norm_g32"], "d", "ef"),
              "R10": independent(f, ["U_raw_g32"], ["U_norm_g32"], "t", "er"),
              "R11": independent(f, ["U_raw_g32"], ["U_norm_g32"], "d_t", "dr_t"),
              "P12": independent(f, ["X_k8"], ["A_medium_p30"], "d", "ef")}
    for k, x in expect.items():
        assert abs(reg.loc[k, "estimate"] - round(x, 6)) < 2e-6, (k, reg.loc[k, "estimate"], x)
    # families E and B: mean over the position's pairs (E: and both signals) of cell minus reference, by hand
    g = f[f.stage == "final"].pivot_table(index=["dataset", "seed", "cut"], columns="arm", values="ap")
    hand = {"E": lambda pos, r, h: np.mean([h[(f"E_r{rho}_{s}", r, k)] - h[(f"A_{s}_p30", r, k)]
                                            for rho, k in E_POSITIONS[pos] for s in ("weak", "medium")]),
            "B": lambda pos, r, h: np.mean([h[(c, r, k)] - h[("A_medium_p30", r, k)] for c, k in B_POSITIONS[pos]])}
    for pid, fam, pos, (a, b) in (("P7", "E", "stop", ("d", "ef")), ("P8", "E", "deep", ("d", "ef")),
                                  ("R6", "E", "stop", ("d_t", "dr_t")), ("R7", "E", "deep", ("d_t", "dr_t")),
                                  ("P9", "B", "blip", ("d_t", "dr_t")), ("P10", "B", "blip", ("d", "ef")),
                                  ("P11", "B", "blip", ("t", "d_own"))):
        h = g[a] - g[b]
        x = np.mean([hand[fam](pos, r, h) for r in range(12)])
        assert abs(reg.loc[pid, "estimate"] - round(x, 6)) < 2e-6, (pid, reg.loc[pid, "estimate"], x)
    planted = {"P7": 0.01, "P8": 0.02, "R6": 0.015, "R7": 0.015, "P9": -0.01, "P10": -0.01, "P11": 0.02}
    for pid, x in planted.items():  # literals, independent of the position tables
        assert abs(reg.loc[pid, "estimate"] - x) < 0.002, (pid, reg.loc[pid, "estimate"], x)
    for k, v in {"P1": "supported", "P2": "supported", "P3": "supported", "P4": "supported", "P5": "supported",
                 "P6": "supported", "P7": "supported", "P8": "supported", "P9": "supported", "P10": "supported",
                 "P11": "supported", "P12": "supported", "R2": "moves up", "R4": "moves down", "R5": "moves up",
                 "R6": "moves up", "R8": "moves down", "R9": "moves up"}.items():
        assert reg.loc[k, "verdict"] == v, (k, reg.loc[k, "verdict"])
    text = texts[0].decode()
    assert "Q1 decision: the generator reproduces H3's sign, with a gradient in the direction" in text
    assert "equivalent to zero within 0.003" in text and "FLAGGED" not in text
    eff2 = eff | {"A_medium_p10": {"t": 0.02}, "A_medium_p30": {"t": 0.02, "d_t": 0.02}}
    text2 = lines_of(fake_fits(eff2))
    assert "Q1 decision: the generator reproduces H3's sign, with the opposite gradient" in text2
    assert "FLAGGED: not shown equivalent" in text2
    assert q1_decision("supported", "unresolved") == "reproduces H3's sign, with no resolved gradient"
    assert q1_decision("contradicted", "supported").startswith("gives H3 the opposite sign, with a gradient")
    m, lo, hi = interval(np.r_[1.0, 2.0, 3.0], 0.95)
    r = stats.ttest_1samp([1.0, 2.0, 3.0], 0).confidence_interval(0.95)
    assert abs(lo - r.low) < 1e-12 and abs(hi - r.high) < 1e-12
    assert verdict(">", -1, -0.5) == "contradicted" and verdict("<", 0.1, 1) == "contradicted"
    assert verdict("", -1, -0.5) == "moves down" and verdict("", -1, 1) == "not resolved"
    for cell in syn.AFTER:
        bad = f.copy()
        bad.loc[(bad.dataset == cell) & (bad.cut == 0) & (bad.arm == "t"), "ap"] += 1e-9
        try:
            report(bad)
            raise SystemExit(f"after identity of {cell} not asserted")
        except AssertionError:
            pass
    print("selftest passed: registered estimates equal a direct computation, verdicts, the Q1 decision and the check "
          "follow the planted effects, two runs byte-identical, the after identities are asserted")


if __name__ == "__main__":
    {"analyse": analyse, "selftest": selftest}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()

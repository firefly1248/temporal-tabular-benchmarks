"""es_time_ext_explore: exploratory, outside the plan's registered analysis; no labels, changes no registered number.

Per dataset (thin E2, the 23 datasets): the median ER_T / EF_T best-iteration ratio over seeds and cuts, against H1
(D_T - DR_T) and H3 (T - D_own), each the mean over three cuts and five seeds, with Spearman's rho over the 23 datasets.
Reads the stored fits whose hashes delta_ext.json records -> results/es_time/ext/explore_ext.txt

Usage: python -m scripts.es_time_ext_explore
"""
import json

import pandas as pd
from scipy.stats import spearmanr

import scripts.es_time_ext as ext
from scripts.es_time_ext_analyse import ALL, HYP, cells, diff, per_dataset, read, stored


def main():
    pub, new = stored()
    assert {n: ext.sha256(p) for n, p in new.items()} == json.load(open(ext.OUT / "delta_ext.json"))["inputs"]
    w = cells(read({**pub, **new}))
    ratio = (w["best"]["er_t"] / w["best"]["ef_t"]).xs(("thin", "e2"), level=("pop", "scheme")).groupby("dataset").median()
    table = pd.DataFrame({"ratio": ratio.loc[ALL], **{h: per_dataset(diff(w, *HYP[h]), "e2") for h in ("H1", "H3")}})
    lines = ["Exploratory (not registered, no labels): median ER_T / EF_T best iteration per dataset against H1 and H3, "
             f"thin E2, {len(table)} datasets",
             "ratio: median over 5 seeds and 3 cuts of the final fits; H1 = D_T - DR_T, H3 = T - D_own, mean over cuts and seeds", ""]
    for h in ("H1", "H3"):
        r = spearmanr(table.ratio, table[h])
        lines.append(f"Spearman rho, ratio vs {h}: {r.statistic:+.3f} (two-sided p {r.pvalue:.3f}, n {len(table)})")
    lines += ["", f"ratio above 1 on {int((table.ratio > 1).sum())} of {len(table)} datasets, equal to 1 on "
                  f"{int((table.ratio == 1).sum())}", "",
              table.sort_values("ratio").to_string(formatters={"ratio": "{:.2f}".format, "H1": "{:+.4f}".format,
                                                               "H3": "{:+.4f}".format})]
    (ext.OUT / "explore_ext.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

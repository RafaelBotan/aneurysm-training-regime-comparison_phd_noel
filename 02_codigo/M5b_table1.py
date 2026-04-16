"""
M5b — Tabela 1 descritiva: 3 coortes MCA lado a lado.

Colunas: HUG MCA | Pseudo-ext MCA | CMHA MCA
Linhas: N, Ruptured, Prevalencia, Demographics (age, sex), 11 features (median [IQR])
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kruskal, chi2_contingency

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:/doutorado_noel")
IN_CSV = ROOT / "03_analises" / "M2_shared_feature_core.csv"
OUT_DIR = ROOT / "03_analises"

FEATURES = ["AR", "BF", "CP", "EI", "NSI", "UI", "Dmax", "Dn", "H", "S", "V"]


def load_cohorts():
    df = pd.read_csv(IN_CSV)
    hug = df[(df.cohort == "AneuX")
             & df.subcohort.isin(["hug2016", "hug2016snf"])
             & df.location.fillna("").str.contains("MCA")
             & df.ruptured.notna()].dropna(subset=FEATURES).reset_index(drop=True)
    pex = df[(df.cohort == "AneuX")
             & df.subcohort.isin(["aneurisk", "aneurist"])
             & df.location.fillna("").str.contains("MCA")
             & df.ruptured.notna()].dropna(subset=FEATURES).reset_index(drop=True)
    cmha = df[(df.cohort == "CMHA") & df.ruptured.notna()]\
              .dropna(subset=FEATURES).reset_index(drop=True)
    return hug, pex, cmha


def med_iqr(series):
    q1, q2, q3 = series.quantile([0.25, 0.5, 0.75])
    return f"{q2:.2f} [{q1:.2f}–{q3:.2f}]"


def main():
    hug, pex, cmha = load_cohorts()
    cohorts = [("HUG MCA", hug), ("Pseudo-ext MCA", pex), ("CMHA MCA", cmha)]
    rows = []

    # N
    rows.append({"Variable": "N (aneurysms)",
                 **{c[0]: str(len(c[1])) for c in cohorts}, "p": ""})

    # Ruptured
    rows.append({"Variable": "Ruptured, n (%)",
                 **{c[0]: f"{int(c[1].ruptured.sum())} ({100*c[1].ruptured.mean():.1f}%)"
                    for c in cohorts}, "p": ""})

    # Age (if available)
    age_cols = [c for c in hug.columns if "age" in c.lower()]
    if age_cols:
        acol = age_cols[0]
        for label, df in cohorts:
            if acol not in df.columns or df[acol].isna().all():
                age_available = False
                break
        else:
            age_available = True
        if age_available:
            vals = [df[acol].dropna() for _, df in cohorts]
            _, p_age = kruskal(*vals)
            rows.append({"Variable": f"Age, median [IQR]",
                         **{c[0]: med_iqr(c[1][acol].dropna()) for c in cohorts},
                         "p": f"{p_age:.3f}"})

    # Sex (if available)
    sex_cols = [c for c in hug.columns if "sex" in c.lower() or "gender" in c.lower()]
    if sex_cols:
        scol = sex_cols[0]
        all_have = all(scol in df.columns and not df[scol].isna().all() for _, df in cohorts)
        if all_have:
            def female_pct(df):
                s = df[scol].dropna()
                if s.dtype == object:
                    n_f = s.str.lower().str.startswith("f").sum()
                else:
                    n_f = (s == 0).sum()
                return f"{n_f} ({100*n_f/len(s):.1f}%)"
            rows.append({"Variable": "Female, n (%)",
                         **{c[0]: female_pct(c[1]) for c in cohorts}, "p": ""})

    # Features
    for f in FEATURES:
        vals = [df[f].dropna() for _, df in cohorts]
        _, p_kw = kruskal(*vals)
        rows.append({"Variable": f"{f}, median [IQR]",
                     **{c[0]: med_iqr(c[1][f]) for c in cohorts},
                     "p": f"{p_kw:.4f}"})

    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    tbl.to_csv(OUT_DIR / "M5b_table1.csv", index=False)
    print(f"\n=> {OUT_DIR / 'M5b_table1.csv'}")


if __name__ == "__main__":
    main()

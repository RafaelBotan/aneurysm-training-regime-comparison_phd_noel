"""
M4d — Distribuicao das features nos ROMPIDOS MCA.

HUG MCA treino: 14 rompidos
AneuX pseudo-external MCA: 30 rompidos
CMHA MCA stress: 77 rompidos

Se a separacao rompidos-vs-nao e diferente entre HUG e pseudo-external (mesmo com
non-rompidos alinhados), isso explica por que modelo treinado em HUG falha em external.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:/doutorado_noel")
IN_CSV = ROOT / "03_analises" / "M2_shared_feature_core.csv"
OUT_DIR = ROOT / "03_analises"
FEATURES = ["AR", "BF", "CP", "EI", "NSI", "UI", "Dmax", "Dn", "H", "S", "V"]


def cohens_d(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return np.nan
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if pooled == 0: return np.nan
    return (x.mean() - y.mean()) / pooled


def main():
    df = pd.read_csv(IN_CSV)
    mca = df[(df.cohort == "AneuX")
             & df.location.fillna("").str.contains("MCA")
             & df.ruptured.notna()].dropna(subset=FEATURES)
    cmha = df[(df.cohort == "CMHA") & df.ruptured.notna()].dropna(subset=FEATURES)

    mca = mca.assign(site=np.where(
        mca.subcohort.isin(["hug2016", "hug2016snf"]), "HUG",
        "Pseudo_ext",  # aneurisk + aneurist combined
    ))

    # Rompidos HUG
    hug_r = mca[(mca.site == "HUG") & (mca.ruptured == 1)]
    # Rompidos pseudo-external
    ext_r = mca[(mca.site == "Pseudo_ext") & (mca.ruptured == 1)]
    # Rompidos CMHA
    cmha_r = cmha[cmha.ruptured == 1]
    # Nao-rompidos por site (referencia)
    hug_u = mca[(mca.site == "HUG") & (mca.ruptured == 0)]
    ext_u = mca[(mca.site == "Pseudo_ext") & (mca.ruptured == 0)]
    cmha_u = cmha[cmha.ruptured == 0]

    print(f"Ns: HUG R/U = {len(hug_r)}/{len(hug_u)}  "
          f"Pseudo_ext R/U = {len(ext_r)}/{len(ext_u)}  "
          f"CMHA R/U = {len(cmha_r)}/{len(cmha_u)}")
    print()

    print("=== Separacao rompidos vs nao-rompidos DENTRO DE CADA COORTE (Cohen's d) ===")
    print(f"{'Feature':6s}  {'d_HUG':>8s}  {'d_PseudoExt':>12s}  {'d_CMHA':>8s}")
    for f in FEATURES:
        d_hug = cohens_d(hug_r[f], hug_u[f])
        d_ext = cohens_d(ext_r[f], ext_u[f])
        d_ch  = cohens_d(cmha_r[f], cmha_u[f])
        print(f"{f:6s}  {d_hug:+8.2f}  {d_ext:+12.2f}  {d_ch:+8.2f}")
    print()

    print("=== Rompidos: HUG vs Pseudo_ext (mesmo dataset, mesma convencao, subcoortes) ===")
    print(f"{'Feature':6s}  {'HUG_R_med':>10s}  {'PseudoExt_R_med':>16s}  "
          f"{'KS':>6s}  {'p':>8s}  {'d':>7s}")
    sig = 0
    for f in FEATURES:
        ks = ks_2samp(hug_r[f], ext_r[f])
        d = cohens_d(hug_r[f], ext_r[f])
        flag = "<<<" if (ks.pvalue < 0.05 and abs(d) > 0.3) else ""
        if flag: sig += 1
        print(f"{f:6s}  {hug_r[f].median():10.3f}  {ext_r[f].median():16.3f}  "
              f"{ks.statistic:6.3f}  {ks.pvalue:8.4f}  {d:+7.2f} {flag}")
    print(f"\n>> Rompidos HUG vs Pseudo_ext: {sig}/{len(FEATURES)} features com shift")
    print()

    print("=== Rompidos: HUG vs CMHA (diferente dataset, diferente convencao) ===")
    print(f"{'Feature':6s}  {'HUG_R_med':>10s}  {'CMHA_R_med':>11s}  "
          f"{'KS':>6s}  {'p':>8s}  {'d':>7s}")
    sig2 = 0
    for f in FEATURES:
        ks = ks_2samp(hug_r[f], cmha_r[f])
        d = cohens_d(hug_r[f], cmha_r[f])
        flag = "<<<" if (ks.pvalue < 0.05 and abs(d) > 0.3) else ""
        if flag: sig2 += 1
        print(f"{f:6s}  {hug_r[f].median():10.3f}  {cmha_r[f].median():11.3f}  "
              f"{ks.statistic:6.3f}  {ks.pvalue:8.4f}  {d:+7.2f} {flag}")
    print(f"\n>> Rompidos HUG vs CMHA: {sig2}/{len(FEATURES)} features com shift")


if __name__ == "__main__":
    main()

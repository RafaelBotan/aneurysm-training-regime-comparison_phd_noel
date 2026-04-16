"""
M4c — Auditoria de site shift intra-AneuX (HUG vs aneurisk vs aneurist, MCA).

Teste hipotese GPT: pseudo-external (aneurisk+aneurist) tem distribuicoes de features
diferentes do treino (HUG), apesar da harmonizacao retrospectiva Juchler. Se sim,
explica por que AUC ext same-conv (0.59) < AUC stress CMHA (0.64).

Metricas por feature:
- Kolmogorov-Smirnov D + p
- Cohen's d (tamanho de efeito)
- Quantis 25/50/75 por subcoorte

Restringe a aneurismas NAO-ROMPIDOS para remover efeito ruptura do shift
(alternativamente: controles pareados, mas aqui usamos unruptured-only por simplicidade).
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
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if pooled == 0: return np.nan
    return (x.mean() - y.mean()) / pooled


def main():
    df = pd.read_csv(IN_CSV)
    mca = df[(df.cohort == "AneuX")
             & df.location.fillna("").str.contains("MCA")
             & df.ruptured.notna()].dropna(subset=FEATURES)

    # Agrupar: HUG (hug2016+hug2016snf) vs aneurisk vs aneurist
    mca = mca.assign(site=np.where(
        mca.subcohort.isin(["hug2016", "hug2016snf"]), "HUG",
        mca.subcohort,
    ))
    mca_unr = mca[mca.ruptured == 0]  # restringe a nao-rompidos
    print("Ns por site (nao-rompidos MCA):")
    print(mca_unr.groupby("site").size())
    print()

    hug = mca_unr[mca_unr.site == "HUG"]
    ak  = mca_unr[mca_unr.site == "aneurisk"]
    at  = mca_unr[mca_unr.site == "aneurist"]

    rows = []
    for f in FEATURES:
        # HUG vs aneurisk
        ks1 = ks_2samp(hug[f], ak[f])
        d1  = cohens_d(hug[f], ak[f])
        # HUG vs aneurist
        ks2 = ks_2samp(hug[f], at[f])
        d2  = cohens_d(hug[f], at[f])
        # HUG vs pseudo-external combinado (aneurisk + aneurist)
        pseudo = pd.concat([ak, at])
        ks3 = ks_2samp(hug[f], pseudo[f])
        d3  = cohens_d(hug[f], pseudo[f])

        rows.append({
            "feature": f,
            "HUG_med": hug[f].median(), "AK_med": ak[f].median(), "AT_med": at[f].median(),
            "KS_HUGvsAK": ks1.statistic, "p_HUGvsAK": ks1.pvalue, "d_HUGvsAK": d1,
            "KS_HUGvsAT": ks2.statistic, "p_HUGvsAT": ks2.pvalue, "d_HUGvsAT": d2,
            "KS_HUGvsPseudo": ks3.statistic, "p_HUGvsPseudo": ks3.pvalue, "d_HUGvsPseudo": d3,
        })

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "M4c_site_shift_audit.csv", index=False)

    # Imprimir tabela-resumo enxuta
    print("Site shift audit (nao-rompidos MCA, HUG como referencia):")
    print()
    print(f"{'Feature':6s}  {'HUG_med':>8s}  {'AK_med':>8s}  {'AT_med':>8s}  "
          f"{'KS_Pseudo':>9s}  {'p':>8s}  {'Cohen_d':>8s}")
    for _, r in res.iterrows():
        flag = "<<<" if (r['p_HUGvsPseudo'] < 0.05 and abs(r['d_HUGvsPseudo']) > 0.3) else ""
        print(f"{r['feature']:6s}  {r['HUG_med']:8.3f}  {r['AK_med']:8.3f}  "
              f"{r['AT_med']:8.3f}  {r['KS_HUGvsPseudo']:9.3f}  "
              f"{r['p_HUGvsPseudo']:8.4f}  {r['d_HUGvsPseudo']:+8.2f} {flag}")

    # Escore global
    signif = ((res['p_HUGvsPseudo'] < 0.05) & (res['d_HUGvsPseudo'].abs() > 0.3)).sum()
    print(f"\nFeatures com site shift significativo HUG vs pseudo-external: "
          f"{signif}/{len(res)}")
    if signif >= 4:
        print(">>> VEREDICTO: site shift intra-AneuX MATERIAL. Hipotese GPT suportada.")
    elif signif >= 1:
        print(">>> VEREDICTO: site shift intra-AneuX DETECTAVEL mas limitado.")
    else:
        print(">>> VEREDICTO: sem site shift detectavel. Hipotese GPT enfraquecida.")


if __name__ == "__main__":
    main()

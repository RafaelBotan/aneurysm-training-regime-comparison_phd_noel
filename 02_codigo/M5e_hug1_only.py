"""
M5e — HUG1-only sensitivity (Q2 GPT).

Verificar se extremos positivos vem de hug2016snf.
Se sim: mecanismo muda de "spectrum bias HUG" para "subcohort composition".

Compara:
- HUG1 (hug2016 apenas, sem snf) → pex, CMHA
- HUG (hug2016+snf) → pex, CMHA  [baseline ja existente]

Tambem: Cohen's d ruptured vs unruptured DENTRO de HUG1-only vs HUG+snf
para ver se extremidade persiste sem snf.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:/doutorado_noel")
IN_CSV = ROOT / "03_analises" / "M2_shared_feature_core.csv"
OUT_DIR = ROOT / "03_analises"

FEATURES_FULL = ["AR", "BF", "CP", "EI", "NSI", "UI", "Dmax", "Dn", "H", "S", "V"]
NPS = {"AR", "BF", "CP", "H", "Dn"}
NON_NPS = [f for f in FEATURES_FULL if f not in NPS]


def cohens_d(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return np.nan
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if pooled == 0: return np.nan
    return (x.mean() - y.mean()) / pooled


def load_all():
    df = pd.read_csv(IN_CSV)
    hug1 = df[(df.cohort == "AneuX")
              & (df.subcohort == "hug2016")
              & df.location.fillna("").str.contains("MCA")
              & df.ruptured.notna()].dropna(subset=FEATURES_FULL).reset_index(drop=True)
    snf = df[(df.cohort == "AneuX")
             & (df.subcohort == "hug2016snf")
             & df.location.fillna("").str.contains("MCA")
             & df.ruptured.notna()].dropna(subset=FEATURES_FULL).reset_index(drop=True)
    hug_all = df[(df.cohort == "AneuX")
                 & df.subcohort.isin(["hug2016", "hug2016snf"])
                 & df.location.fillna("").str.contains("MCA")
                 & df.ruptured.notna()].dropna(subset=FEATURES_FULL).reset_index(drop=True)
    pex = df[(df.cohort == "AneuX")
             & df.subcohort.isin(["aneurisk", "aneurist"])
             & df.location.fillna("").str.contains("MCA")
             & df.ruptured.notna()].dropna(subset=FEATURES_FULL).reset_index(drop=True)
    cmha = df[(df.cohort == "CMHA") & df.ruptured.notna()]\
              .dropna(subset=FEATURES_FULL).reset_index(drop=True)
    return hug1, snf, hug_all, pex, cmha


def eval_regime(train_df, test_dict, feats):
    Xt, yt = train_df[feats].values.astype(float), train_df["ruptured"].astype(int).values
    sc = StandardScaler().fit(Xt)
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                             max_iter=5000, class_weight="balanced").fit(sc.transform(Xt), yt)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_cv = cross_val_predict(
        LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                           max_iter=5000, class_weight="balanced"),
        sc.transform(Xt), yt, cv=cv, method="predict_proba",
    )[:, 1]
    res = {"auc_cv5": roc_auc_score(yt, p_cv), "n": len(yt), "r": int(yt.sum())}
    for name, tdf in test_dict.items():
        Xe = tdf[feats].values.astype(float)
        ye = tdf["ruptured"].astype(int).values
        pe = clf.predict_proba(sc.transform(Xe))[:, 1]
        res[f"auc_{name}"] = roc_auc_score(ye, pe)
    return res


def main():
    hug1, snf, hug_all, pex, cmha = load_all()

    print("=" * 70)
    print("M5e — HUG1-only sensitivity")
    print("=" * 70)
    print(f"hug2016 MCA:    n={len(hug1)}, R={int(hug1.ruptured.sum())}")
    print(f"hug2016snf MCA: n={len(snf)}, R={int(snf.ruptured.sum())}")
    print(f"hug_all MCA:    n={len(hug_all)}, R={int(hug_all.ruptured.sum())}")
    print()

    # Cohen's d ruptured vs unruptured por subcoorte
    print("Cohen's d (ruptured - unruptured) por subcoorte:")
    print(f"{'Feature':6s}  {'d_hug2016':>10s}  {'d_snf':>10s}  {'d_hug_all':>10s}")
    for f in FEATURES_FULL:
        d1 = cohens_d(hug1[hug1.ruptured == 1][f], hug1[hug1.ruptured == 0][f])
        ds = cohens_d(snf[snf.ruptured == 1][f], snf[snf.ruptured == 0][f])
        da = cohens_d(hug_all[hug_all.ruptured == 1][f], hug_all[hug_all.ruptured == 0][f])
        print(f"{f:6s}  {d1:+10.2f}  {ds:+10.2f}  {da:+10.2f}")
    print()

    # AUCs por regime
    print("AUCs (ablacao C = sem NPS, 6 features):")
    r1 = eval_regime(hug1, {"pex": pex, "cmha": cmha}, NON_NPS)
    ra = eval_regime(hug_all, {"pex": pex, "cmha": cmha}, NON_NPS)
    rp = eval_regime(pex, {"hug1": hug1, "hug_all": hug_all, "cmha": cmha}, NON_NPS)
    print(f"  HUG1-only (n={r1['n']}, R={r1['r']}): "
          f"CV5={r1['auc_cv5']:.3f}  pex={r1['auc_pex']:.3f}  CMHA={r1['auc_cmha']:.3f}")
    print(f"  HUG+snf   (n={ra['n']}, R={ra['r']}): "
          f"CV5={ra['auc_cv5']:.3f}  pex={ra['auc_pex']:.3f}  CMHA={ra['auc_cmha']:.3f}")
    print(f"  pex ref   (n={rp['n']}, R={rp['r']}): "
          f"CV5={rp['auc_cv5']:.3f}  hug1={rp['auc_hug1']:.3f}  hug_all={rp['auc_hug_all']:.3f}  "
          f"CMHA={rp['auc_cmha']:.3f}")
    print()

    print("AUCs (ablacao A = full 11 features):")
    r1a = eval_regime(hug1, {"pex": pex, "cmha": cmha}, FEATURES_FULL)
    raa = eval_regime(hug_all, {"pex": pex, "cmha": cmha}, FEATURES_FULL)
    print(f"  HUG1-only: CV5={r1a['auc_cv5']:.3f}  pex={r1a['auc_pex']:.3f}  CMHA={r1a['auc_cmha']:.3f}")
    print(f"  HUG+snf:   CV5={raa['auc_cv5']:.3f}  pex={raa['auc_pex']:.3f}  CMHA={raa['auc_cmha']:.3f}")
    print()

    # Veredicto
    delta_cv = r1['auc_cv5'] - ra['auc_cv5']
    delta_cmha = r1['auc_cmha'] - ra['auc_cmha']
    print("=== VEREDICTO ===")
    print(f"Delta CV5 (hug1_only - hug_all): {delta_cv:+.3f}")
    print(f"Delta CMHA (hug1_only - hug_all): {delta_cmha:+.3f}")
    if abs(delta_cmha) < 0.03:
        print(">>> Remover snf nao muda transportabilidade. Spectrum bias NAO vem de snf.")
    elif delta_cmha > 0.03:
        print(">>> HUG1-only transporta MELHOR. snf contribui para spectrum bias.")
    else:
        print(">>> HUG1-only transporta PIOR. snf dilui extremos (inesperado).")


if __name__ == "__main__":
    main()

"""
M5c — Sensitivities.

1. Excluir pacientes mistos R+UR (38 no AneuX com ruptured+unruptured no mesmo paciente).
   Retestar ablacao C, ambos regimes.
2. Pooled HUG+pex como terceiro regime de treino.
   Se pooling "estabiliza" a classe positiva e melhora transport, reforca tese.
"""
from __future__ import annotations
import sys, re
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

_pid_re = re.compile(r"^(p\d+)_")


def get_patient_id(row):
    pid = row.get("patient_id", None)
    if isinstance(pid, str) and len(pid) > 0:
        return pid
    aid = str(row.get("aneurysm_id", ""))
    m = _pid_re.match(aid)
    if m:
        return m.group(1)
    return aid


def load_all():
    df = pd.read_csv(IN_CSV)
    df["pid"] = df.apply(get_patient_id, axis=1)
    hug = df[(df.cohort == "AneuX")
             & df.subcohort.isin(["hug2016", "hug2016snf"])
             & df.location.fillna("").str.contains("MCA")
             & df.ruptured.notna()].dropna(subset=FEATURES_FULL).reset_index(drop=True)
    pex = df[(df.cohort == "AneuX")
             & df.subcohort.isin(["aneurisk", "aneurist"])
             & df.location.fillna("").str.contains("MCA")
             & df.ruptured.notna()].dropna(subset=FEATURES_FULL).reset_index(drop=True)
    cmha = df[(df.cohort == "CMHA") & df.ruptured.notna()]\
              .dropna(subset=FEATURES_FULL).reset_index(drop=True)
    return hug, pex, cmha


def train_eval(train_df, test_dfs, feats, label):
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
    results = {"regime": label, "feats": len(feats),
               "n_train": len(yt), "r_train": int(yt.sum()),
               "auc_cv5": roc_auc_score(yt, p_cv)}
    for name, tdf in test_dfs.items():
        Xe = tdf[feats].values.astype(float)
        ye = tdf["ruptured"].astype(int).values
        pe = clf.predict_proba(sc.transform(Xe))[:, 1]
        results[f"auc_{name}"] = roc_auc_score(ye, pe)
        results[f"n_{name}"] = len(ye)
    return results


def main():
    hug, pex, cmha = load_all()

    # ── Sensitivity 1: excluir pacientes mistos R+UR ──
    print("=" * 70)
    print("SENSITIVITY 1: Excluir pacientes mistos R+UR (AneuX)")
    print("=" * 70)

    aneux_mca = pd.concat([hug, pex], ignore_index=True)
    mixed_pids = set()
    for pid, grp in aneux_mca.groupby("pid"):
        vals = grp.ruptured.unique()
        if len(vals) > 1 and 0 in vals and 1 in vals:
            mixed_pids.add(pid)
    print(f"Pacientes mistos R+UR em AneuX MCA: {len(mixed_pids)}")

    hug_clean = hug[~hug.pid.isin(mixed_pids)].reset_index(drop=True)
    pex_clean = pex[~pex.pid.isin(mixed_pids)].reset_index(drop=True)
    print(f"HUG MCA: {len(hug)} -> {len(hug_clean)} (removidos {len(hug)-len(hug_clean)})")
    print(f"pex MCA: {len(pex)} -> {len(pex_clean)} (removidos {len(pex)-len(pex_clean)})")
    print(f"HUG R: {int(hug.ruptured.sum())} -> {int(hug_clean.ruptured.sum())}")
    print(f"pex R: {int(pex.ruptured.sum())} -> {int(pex_clean.ruptured.sum())}")
    print()

    rows_s1 = []
    for feats, abl in [(NON_NPS, "C_sem_NPS"), (FEATURES_FULL, "A_full_11")]:
        r1 = train_eval(hug_clean, {"pex": pex_clean, "cmha": cmha}, feats, f"HUG_clean_{abl}")
        r2 = train_eval(pex_clean, {"hug": hug_clean, "cmha": cmha}, feats, f"pex_clean_{abl}")
        rows_s1.extend([r1, r2])
        print(f"--- {abl} ---")
        print(f"  HUG-clean CV5={r1['auc_cv5']:.3f}  pex={r1['auc_pex']:.3f}  cmha={r1['auc_cmha']:.3f}")
        print(f"  pex-clean CV5={r2['auc_cv5']:.3f}  hug={r2['auc_hug']:.3f}  cmha={r2['auc_cmha']:.3f}")
    print()

    # Referencia baseline (com mistos)
    print("Referencia baseline (COM mistos, ablacao C):")
    ref_hug = train_eval(hug, {"pex": pex, "cmha": cmha}, NON_NPS, "HUG_baseline_C")
    ref_pex = train_eval(pex, {"hug": hug, "cmha": cmha}, NON_NPS, "pex_baseline_C")
    print(f"  HUG CV5={ref_hug['auc_cv5']:.3f} pex={ref_hug['auc_pex']:.3f} cmha={ref_hug['auc_cmha']:.3f}")
    print(f"  pex CV5={ref_pex['auc_cv5']:.3f} hug={ref_pex['auc_hug']:.3f} cmha={ref_pex['auc_cmha']:.3f}")
    print()

    # ── Sensitivity 2: Pooled HUG+pex como terceiro regime ──
    print("=" * 70)
    print("SENSITIVITY 2: Pooled HUG+pex como terceiro regime de treino")
    print("=" * 70)

    pooled = pd.concat([hug, pex], ignore_index=True)
    print(f"Pooled: n={len(pooled)}, R={int(pooled.ruptured.sum())}")
    print()

    rows_s2 = []
    for feats, abl in [(NON_NPS, "C_sem_NPS"), (FEATURES_FULL, "A_full_11")]:
        r = train_eval(pooled, {"cmha": cmha}, feats, f"pooled_{abl}")
        rows_s2.append(r)
        print(f"--- {abl} ---")
        print(f"  Pooled CV5={r['auc_cv5']:.3f}  cmha={r['auc_cmha']:.3f}")
    print()

    # Comparar pooled vs pex-only vs hug-only em CMHA (ablacao C)
    print("Comparacao -> CMHA (ablacao C):")
    print(f"  HUG-only:  {ref_hug['auc_cmha']:.3f}")
    print(f"  pex-only:  {ref_pex['auc_cmha']:.3f}")
    print(f"  Pooled:    {rows_s2[0]['auc_cmha']:.3f}")
    print()

    delta_pool_vs_pex = rows_s2[0]['auc_cmha'] - ref_pex['auc_cmha']
    delta_pool_vs_hug = rows_s2[0]['auc_cmha'] - ref_hug['auc_cmha']
    print(f"Delta pooled - pex -> CMHA: {delta_pool_vs_pex:+.3f}")
    print(f"Delta pooled - hug -> CMHA: {delta_pool_vs_hug:+.3f}")

    if rows_s2[0]['auc_cmha'] > ref_pex['auc_cmha'] + 0.02:
        print(">>> Pooling MELHORA sobre pex-only. Estabilizacao de positivos funciona.")
    elif abs(rows_s2[0]['auc_cmha'] - ref_pex['auc_cmha']) < 0.02:
        print(">>> Pooling equivalente a pex-only. N extra nao ajuda nem atrapalha.")
    else:
        print(">>> Pooling PIORA vs pex-only. Extremos HUG 'contaminam' o pooled.")

    all_rows = rows_s1 + rows_s2
    pd.DataFrame(all_rows).to_csv(OUT_DIR / "M5c_sensitivity.csv", index=False)
    print(f"\n=> {OUT_DIR / 'M5c_sensitivity.csv'}")


if __name__ == "__main__":
    main()

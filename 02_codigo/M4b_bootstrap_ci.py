"""
M4b — Bootstrap 95% CI nas 9 AUCs do cascade M4.

Testar se 0.59 (ext same-conv) vs 0.64 (stress CMHA) e estatisticamente distinguivel.
Bootstrap stratificado (mantem razao ruptos/total) com 2000 resamples. IC percentil.

Tambem: bootstrap da diferenca AUC_stress - AUC_ext_sameConv para ver se CI exclui 0
(argumento central para "non-monotonicity e real, nao ruido").
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:/doutorado_noel")
IN_CSV = ROOT / "03_analises" / "M2_shared_feature_core.csv"
OUT_DIR = ROOT / "03_analises"

FEATURES_FULL = ["AR", "BF", "CP", "EI", "NSI", "UI", "Dmax", "Dn", "H", "S", "V"]
NPS = {"AR", "BF", "CP", "H", "Dn"}
ABLATIONS = {
    "A_full_11":              FEATURES_FULL,
    "B_sem_CP":               [f for f in FEATURES_FULL if f != "CP"],
    "C_sem_neck_plane_sens":  [f for f in FEATURES_FULL if f not in NPS],
}

RNG = np.random.default_rng(42)
N_BOOT = 2000


def load_all():
    df = pd.read_csv(IN_CSV)
    tr = df[(df.cohort == "AneuX")
            & df.subcohort.isin(["hug2016", "hug2016snf"])
            & df.location.fillna("").str.contains("MCA")
            & df.ruptured.notna()].dropna(subset=FEATURES_FULL).reset_index(drop=True)
    ext = df[(df.cohort == "AneuX")
             & df.subcohort.isin(["aneurisk", "aneurist"])
             & df.location.fillna("").str.contains("MCA")
             & df.ruptured.notna()].dropna(subset=FEATURES_FULL).reset_index(drop=True)
    cmha = df[(df.cohort == "CMHA") & df.ruptured.notna()]\
              .dropna(subset=FEATURES_FULL).reset_index(drop=True)
    return tr, ext, cmha


def stratified_bootstrap_auc(y, p, rng=RNG, n_boot=N_BOOT):
    y = np.asarray(y); p = np.asarray(p)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    aucs = np.empty(n_boot)
    for i in range(n_boot):
        bp = rng.choice(pos, len(pos), replace=True)
        bn = rng.choice(neg, len(neg), replace=True)
        idx = np.concatenate([bp, bn])
        aucs[i] = roc_auc_score(y[idx], p[idx])
    return aucs


def predict_layers(train, ext, cmha, feats):
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    Xtr, ytr = train[feats].values.astype(float), train["ruptured"].astype(int).values
    Xex, yex = ext[feats].values.astype(float), ext["ruptured"].astype(int).values
    Xch, ych = cmha[feats].values.astype(float), cmha["ruptured"].astype(int).values
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xex_s, Xch_s = sc.transform(Xtr), sc.transform(Xex), sc.transform(Xch)

    clf_fit = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                 max_iter=5000, class_weight="balanced").fit(Xtr_s, ytr)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_int = cross_val_predict(
        LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                           max_iter=5000, class_weight="balanced"),
        Xtr_s, ytr, cv=cv, method="predict_proba",
    )[:, 1]
    p_ex = clf_fit.predict_proba(Xex_s)[:, 1]
    p_ch = clf_fit.predict_proba(Xch_s)[:, 1]
    return (ytr, p_int), (yex, p_ex), (ych, p_ch)


def main():
    train, ext, cmha = load_all()
    rows = []
    for abl, feats in ABLATIONS.items():
        (y1, p1), (y2, p2), (y3, p3) = predict_layers(train, ext, cmha, feats)
        a1 = stratified_bootstrap_auc(y1, p1)
        a2 = stratified_bootstrap_auc(y2, p2)
        a3 = stratified_bootstrap_auc(y3, p3)
        diff_3_2 = a3 - a2  # queremos que CI exclua 0 pra direita

        def summary(name, y, p, a):
            return {
                "ablation": abl, "layer": name,
                "n": len(y), "rupt": int(y.sum()),
                "AUC_point": roc_auc_score(y, p),
                "AUC_boot_mean": float(a.mean()),
                "CI_lo": float(np.quantile(a, 0.025)),
                "CI_hi": float(np.quantile(a, 0.975)),
            }
        rows.append(summary("1_internal_HUG_MCA_CV5", y1, p1, a1))
        rows.append(summary("2_external_AneuX_MCA_same_conv", y2, p2, a2))
        rows.append(summary("3_stress_CMHA_MCA_diff_conv", y3, p3, a3))

        delta_mean = float(diff_3_2.mean())
        delta_lo = float(np.quantile(diff_3_2, 0.025))
        delta_hi = float(np.quantile(diff_3_2, 0.975))
        p_negative = float((diff_3_2 < 0).mean())
        print(f"\n=== Ablacao {abl} ===")
        for r in rows[-3:]:
            print(f"  {r['layer']:35s}  "
                  f"AUC={r['AUC_point']:.3f}  [{r['CI_lo']:.3f}, {r['CI_hi']:.3f}]")
        print(f"  Delta AUC (stress - ext_same_conv) = {delta_mean:+.3f} "
              f"[{delta_lo:+.3f}, {delta_hi:+.3f}]  P(delta<0)={p_negative:.3f}")
        print(f"  Non-monotonicity real? {'SIM' if delta_lo > 0 else 'NAO (CI inclui 0)'}")

    df_ci = pd.DataFrame(rows)
    df_ci.to_csv(OUT_DIR / "M4b_bootstrap_ci.csv", index=False)
    print(f"\n=> {OUT_DIR / 'M4b_bootstrap_ci.csv'}")


if __name__ == "__main__":
    main()

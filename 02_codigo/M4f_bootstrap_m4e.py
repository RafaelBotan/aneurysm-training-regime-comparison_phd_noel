"""
M4f — Bootstrap 95% CI sobre M4e (pex-trained).

Testar se as AUCs do reverse-train (0.66 CV, 0.67/0.75 HUG, 0.59/0.70 CMHA) tem
CI estreitos o bastante para fundamentar tese.

Bootstrap estratificado 2000 resamples. Inclui ablacao A (full-11) e C (sem NPS).
Tambem imprime CIs dos deltas vs baseline HUG-trained para ver se diferencas
[pex-C_cmha 0.697 vs hug-C_cmha 0.653] tem CI que exclui zero.
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
ABLATIONS = {
    "A_full_11":             FEATURES_FULL,
    "C_sem_neck_plane_sens": [f for f in FEATURES_FULL if f not in NPS],
}
RNG = np.random.default_rng(42)
N_BOOT = 2000


def load_all():
    df = pd.read_csv(IN_CSV)
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


def strat_boot_auc(y, p, n_boot=N_BOOT, rng=RNG):
    y = np.asarray(y); p = np.asarray(p)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    out = np.empty(n_boot)
    for i in range(n_boot):
        bp = rng.choice(pos, len(pos), replace=True)
        bn = rng.choice(neg, len(neg), replace=True)
        idx = np.concatenate([bp, bn])
        out[i] = roc_auc_score(y[idx], p[idx])
    return out


def run_pex_trained(pex, hug, cmha, feats):
    Xp, yp = pex[feats].values.astype(float), pex["ruptured"].astype(int).values
    Xh, yh = hug[feats].values.astype(float), hug["ruptured"].astype(int).values
    Xc, yc = cmha[feats].values.astype(float), cmha["ruptured"].astype(int).values
    sc = StandardScaler().fit(Xp)
    Xp_s, Xh_s, Xc_s = sc.transform(Xp), sc.transform(Xh), sc.transform(Xc)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_int = cross_val_predict(
        LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                           max_iter=5000, class_weight="balanced"),
        Xp_s, yp, cv=cv, method="predict_proba",
    )[:, 1]
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                             max_iter=5000, class_weight="balanced").fit(Xp_s, yp)
    p_h = clf.predict_proba(Xh_s)[:, 1]
    p_c = clf.predict_proba(Xc_s)[:, 1]
    return (yp, p_int), (yh, p_h), (yc, p_c)


def run_hug_trained(hug, pex, cmha, feats):
    Xh, yh = hug[feats].values.astype(float), hug["ruptured"].astype(int).values
    Xp, yp = pex[feats].values.astype(float), pex["ruptured"].astype(int).values
    Xc, yc = cmha[feats].values.astype(float), cmha["ruptured"].astype(int).values
    sc = StandardScaler().fit(Xh)
    Xh_s, Xp_s, Xc_s = sc.transform(Xh), sc.transform(Xp), sc.transform(Xc)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_int = cross_val_predict(
        LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                           max_iter=5000, class_weight="balanced"),
        Xh_s, yh, cv=cv, method="predict_proba",
    )[:, 1]
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                             max_iter=5000, class_weight="balanced").fit(Xh_s, yh)
    p_p = clf.predict_proba(Xp_s)[:, 1]
    p_c = clf.predict_proba(Xc_s)[:, 1]
    return (yh, p_int), (yp, p_p), (yc, p_c)


def summarize(label, y, p, boot):
    return {
        "split": label, "n": len(y), "rupt": int(np.sum(y)),
        "AUC": roc_auc_score(y, p),
        "mean_boot": float(boot.mean()),
        "CI_lo": float(np.quantile(boot, 0.025)),
        "CI_hi": float(np.quantile(boot, 0.975)),
    }


def main():
    hug, pex, cmha = load_all()
    rows = []

    for abl, feats in ABLATIONS.items():
        print(f"\n=== Ablacao {abl} ({len(feats)} feats) ===")
        # pex-trained
        (y1, p1), (y2, p2), (y3, p3) = run_pex_trained(pex, hug, cmha, feats)
        b1 = strat_boot_auc(y1, p1)
        b2 = strat_boot_auc(y2, p2)
        b3 = strat_boot_auc(y3, p3)
        for label, y, p, b in [
            (f"pex_CV5__{abl}", y1, p1, b1),
            (f"pex->HUG__{abl}", y2, p2, b2),
            (f"pex->CMHA__{abl}", y3, p3, b3),
        ]:
            r = summarize(label, y, p, b)
            rows.append(r)
            print(f"  {label:28s} AUC={r['AUC']:.3f}  CI=[{r['CI_lo']:.3f}, {r['CI_hi']:.3f}]")

        # hug-trained (comparacao)
        (y4, p4), (y5, p5), (y6, p6) = run_hug_trained(hug, pex, cmha, feats)
        b4 = strat_boot_auc(y4, p4)
        b5 = strat_boot_auc(y5, p5)
        b6 = strat_boot_auc(y6, p6)
        for label, y, p, b in [
            (f"hug_CV5__{abl}", y4, p4, b4),
            (f"hug->pex__{abl}", y5, p5, b5),
            (f"hug->CMHA__{abl}", y6, p6, b6),
        ]:
            r = summarize(label, y, p, b)
            rows.append(r)
            print(f"  {label:28s} AUC={r['AUC']:.3f}  CI=[{r['CI_lo']:.3f}, {r['CI_hi']:.3f}]")

        # Deltas chave: pex-trained vs hug-trained no mesmo teste
        # Precisamos bootstrap conjunto para diferenca; fazemos aqui com mesmos ndx
        # Para HUG test: pex->HUG (b2) vs hug_CV5 (b4, mesmo yh)
        # Para CMHA test: pex->CMHA (b3) vs hug->CMHA (b6, mesmo yc)
        delta_cmha = b3 - b6
        delta_cmha_mean = float(delta_cmha.mean())
        delta_cmha_lo = float(np.quantile(delta_cmha, 0.025))
        delta_cmha_hi = float(np.quantile(delta_cmha, 0.975))
        print(f"  Delta CMHA (pex-trained - hug-trained) = {delta_cmha_mean:+.3f} "
              f"[{delta_cmha_lo:+.3f}, {delta_cmha_hi:+.3f}]")
        excl = "SIM" if delta_cmha_lo > 0 or delta_cmha_hi < 0 else "NAO"
        print(f"  CI exclui zero? {excl}")

        # Assimetria: pex->HUG vs HUG->pex
        delta_asym = b2 - b5
        d_mean = float(delta_asym.mean())
        d_lo = float(np.quantile(delta_asym, 0.025))
        d_hi = float(np.quantile(delta_asym, 0.975))
        print(f"  Assimetria (pex->HUG - hug->pex) = {d_mean:+.3f} "
              f"[{d_lo:+.3f}, {d_hi:+.3f}]")
        excl = "SIM" if d_lo > 0 or d_hi < 0 else "NAO"
        print(f"  CI exclui zero? {excl}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "M4f_bootstrap_m4e.csv", index=False)
    print(f"\n=> {OUT_DIR / 'M4f_bootstrap_m4e.csv'}")


if __name__ == "__main__":
    main()

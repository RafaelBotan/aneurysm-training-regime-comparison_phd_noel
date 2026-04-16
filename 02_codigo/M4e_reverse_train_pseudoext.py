"""
M4e — Teste decisivo GPT: treinar em pseudo-external (30R, fenotipo aparentemente mais
tipico) e testar em HUG + CMHA.

Ataque direto a hipotese "training positive-class spectrum bias":
- Se pseudo-ext -> CMHA melhora vs HUG -> CMHA (0.64) = C-novo ganha forca
- Se pseudo-ext -> HUG razoavel = melhor ainda
- Se nao melhora nada = C-novo enfraquece

Ablacoes: full-11, sem CP, sem neck-plane-sensitive (AR BF CP H Dn).
Modo: train-scaled (fit scaler em pseudo-ext, aplicar sem refit).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss

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


def calib(y, p):
    from sklearn.linear_model import LogisticRegression as LR
    lo = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(lo / (1 - lo)).reshape(-1, 1)
    lr = LR().fit(logit, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0]), float(brier_score_loss(y, p))


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


def eval_abl(hug, pex, cmha, feats):
    Xp, yp = pex[feats].values.astype(float), pex["ruptured"].astype(int).values
    Xh, yh = hug[feats].values.astype(float), hug["ruptured"].astype(int).values
    Xc, yc = cmha[feats].values.astype(float), cmha["ruptured"].astype(int).values

    sc = StandardScaler().fit(Xp)
    Xp_s, Xh_s, Xc_s = sc.transform(Xp), sc.transform(Xh), sc.transform(Xc)

    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                             max_iter=5000, class_weight="balanced").fit(Xp_s, yp)

    # CV interna em pseudo-ext
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_int = cross_val_predict(
        LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                           max_iter=5000, class_weight="balanced"),
        Xp_s, yp, cv=cv, method="predict_proba",
    )[:, 1]

    p_hug = clf.predict_proba(Xh_s)[:, 1]
    p_cmha = clf.predict_proba(Xc_s)[:, 1]

    auc_int = roc_auc_score(yp, p_int)
    auc_hug = roc_auc_score(yh, p_hug)
    auc_cmha = roc_auc_score(yc, p_cmha)

    sl_int, ic_int, br_int = calib(yp, p_int)
    sl_hug, ic_hug, br_hug = calib(yh, p_hug)
    sl_cmha, ic_cmha, br_cmha = calib(yc, p_cmha)

    return {
        "auc_int_pex_CV5": auc_int, "auc_ext_HUG": auc_hug, "auc_stress_CMHA": auc_cmha,
        "slope_int": sl_int, "slope_HUG": sl_hug, "slope_CMHA": sl_cmha,
        "int_int": ic_int, "int_HUG": ic_hug, "int_CMHA": ic_cmha,
        "brier_int": br_int, "brier_HUG": br_hug, "brier_CMHA": br_cmha,
    }


def main():
    hug, pex, cmha = load_all()
    print(f"Ns: pex={len(pex)} (R={pex.ruptured.sum():.0f})  "
          f"hug={len(hug)} (R={hug.ruptured.sum():.0f})  "
          f"cmha={len(cmha)} (R={cmha.ruptured.sum():.0f})")
    print()
    print("REVERSE TRAIN: treino em pseudo-external (aneurisk+aneurist MCA, 30R)")
    print("Teste em HUG (14R) e CMHA (77R).")
    print("Comparacao com HUG-trained (M4 original): HUG-CV 0.886, ext_same_conv 0.588, CMHA 0.640")
    print()

    rows = []
    for abl, feats in ABLATIONS.items():
        r = eval_abl(hug, pex, cmha, feats)
        r["ablation"] = abl
        rows.append(r)
        print(f"=== {abl} ({len(feats)} feats) ===")
        print(f"  AUC pex-CV5 = {r['auc_int_pex_CV5']:.3f}  "
              f"slope={r['slope_int']:+.2f} int={r['int_int']:+.2f} brier={r['brier_int']:.3f}")
        print(f"  AUC -> HUG  = {r['auc_ext_HUG']:.3f}  "
              f"slope={r['slope_HUG']:+.2f} int={r['int_HUG']:+.2f} brier={r['brier_HUG']:.3f}")
        print(f"  AUC -> CMHA = {r['auc_stress_CMHA']:.3f}  "
              f"slope={r['slope_CMHA']:+.2f} int={r['int_CMHA']:+.2f} brier={r['brier_CMHA']:.3f}")
        print()

    pd.DataFrame(rows).to_csv(OUT_DIR / "M4e_reverse_train.csv", index=False)

    # Comparacao direta
    print("=== VEREDICTO ===")
    r_full = rows[0]
    hug_cv_ref = 0.886
    hug_ext_ref = 0.588
    cmha_ref = 0.640
    print(f"Baseline HUG-trained (full-11): CV={hug_cv_ref:.3f} ext={hug_ext_ref:.3f} stress={cmha_ref:.3f}")
    print(f"Novo    pex-trained  (full-11): CV={r_full['auc_int_pex_CV5']:.3f} "
          f"hug={r_full['auc_ext_HUG']:.3f} cmha={r_full['auc_stress_CMHA']:.3f}")
    print()

    delta_cmha = r_full['auc_stress_CMHA'] - cmha_ref
    delta_hug = r_full['auc_ext_HUG'] - hug_ext_ref  # compara com ext_same_conv reverso
    print(f"Delta CMHA (pex-trained - hug-trained) = {delta_cmha:+.3f}")
    print(f"Delta cross (pex-hug - hug-pex)        = {delta_hug:+.3f}")
    print()

    if r_full['auc_stress_CMHA'] > 0.70:
        print(">>> pex -> CMHA MELHOROU substancialmente. Spectrum bias HUG CONFIRMADO.")
    elif r_full['auc_stress_CMHA'] > cmha_ref + 0.03:
        print(">>> pex -> CMHA melhorou marginalmente. Spectrum bias PARCIAL.")
    elif abs(r_full['auc_stress_CMHA'] - cmha_ref) < 0.03:
        print(">>> pex -> CMHA equivalente. Spectrum bias NAO confirmado.")
    else:
        print(">>> pex -> CMHA piorou. Spectrum bias REFUTADO.")

    if r_full['auc_ext_HUG'] < 0.60:
        print(">>> pex -> HUG tambem falha. Consistente com fenotipo rompido HUG idiossincratico.")
    elif r_full['auc_ext_HUG'] > 0.75:
        print(">>> pex -> HUG bom. Rompidos HUG separaveis por modelo externo — enfraquece spectrum bias.")
    else:
        print(">>> pex -> HUG medio.")


if __name__ == "__main__":
    main()

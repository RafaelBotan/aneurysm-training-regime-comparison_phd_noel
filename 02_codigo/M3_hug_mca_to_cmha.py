"""
M3 — Baseline transportability: AneuX HUG MCA -> CMHA MCA
==========================================================
Regressao logistica sobre o Shared Feature Core (11 features Raghavan/Ma + Ujiie/Dhar).

Desenho (corrigido pos-/double-check GPT, ver memoria M2):
  - Primario: features escalonadas com scaler fitado SO em AneuX HUG MCA,
    aplicado sem refit ao CMHA (train-scaled, nao per-cohort).
  - Sensibilidade: per-cohort z-score (apaga o shift de media/variancia — roda mesmo
    assim para comparacao, mas rotulado como "scale-normalized").
  - Ablacao obrigatoria de robustez (Q4 GPT):
      A. full-11: AR, BF, CP, EI, NSI, UI, Dmax, Dn, H, S, V
      B. sem-CP: dropa CP (feature com inversao direcional)
      C. sem-neck-plane-sensitive: dropa AR, BF, CP, H, Dn
         (features cujo valor depende materialmente do posicionamento do neck plane;
         AneuX usa corte planar manual, CMHA usa vmtk4aneurysm Voronoi automatico)

Treino: AneuX subcohort in {hug2016, hug2016snf} AND location contem "MCA"
Teste : CMHA (todos os aneurismas MCA, N=105)

Metricas primarias: AUROC, calibration slope, calibration intercept, Brier, ICI.
Classe rara em treino: ~14 rupturados em ~121 total -> usar class_weight='balanced'.

Saida: 03_analises/M3_results.csv + M3_predictions.csv + M3_summary.md
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:/doutorado_noel")
IN_CSV = ROOT / "03_analises" / "M2_shared_feature_core.csv"
OUT_DIR = ROOT / "03_analises"

FEATURES_FULL = ["AR", "BF", "CP", "EI", "NSI", "UI", "Dmax", "Dn", "H", "S", "V"]
FEATURES_NO_CP = [f for f in FEATURES_FULL if f != "CP"]
NECK_PLANE_SENSITIVE = {"AR", "BF", "CP", "H", "Dn"}
FEATURES_NO_NPS = [f for f in FEATURES_FULL if f not in NECK_PLANE_SENSITIVE]

ABLATIONS = {
    "A_full_11":               FEATURES_FULL,
    "B_sem_CP":                FEATURES_NO_CP,
    "C_sem_neck_plane_sens":   FEATURES_NO_NPS,
}

# -------------------------------------------------------------------
# Load + split
# -------------------------------------------------------------------
def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(IN_CSV)

    # Train: AneuX HUG + MCA
    mask_train = (
        (df["cohort"] == "AneuX")
        & (df["subcohort"].isin(["hug2016", "hug2016snf"]))
        & (df["location"].fillna("").str.contains("MCA"))
        & (df["ruptured"].notna())
    )
    train = df[mask_train].copy().reset_index(drop=True)

    # Test: CMHA MCA — todos os aneurismas (44 controles excluidos por ruptured NaN)
    mask_test = (df["cohort"] == "CMHA") & df["ruptured"].notna()
    test = df[mask_test].copy().reset_index(drop=True)

    # Drop linhas sem features
    train = train.dropna(subset=FEATURES_FULL).reset_index(drop=True)
    test  = test.dropna(subset=FEATURES_FULL).reset_index(drop=True)

    return train, test


# -------------------------------------------------------------------
# Calibracao
# -------------------------------------------------------------------
def calibration_slope_intercept(y_true: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Slope e intercept da regressao logistica de y sobre logit(p)."""
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    X = logit.reshape(-1, 1)
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    clf.fit(X, y_true)
    slope = float(clf.coef_[0, 0])
    intercept = float(clf.intercept_[0])
    return slope, intercept


def ici(y_true: np.ndarray, p: np.ndarray) -> float:
    """Integrated calibration index (loess seria ideal; aproximamos com isotonic)."""
    from sklearn.isotonic import IsotonicRegression
    order = np.argsort(p)
    iso = IsotonicRegression(out_of_bounds="clip")
    p_cal = iso.fit_transform(p[order], y_true[order])
    # desordena de volta para alinhar
    inv = np.argsort(order)
    p_cal_aligned = p_cal[inv]
    return float(np.mean(np.abs(p_cal_aligned - p)))


# -------------------------------------------------------------------
# Fit + evaluate
# -------------------------------------------------------------------
def evaluate(train: pd.DataFrame, test: pd.DataFrame,
             feats: list[str], mode: str) -> dict:
    """
    mode: "train_scaled"    -> StandardScaler fit em train, transform test
          "per_cohort"      -> StandardScaler fit separado em cada coorte (sensibilidade)
          "raw"             -> sem normalizacao
    """
    Xtr = train[feats].values.astype(float)
    ytr = train["ruptured"].astype(int).values
    Xte = test[feats].values.astype(float)
    yte = test["ruptured"].astype(int).values

    if mode == "train_scaled":
        sc = StandardScaler().fit(Xtr)
        Xtr_s = sc.transform(Xtr)
        Xte_s = sc.transform(Xte)
    elif mode == "per_cohort":
        sc_tr = StandardScaler().fit(Xtr); Xtr_s = sc_tr.transform(Xtr)
        sc_te = StandardScaler().fit(Xte); Xte_s = sc_te.transform(Xte)
    elif mode == "raw":
        Xtr_s, Xte_s = Xtr, Xte
    else:
        raise ValueError(mode)

    # L2-penalized logistic (class_weight balanced por raridade de ruptura em HUG)
    clf = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs",
        max_iter=5000, class_weight="balanced",
    )
    clf.fit(Xtr_s, ytr)

    # Internal validation (AneuX HUG MCA): 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_tr_cv = cross_val_predict(
        LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                           max_iter=5000, class_weight="balanced"),
        Xtr_s, ytr, cv=cv, method="predict_proba",
    )[:, 1]
    auc_internal = roc_auc_score(ytr, p_tr_cv)

    # External: CMHA
    p_te = clf.predict_proba(Xte_s)[:, 1]
    auc_ext = roc_auc_score(yte, p_te)
    brier_ext = brier_score_loss(yte, p_te)
    slope_ext, intercept_ext = calibration_slope_intercept(yte, p_te)
    ici_ext = ici(yte, p_te)

    return {
        "mode": mode,
        "n_feat": len(feats),
        "n_train": len(ytr),
        "n_test": len(yte),
        "rupt_train": int(ytr.sum()),
        "rupt_test": int(yte.sum()),
        "AUC_internal_cv5_aneux": auc_internal,
        "AUC_external_cmha": auc_ext,
        "Brier_external": brier_ext,
        "calib_slope_external": slope_ext,
        "calib_intercept_external": intercept_ext,
        "ICI_external": ici_ext,
        "p_test": p_te,
        "y_test": yte,
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    train, test = load_splits()
    print(f"Treino: AneuX HUG MCA -> N={len(train)}, rupt={int(train['ruptured'].sum())}")
    print(f"Teste : CMHA MCA       -> N={len(test)}, rupt={int(test['ruptured'].sum())}")
    print(f"Prev treino: {train['ruptured'].mean():.3f} | prev teste: {test['ruptured'].mean():.3f}")
    print()

    rows = []
    pred_blocks = []
    for abl_name, feats in ABLATIONS.items():
        print(f"--- Ablacao {abl_name} (n_feat={len(feats)}) ---")
        for mode in ("train_scaled", "per_cohort", "raw"):
            r = evaluate(train, test, feats, mode)
            r_print = {k: v for k, v in r.items() if k not in ("p_test", "y_test")}
            r_print["ablation"] = abl_name
            rows.append({k: v for k, v in r_print.items()})
            print(f"  {mode:13s}  AUC_cv={r['AUC_internal_cv5_aneux']:.3f}  "
                  f"AUC_ext={r['AUC_external_cmha']:.3f}  "
                  f"slope={r['calib_slope_external']:.2f}  "
                  f"int={r['calib_intercept_external']:.2f}  "
                  f"Brier={r['Brier_external']:.3f}  "
                  f"ICI={r['ICI_external']:.3f}")
            pred_blocks.append(pd.DataFrame({
                "ablation": abl_name, "mode": mode,
                "p_test": r["p_test"], "y_test": r["y_test"],
            }))
        print()

    res = pd.DataFrame(rows)
    cols = ["ablation", "mode", "n_feat", "n_train", "n_test",
            "rupt_train", "rupt_test",
            "AUC_internal_cv5_aneux", "AUC_external_cmha",
            "Brier_external", "calib_slope_external",
            "calib_intercept_external", "ICI_external"]
    res = res[cols]
    res.to_csv(OUT_DIR / "M3_results.csv", index=False)
    pd.concat(pred_blocks, ignore_index=True).to_csv(
        OUT_DIR / "M3_predictions.csv", index=False
    )

    # Resumo markdown
    md_lines = [
        "# M3 — HUG MCA -> CMHA MCA (baseline logistic)",
        f"Treino: AneuX HUG MCA, N={len(train)}, ruptos={int(train['ruptured'].sum())}",
        f"Teste : CMHA MCA,     N={len(test)}, ruptos={int(test['ruptured'].sum())}",
        "",
        "## Resultados",
        "",
        "| Ablacao | Mode | AUC int (CV5) | AUC ext CMHA | Brier | Slope | Int | ICI |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in res.iterrows():
        md_lines.append(
            f"| {r['ablation']} | {r['mode']} | "
            f"{r['AUC_internal_cv5_aneux']:.3f} | {r['AUC_external_cmha']:.3f} | "
            f"{r['Brier_external']:.3f} | {r['calib_slope_external']:.2f} | "
            f"{r['calib_intercept_external']:.2f} | {r['ICI_external']:.3f} |"
        )
    (OUT_DIR / "M3_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"=> {OUT_DIR / 'M3_results.csv'}")
    print(f"=> {OUT_DIR / 'M3_predictions.csv'}")
    print(f"=> {OUT_DIR / 'M3_summary.md'}")


if __name__ == "__main__":
    main()

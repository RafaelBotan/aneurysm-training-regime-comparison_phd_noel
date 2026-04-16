"""
M4 — Cascade de validacao: interno -> external otimista -> stress test.

Treino fixo: AneuX HUG MCA (N=118, 14R, 12% prev)
  -> Cascade 1 (internal CV5): AUC interno
  -> Cascade 2 (external otimista, mesma convencao Juchler planar manual):
     AneuX aneurisk+aneurist MCA (N=70, 30R, 43% prev)
  -> Cascade 3 (stress test out-of-domain, convencao diferente vmtk Voronoi):
     CMHA MCA (N=105, 77R, 73% prev)

Esperado: AUC_interno ~0.87, AUC_external ~0.70-0.78 (queda por site shift + prev shift,
mas convencao preservada), AUC_stress ~0.64 (queda adicional por convencao + case mix).

Isso isola a contribuicao:
  Site shift   = AUC_int - AUC_ext_same_convention
  Procedure+biology drift = AUC_ext_same_conv - AUC_stress

Saida:
  03_analises/M4_cascade_results.csv
  03_analises/M4_fig_cascade_roc.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

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


def calibration_slope_intercept(y, p):
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    clf.fit(logit.reshape(-1, 1), y)
    return float(clf.coef_[0, 0]), float(clf.intercept_[0])


def ici(y, p):
    from sklearn.isotonic import IsotonicRegression
    order = np.argsort(p)
    iso = IsotonicRegression(out_of_bounds="clip")
    p_cal = iso.fit_transform(p[order], y[order])
    inv = np.argsort(order)
    return float(np.mean(np.abs(p_cal[inv] - p)))


def evaluate_cascade(train, ext, cmha, feats: list[str]):
    Xtr = train[feats].values.astype(float)
    ytr = train["ruptured"].astype(int).values
    Xex = ext[feats].values.astype(float)
    yex = ext["ruptured"].astype(int).values
    Xch = cmha[feats].values.astype(float)
    ych = cmha["ruptured"].astype(int).values

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xex_s, Xch_s = sc.transform(Xtr), sc.transform(Xex), sc.transform(Xch)

    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                             max_iter=5000, class_weight="balanced")
    clf.fit(Xtr_s, ytr)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_int = cross_val_predict(
        LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                           max_iter=5000, class_weight="balanced"),
        Xtr_s, ytr, cv=cv, method="predict_proba",
    )[:, 1]
    p_ex = clf.predict_proba(Xex_s)[:, 1]
    p_ch = clf.predict_proba(Xch_s)[:, 1]

    def pack(name, y, p, prev):
        slope, inter = calibration_slope_intercept(y, p)
        return {
            "layer": name, "n": len(y), "rupt": int(y.sum()),
            "prev": prev, "AUC": roc_auc_score(y, p),
            "Brier": brier_score_loss(y, p),
            "calib_slope": slope, "calib_intercept": inter,
            "ICI": ici(y, p), "_y": y, "_p": p,
        }

    return [
        pack("1_internal_HUG_MCA_CV5", ytr, p_int, ytr.mean()),
        pack("2_external_AneuX_MCA_same_conv", yex, p_ex, yex.mean()),
        pack("3_stress_CMHA_MCA_diff_conv", ych, p_ch, ych.mean()),
    ]


def main():
    train, ext, cmha = load_all()
    print(f"Treino HUG MCA: N={len(train)}, R={int(train.ruptured.sum())} "
          f"({train.ruptured.mean():.2%})")
    print(f"External AneuX MCA: N={len(ext)}, R={int(ext.ruptured.sum())} "
          f"({ext.ruptured.mean():.2%})")
    print(f"Stress CMHA MCA: N={len(cmha)}, R={int(cmha.ruptured.sum())} "
          f"({cmha.ruptured.mean():.2%})")
    print()

    results = []
    all_layers = {}  # p,y por (ablation, layer) para plots

    for abl, feats in ABLATIONS.items():
        print(f"--- {abl} (n_feat={len(feats)}) ---")
        layers = evaluate_cascade(train, ext, cmha, feats)
        for L in layers:
            print(f"  {L['layer']:35s}  AUC={L['AUC']:.3f}  "
                  f"slope={L['calib_slope']:.2f}  int={L['calib_intercept']:.2f}  "
                  f"Brier={L['Brier']:.3f}  ICI={L['ICI']:.3f}")
            all_layers[(abl, L["layer"])] = (L["_y"], L["_p"])
            results.append({
                "ablation": abl, **{k: v for k, v in L.items()
                                    if k not in ("_y", "_p")},
            })
        print()

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT_DIR / "M4_cascade_results.csv", index=False)
    print(f"=> {OUT_DIR / 'M4_cascade_results.csv'}")

    # Figura: cascade ROC por ablacao (3 paineis, 3 curvas cada)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors = {"1_internal_HUG_MCA_CV5":           "#1f77b4",
              "2_external_AneuX_MCA_same_conv":   "#ff7f0e",
              "3_stress_CMHA_MCA_diff_conv":      "#d62728"}
    labels = {"1_internal_HUG_MCA_CV5":           "1. Internal HUG MCA (CV5)",
              "2_external_AneuX_MCA_same_conv":   "2. External AneuX MCA (same conv.)",
              "3_stress_CMHA_MCA_diff_conv":      "3. Stress CMHA MCA (diff. conv.)"}
    titles = {"A_full_11":              "A) full 11 features",
              "B_sem_CP":               "B) sem CP",
              "C_sem_neck_plane_sens":  "C) sem AR,BF,CP,H,Dn"}
    for ax, abl in zip(axs, ABLATIONS.keys()):
        for layer in colors:
            y, p = all_layers[(abl, layer)]
            fpr, tpr, _ = roc_curve(y, p)
            auc = roc_auc_score(y, p)
            ax.plot(fpr, tpr, lw=2.2, color=colors[layer],
                    label=f"{labels[layer]}  AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], ls="--", color="grey", alpha=0.6)
        ax.set_xlabel("1 - Specificity (FPR)")
        ax.set_title(titles[abl])
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Sensitivity (TPR)")
    fig.suptitle("Cascade ROC: internal -> same-convention external -> "
                 "different-convention stress", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "M4_fig_cascade_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"=> {OUT_DIR / 'M4_fig_cascade_roc.png'}")


if __name__ == "__main__":
    main()

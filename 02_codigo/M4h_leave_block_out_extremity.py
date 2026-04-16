"""
M4h — Extremidade out-of-model (Q1 GPT).

Modelo (ablacao C) usa 6 features non-NPS: EI, NSI, UI, Dmax, S, V.
Features EXCLUIDAS do modelo: NPS = AR, BF, CP, H, Dn.

Recomputa correlacao predicao x extremidade onde:
  - extrem_outOOM = mean(|z|) DOS NPS contra media+std dos nao-rompidos CMHA
  - predicao = HUG-trained ou pex-trained na ablacao C (so non-NPS)

Sem sobreposicao entre features do modelo e features de extremidade.
Se HUG-trained ainda correlaciona alto com extrem_outOOM e pex-trained nao, blinda
contra circularidade (Q1).

Tambem testa Mahalanobis multivariada como segunda metrica de extremidade.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import mahalanobis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:/doutorado_noel")
IN_CSV = ROOT / "03_analises" / "M2_shared_feature_core.csv"

FEATURES_FULL = ["AR", "BF", "CP", "EI", "NSI", "UI", "Dmax", "Dn", "H", "S", "V"]
NPS = ["AR", "BF", "CP", "H", "Dn"]                          # excluidas do modelo C
NON_NPS = [f for f in FEATURES_FULL if f not in NPS]         # usadas no modelo C


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


def fit_predict(train_df, test_df, feats):
    Xt, yt = train_df[feats].values.astype(float), train_df["ruptured"].astype(int).values
    Xe = test_df[feats].values.astype(float)
    sc = StandardScaler().fit(Xt)
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                             max_iter=5000, class_weight="balanced")\
        .fit(sc.transform(Xt), yt)
    return clf.predict_proba(sc.transform(Xe))[:, 1]


def main():
    hug, pex, cmha = load_all()

    # Predicoes em CMHA com modelos non-NPS (ablacao C)
    cmha = cmha.copy()
    cmha["p_hug_C"] = fit_predict(hug, cmha, NON_NPS)
    cmha["p_pex_C"] = fit_predict(pex, cmha, NON_NPS)

    # Extremidade out-of-model: mean(|z|) DOS NPS apenas
    unr = cmha[cmha.ruptured == 0]
    mu_nps = unr[NPS].mean()
    sd_nps = unr[NPS].std(ddof=1).replace(0, 1)

    z_nps = (cmha[NPS] - mu_nps) / sd_nps
    cmha["extrem_outOOM_uniNPS"] = z_nps.abs().mean(axis=1)

    # Mahalanobis nos NPS (multivariada, captura correlacao)
    cov = np.cov(unr[NPS].values, rowvar=False)
    inv_cov = np.linalg.pinv(cov)
    mu_vec = mu_nps.values
    cmha["maha_NPS"] = cmha[NPS].apply(
        lambda r: mahalanobis(r.values, mu_vec, inv_cov), axis=1
    )

    # Para referencia: extremidade IN-model (NON_NPS)
    mu_nn = unr[NON_NPS].mean()
    sd_nn = unr[NON_NPS].std(ddof=1).replace(0, 1)
    z_nn = (cmha[NON_NPS] - mu_nn) / sd_nn
    cmha["extrem_inModel_uniNON_NPS"] = z_nn.abs().mean(axis=1)

    rom = cmha[cmha.ruptured == 1]

    print("=" * 76)
    print("M4h — Extremidade OUT-OF-MODEL (Q1 GPT, blindagem contra circularidade)")
    print("=" * 76)
    print()
    print("Modelo C usa: EI, NSI, UI, Dmax, S, V (6 non-NPS)")
    print("Extremidade  : AR, BF, CP, H, Dn (5 NPS, EXCLUIDAS do modelo)")
    print()
    print("Spearman rho entre p_predicao e extremidade (entre rompidos CMHA, n=77):")
    print()
    print(f"{'Predicao':14s} {'vs extrem_uniNPS':>22s} {'vs maha_NPS':>20s} "
          f"{'vs extrem_inModel':>22s}")
    for col_p, name in [("p_hug_C", "HUG-trained"), ("p_pex_C", "pex-trained")]:
        rho1, p1 = spearmanr(rom[col_p], rom["extrem_outOOM_uniNPS"])
        rho2, p2 = spearmanr(rom[col_p], rom["maha_NPS"])
        rho3, p3 = spearmanr(rom[col_p], rom["extrem_inModel_uniNON_NPS"])
        print(f"{name:14s}  rho={rho1:+.3f} (p={p1:.4f})  "
              f"rho={rho2:+.3f} (p={p2:.4f})  "
              f"rho={rho3:+.3f} (p={p3:.4f})")
    print()

    # Veredicto
    rho_hug_oom, p_hug_oom = spearmanr(rom["p_hug_C"], rom["extrem_outOOM_uniNPS"])
    rho_pex_oom, p_pex_oom = spearmanr(rom["p_pex_C"], rom["extrem_outOOM_uniNPS"])
    rho_hug_mh, p_hug_mh = spearmanr(rom["p_hug_C"], rom["maha_NPS"])
    rho_pex_mh, p_pex_mh = spearmanr(rom["p_pex_C"], rom["maha_NPS"])

    print("=== VEREDICTO M4h ===")
    print()
    print(f"HUG-trained vs extrem_outOOM (uniNPS): rho={rho_hug_oom:+.3f} (p={p_hug_oom:.4f})")
    print(f"pex-trained vs extrem_outOOM (uniNPS): rho={rho_pex_oom:+.3f} (p={p_pex_oom:.4f})")
    print(f"HUG-trained vs Mahalanobis NPS       : rho={rho_hug_mh:+.3f} (p={p_hug_mh:.4f})")
    print(f"pex-trained vs Mahalanobis NPS       : rho={rho_pex_mh:+.3f} (p={p_pex_mh:.4f})")
    print()
    if rho_hug_oom > 0.25 and abs(rho_pex_oom) < 0.20:
        print(">>> Diferenca robusta: HUG correlaciona com extremos NPS mesmo SEM usar")
        print(">>> NPS no modelo. Circularidade trivial REFUTADA. Mecanismo")
        print(">>> mechanistically consistent.")
    elif rho_hug_oom > 0.25:
        print(">>> HUG correlaciona; pex tambem nao-zero. Mecanismo parcial.")
    else:
        print(">>> Correlacao desaparece sem features NPS. M4g original era circular.")

    cmha[["aneurysm_id", "ruptured", "p_hug_C", "p_pex_C",
          "extrem_outOOM_uniNPS", "maha_NPS", "extrem_inModel_uniNON_NPS"]]\
        .to_csv(ROOT / "03_analises" / "M4h_extremity_outOOM.csv", index=False)


if __name__ == "__main__":
    main()

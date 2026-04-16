"""
M3b — GroupKFold por patient_id para checar leakage.

AneuX HUG MCA tem 77 pacientes para 118 aneurismas (14 pacientes com 2-4
aneurismas cada; 34 aneurismas em pacientes multi). StratifiedKFold pode
super-estimar internal AUC se aneurismas do mesmo paciente caem em folds diferentes.

Comparacao: StratifiedKFold (5) vs GroupKFold (5, groups=patient_id).
Mesmo classifier, mesmas features, mesmo class_weight. So muda o split.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:/doutorado_noel")
IN_CSV = ROOT / "03_analises" / "M2_shared_feature_core.csv"

FEATURES = ["AR", "BF", "CP", "EI", "NSI", "UI", "Dmax", "Dn", "H", "S", "V"]


def main() -> None:
    df = pd.read_csv(IN_CSV)
    mask = (
        (df["cohort"] == "AneuX")
        & (df["subcohort"].isin(["hug2016", "hug2016snf"]))
        & (df["location"].fillna("").str.contains("MCA"))
        & (df["ruptured"].notna())
    )
    train = df[mask].copy().dropna(subset=FEATURES).reset_index(drop=True)

    X = train[FEATURES].values.astype(float)
    y = train["ruptured"].astype(int).values

    # patient_id tem 21 NaN em HUG MCA (17.8%) — clinical.csv deixou em branco.
    # Mas vesselFileID comeca com "p<num>_" (p126, p383, etc) que e o ID do paciente.
    # Fallback deterministico: extrair prefixo p<num>_ do aneurysm_id quando
    # patient_id estiver ausente.
    import re as _re
    def _fallback(row):
        pid = row["patient_id"]
        if isinstance(pid, str) and len(pid) > 0:
            return pid
        aid = str(row["aneurysm_id"])
        m = _re.match(r"^(p\d+)_", aid)
        if m:
            return m.group(1)
        # fallback final: aneurysm_id unico (sem leakage possivel)
        return aid
    train["_grp"] = train.apply(_fallback, axis=1)
    groups = train["_grp"].values

    n_pat = len(np.unique(groups))
    print(f"N aneurismas = {len(y)}, ruptos = {y.sum()}, pacientes unicos = {n_pat}")
    print(f"Aneurismas em pacientes multi = "
          f"{(pd.Series(groups).value_counts()>1).sum()} pacientes, "
          f"{(pd.Series(groups).value_counts().loc[lambda s: s>1]).sum()} aneurismas")
    print()

    # scaler ANTES do CV porque queremos replicar condicoes do M3 primario
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)

    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                             max_iter=5000, class_weight="balanced")

    # Stratified
    rng = 42
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng)
    p_skf = cross_val_predict(clf, Xs, y, cv=skf, method="predict_proba")[:, 1]
    auc_skf = roc_auc_score(y, p_skf)

    # GroupKFold (nao tem shuffle, mas podemos permutar grupos)
    gkf = GroupKFold(n_splits=5)
    p_gkf = cross_val_predict(clf, Xs, y, cv=gkf, method="predict_proba",
                              groups=groups)[:, 1]
    auc_gkf = roc_auc_score(y, p_gkf)

    print(f"AUC StratifiedKFold(5) = {auc_skf:.3f}")
    print(f"AUC GroupKFold(5)      = {auc_gkf:.3f}")
    print(f"Delta (SKF - GKF)      = {auc_skf - auc_gkf:+.3f}")
    print()

    if abs(auc_skf - auc_gkf) < 0.02:
        print(">>> Leakage negligivel. Numeros M3 internos OK.")
    elif auc_skf > auc_gkf:
        print(">>> Leakage positivo: SKF infla AUC. Reportar GKF como primario.")
    else:
        print(">>> Anti-leakage (GKF>SKF): provavelmente pacientes multi tem ruptura "
              "consistente; grupos ajudam generalizacao. Reportar ambos.")


if __name__ == "__main__":
    main()

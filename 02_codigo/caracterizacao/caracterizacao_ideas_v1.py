"""
Caracterizacao descritiva IDEAS v1.
Gera os numeros citados em 03_analises/caracterizacao_ideas/caracterizacao_v1.md.

Pre-requisito: datasets baixados em 01_datasets/N5_IDEAS/figshare/
Uso: python 02_codigo/caracterizacao/caracterizacao_ideas_v1.py
"""
from pathlib import Path
import os
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[2]
FIGSHARE = BASE / "01_datasets" / "N5_IDEAS" / "figshare"
OUT = BASE / "03_analises" / "caracterizacao_ideas"
OUT.mkdir(parents=True, exist_ok=True)


# Midpoints das faixas etarias binned (dataset IDEAS anonimiza em bins de 5 anos)
# "Over 40" nao tem teto definido; usamos 45 como aproximacao — declarar como limitacao
ONSET_MID = {
    "Less than 1": 0.5, "1 to 2": 1.5, "3 to 4": 3.5, "5 to 7": 6.0,
    "8 to 10": 9.0, "11 to 14": 12.5, "15 to 19": 17.0, "20 to 24": 22.0,
    "25 to 29": 27.0, "30 to 34": 32.0, "35 to 39": 37.0, "Over 40": 45.0,
}
SURG_MID = {
    "15 to 20": 17.5, "20 to 24": 22.0, "25 to 29": 27.0, "30 to 34": 32.0,
    "35 to 39": 37.0, "40 to 44": 42.0, "45 to 49": 47.0, "50 to 54": 52.0,
    "55 to 59": 57.0, "60 to 64": 62.0, "65 to 70": 67.5,
}


def load_metadata():
    df = pd.read_csv(FIGSHARE / "tables_metadata" / "Metadata_Release_Anon.csv")
    # Remove linha duplicata conhecida (ID 209 aparece duas vezes, com NaN em ILAE_Year5 em ambas)
    # Comparacao direta com == trata NaN!=NaN; usar .equals ou comparar com fillna
    dupes = df[df["ID"].duplicated(keep=False)]
    if len(dupes) == 2:
        r0, r1 = dupes.iloc[0], dupes.iloc[1]
        same = all(
            (pd.isna(r0[c]) and pd.isna(r1[c])) or r0[c] == r1[c]
            for c in df.columns
        )
        assert same, "Duplicate ID 209 rows differ in non-NaN field — investigate."
    else:
        raise AssertionError(
            f"Expected 2 duplicate rows (ID 209), got {len(dupes)}. Re-verify dataset."
        )
    df_dedup = df.drop_duplicates(subset="ID").reset_index(drop=True)
    return df, df_dedup


def report_counts_consistency(df_dedup):
    all_ids = set(df_dedup["ID"].astype(str))
    masks_base = FIGSHARE / "masks" / "masks_extracted" / "masks"
    mask_ids = {x for x in os.listdir(masks_base) if x.isdigit()}
    fs = pd.read_csv(FIGSHARE / "tables_stats_freesurfer" / "aparc_thick.txt",
                     sep="\t", nrows=0)
    fs_ids = {c for c in fs.columns if c.isdigit()}

    return {
        "n_metadata_unique": len(all_ids),
        "n_masks": len(mask_ids),
        "n_freesurfer": len(fs_ids),
        "ids_without_mask": sorted(int(x) for x in (all_ids - mask_ids)),
        "ids_without_freesurfer": sorted(int(x) for x in (all_ids - fs_ids)),
    }


def report_missingness(df_dedup):
    miss = df_dedup.isna().sum().sort_values(ascending=False)
    return {c: (int(n), round(100 * n / len(df_dedup), 1))
            for c, n in miss.items() if n > 0}


def report_outcome_distribution(df_dedup):
    out = {}
    for year in range(1, 6):
        col = f"ILAE_Year{year}"
        n_with = df_dedup[col].notna().sum()
        n_free = (df_dedup[col] == 1).sum()
        out[f"Year{year}"] = {
            "n_with_data": int(n_with),
            "pct_with_data": round(100 * n_with / len(df_dedup), 1),
            "n_ilae1_seizure_free": int(n_free),
            "pct_free_among_with_data": round(100 * n_free / n_with, 1) if n_with else None,
        }
    return out


def report_pathology_by_outcome(df_dedup):
    df_dedup = df_dedup.copy()
    df_dedup["primary"] = np.where(df_dedup["ILAE_Year1"].isna(), np.nan,
                                    (df_dedup["ILAE_Year1"] == 1).astype(float))
    grp = df_dedup.groupby("Pathology")["primary"].agg(["count", "sum", "mean"])
    grp.columns = ["n_with_data", "n_seizure_free", "rate"]
    grp["rate"] = grp["rate"].round(3)
    return grp.sort_values("rate", ascending=False)


def report_epilepsy_duration(df_dedup):
    """Duracao aproximada = midpoint(Age_at_Surgery) - midpoint(Onset_Age).
    Noise floor ~5 anos em cada lado (propagado ~7 anos). Usar so para descritivo.
    """
    df = df_dedup.copy()
    df["onset_mid"] = df["Binned_Onset_Age"].map(ONSET_MID)
    df["surg_mid"] = df["Binned_Age_at_Surgery"].map(SURG_MID)
    df["duration_yr"] = df["surg_mid"] - df["onset_mid"]
    # Durations negativas sao artefato de bins largos (e.g., onset "15-19" + surgery "15-20")
    n_neg = int((df["duration_yr"] < 0).sum())
    n_valid = int(df["duration_yr"].notna().sum())
    summary = {
        "n_valid": n_valid,
        "n_negative_artifact": n_neg,
        "mean_yr": round(df["duration_yr"].mean(), 1),
        "median_yr": round(df["duration_yr"].median(), 1),
        "q25": round(df["duration_yr"].quantile(0.25), 1),
        "q75": round(df["duration_yr"].quantile(0.75), 1),
        "min": round(df["duration_yr"].min(), 1),
        "max": round(df["duration_yr"].max(), 1),
    }
    by_path = df.groupby("Pathology")["duration_yr"].agg(["count", "mean", "median"]).round(1)
    return summary, by_path


def report_crosstab_pathology_side_type(df_dedup):
    """Caracterizacao cirurgica: patologia x lado x tipo de operacao."""
    ct = pd.crosstab(
        [df_dedup["Pathology"], df_dedup["Op_Side"]],
        df_dedup["Op_Type"],
        margins=True, margins_name="TOTAL",
    )
    return ct


def report_longitudinal_trajectories(df_dedup):
    """Categoriza trajetorias ILAE Year1->Year5 entre pacientes com dado em ambos extremos.
    - stayed_free: ILAE 1 em Year1 E Year5
    - relapsed: ILAE 1 em Year1 mas >=2 em Year5
    - improved: >=2 em Year1 mas 1 em Year5
    - stayed_not_free: >=2 em Year1 E Year5
    """
    df = df_dedup.copy()
    y1 = df["ILAE_Year1"]
    y5 = df["ILAE_Year5"]
    mask = y1.notna() & y5.notna()
    sub = df[mask].copy()
    sub["free_y1"] = sub["ILAE_Year1"] == 1
    sub["free_y5"] = sub["ILAE_Year5"] == 1
    labels = np.where(sub["free_y1"] & sub["free_y5"], "stayed_free",
              np.where(sub["free_y1"] & ~sub["free_y5"], "relapsed",
              np.where(~sub["free_y1"] & sub["free_y5"], "improved", "stayed_not_free")))
    sub["trajectory"] = labels
    counts = sub["trajectory"].value_counts().to_dict()
    total = int(mask.sum())
    pct = {k: round(100 * v / total, 1) for k, v in counts.items()}
    return {"n_with_both": total, "counts": counts, "pct": pct}


def report_thickness_by_pathology(df_dedup):
    """Espessura cortical media (Desikan-Killiany) por patologia.
    aparc_thick.txt tem 75 linhas: 68 parcelas DK + 2 MeanThickness por hemisferio
    + BrainSegVolNotVent + eTIV (summary rows em unidades diferentes, excluir).
    Usa as duas linhas _MeanThickness_thickness (media FS por hemisferio) e
    calcula media bi-hemisferica por paciente.
    """
    thick = pd.read_csv(
        FIGSHARE / "tables_stats_freesurfer" / "aparc_thick.txt", sep="\t"
    )
    label_col = thick.columns[0]
    mean_rows = thick[thick[label_col].str.contains("MeanThickness_thickness", na=False)]
    assert len(mean_rows) == 2, f"Expected 2 hemisphere means, got {len(mean_rows)}"
    id_cols = [c for c in thick.columns if c != label_col]
    bihemi_mean = mean_rows[id_cols].mean(axis=0)
    per_pt = pd.DataFrame(
        {"ID": [int(c) for c in id_cols], "mean_thick_mm": bihemi_mean.values}
    )
    merged = per_pt.merge(df_dedup[["ID", "Pathology"]], on="ID", how="inner")
    by_path = (
        merged.groupby("Pathology")["mean_thick_mm"]
        .agg(["count", "mean", "std"])
        .round(3)
    )
    return by_path


def main():
    df_raw, df_dedup = load_metadata()
    print(f"[INFO] Metadata rows: {len(df_raw)} -> unique IDs: {len(df_dedup)}")
    print("[INFO] ID dedup check: duplicate confirmed on ID 209, removed.")
    print()

    counts = report_counts_consistency(df_dedup)
    print("=== COUNTS CONSISTENCY ===")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print()

    print("=== MISSINGNESS ===")
    for c, (n, pct) in report_missingness(df_dedup).items():
        print(f"  {c}: {n} ({pct}%)")
    print()

    print("=== OUTCOME DISTRIBUTION (ILAE) ===")
    for y, d in report_outcome_distribution(df_dedup).items():
        print(f"  {y}: {d}")
    print()

    print("=== OUTCOME BY PATHOLOGY (Year1) ===")
    print(report_pathology_by_outcome(df_dedup))
    print()

    print("=== DURACAO EPILEPSIA (midpoints binned) ===")
    dur_sum, dur_by_path = report_epilepsy_duration(df_dedup)
    for k, v in dur_sum.items():
        print(f"  {k}: {v}")
    print()
    print("--- por patologia ---")
    print(dur_by_path)
    print()

    print("=== CROSSTAB Pathology x Op_Side x Op_Type ===")
    ct = report_crosstab_pathology_side_type(df_dedup)
    print(ct)
    print()

    print("=== TRAJETORIAS LONGITUDINAIS Year1 -> Year5 ===")
    traj = report_longitudinal_trajectories(df_dedup)
    print(f"  N com outcome em Year1 e Year5: {traj['n_with_both']}")
    for k in ["stayed_free", "relapsed", "improved", "stayed_not_free"]:
        n = traj["counts"].get(k, 0)
        pct = traj["pct"].get(k, 0)
        print(f"  {k}: {n} ({pct}%)")
    print()

    print("=== ESPESSURA CORTICAL MEDIA (DK 74 parcelas) por PATOLOGIA ===")
    print(report_thickness_by_pathology(df_dedup))
    print()

    # Persist dedup'd metadata for downstream use
    df_dedup.to_csv(OUT / "metadata_dedup.csv", index=False)
    print(f"[SAVED] {OUT / 'metadata_dedup.csv'}")


if __name__ == "__main__":
    main()

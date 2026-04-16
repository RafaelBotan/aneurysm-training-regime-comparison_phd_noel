"""
CMHA Via A2 — gera screenshots axiais dos 77 CTAs rupturados
para triagem radiologica rapida do K1 gate.

Output: Y:/doutorado_noel/00_briefing/via_A2_screenshots/<patient_id>.png
Cada PNG tem 6 slices axiais em window CT brain (W80 L40)
mostrando: base do cranio, cisternas, ventriculos laterais, convexidade alta.
"""
import os
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd

CMHA = Path("Y:/doutorado_noel/01_datasets/CMHA")
NIFTI_DIR = CMHA / "patients_extracted" / "patients"
OUT_DIR = Path("Y:/doutorado_noel/00_briefing/via_A2_screenshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLINICAL = CMHA / "statistical_results" / "statistical results" / "clinical_all.csv"

# CT brain window (para ver sangue subaracnoideo)
WL_BRAIN = 40
WW_BRAIN = 80
# CTA window (para ver clips/coils como metal)
WL_ANGIO = 150
WW_ANGIO = 600


def window(img, wl, ww):
    lo, hi = wl - ww / 2, wl + ww / 2
    out = np.clip(img, lo, hi)
    return (out - lo) / (hi - lo)


def make_sheet(nifti_path, out_png, patient_id, meta_line):
    img = nib.load(str(nifti_path))
    data = img.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
    nz = data.shape[2]
    # 6 slice positions: base(15%), basal cisterns(30%), ventriculos(45%, 55%), convexidade(70%, 85%)
    positions = [0.15, 0.30, 0.45, 0.55, 0.70, 0.85]
    slice_idxs = [int(p * nz) for p in positions]

    fig, axes = plt.subplots(2, 6, figsize=(24, 9))
    fig.suptitle(f"{patient_id} — {meta_line}", fontsize=14, y=0.98)

    for col, si in enumerate(slice_idxs):
        sl = data[:, :, si]
        sl = np.rot90(sl)
        # linha 0 = brain window (SAH, hematoma, hidrocefalia visivel)
        axes[0, col].imshow(window(sl, WL_BRAIN, WW_BRAIN), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"z={si}/{nz} brain (W{WW_BRAIN} L{WL_BRAIN})", fontsize=9)
        axes[0, col].axis("off")
        # linha 1 = CTA window (clips/coils como metal saturado)
        axes[1, col].imshow(window(sl, WL_ANGIO, WW_ANGIO), cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"z={si}/{nz} angio (W{WW_ANGIO} L{WL_ANGIO})", fontsize=9)
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main():
    clinical = pd.read_csv(CLINICAL, encoding="utf-8-sig", skiprows=[1])
    ruptured = clinical[clinical["Rupture"] == 1]
    print(f"[INFO] {len(ruptured)} pacientes rupturados para processar")

    done = 0
    skipped = 0
    for _, row in ruptured.iterrows():
        pid = row["number"]
        nifti = NIFTI_DIR / pid / f"cta_images_head_{pid}.nii.gz"
        if not nifti.exists():
            print(f"[SKIP] {pid} — NIfTI nao encontrado")
            skipped += 1
            continue
        out_png = OUT_DIR / f"{pid}.png"
        if out_png.exists():
            continue
        meta = f"loc={row['location']} shape={row['Shape']} age={row['Age']} sex={row['Gender']}"
        try:
            make_sheet(nifti, out_png, pid, meta)
            done += 1
            if done % 10 == 0:
                print(f"[PROGRESS] {done} processados")
        except Exception as e:
            print(f"[ERROR] {pid}: {e}")
            skipped += 1

    print(f"\n[OK] {done} screenshots gerados em {OUT_DIR}")
    print(f"[WARN] {skipped} pulados")


if __name__ == "__main__":
    main()

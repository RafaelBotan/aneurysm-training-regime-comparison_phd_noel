"""
Cohort flowchart (Fig. 1) for manuscript.
STROBE-style inclusion/exclusion diagram.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13, 10))
ax.set_xlim(0, 13)
ax.set_ylim(0, 11)
ax.axis('off')

def box(x, y, w, h, text, fc='#E8F0FE', ec='#1A73E8', fontsize=9, fontweight='normal'):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.05",
                          linewidth=1.4, edgecolor=ec, facecolor=fc)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight)

def arrow(x1, y1, x2, y2):
    ar = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle='->,head_width=5,head_length=8',
                         color='#333', linewidth=1.2,
                         connectionstyle="arc3,rad=0")
    ax.add_patch(ar)

def exclusion(x, y, w, h, text):
    box(x, y, w, h, text, fc='#FEEBE8', ec='#C5221F', fontsize=8)

# TITLE
ax.text(6.5, 10.55, 'Study cohorts: inclusion and exclusion',
        ha='center', va='center', fontsize=13, fontweight='bold')

# ===== LEFT BRANCH: AneuX -> HUG + pex =====
box(3.5, 9.6, 3.6, 0.8,
    'AneuX database\n(multicentre European repository)',
    fc='#E8F0FE', ec='#1A73E8', fontsize=9, fontweight='bold')

# AneuX -> MCA subsets (arrow DOWN)
arrow(3.5, 9.2, 3.5, 8.55)

# exclusion for AneuX -> MCA only (placed to the right of arrow)
exclusion(7.6, 8.85, 3.2, 0.8,
          'Excluded from AneuX:\nnon-MCA locations\n(ACA, ICA, ACoA, PCoA, BA, PICA)')
# line from AneuX arrow to exclusion
ax.plot([3.5, 6.0], [8.85, 8.85], color='#888', linestyle=':', linewidth=1)

# split into HUG and pex
box(2.0, 8.15, 3.0, 0.9,
    'HUG subcohorts\n(hug2016 + hug2016snf)\nMCA aneurysms',
    fc='#E3F2FD', ec='#0D47A1', fontsize=9)
box(5.0, 8.15, 3.0, 0.9,
    'Pseudo-external\n(AneuRisk + AneuRist)\nMCA aneurysms',
    fc='#E3F2FD', ec='#0D47A1', fontsize=9)

# arrows from AneuX down then split
arrow(3.0, 8.55, 2.3, 8.6)
arrow(4.0, 8.55, 4.7, 8.6)

# final HUG and pex counts
box(2.0, 6.3, 3.0, 1.15,
    'HUG training cohort\nn = 118 MCA aneurysms\n(78 patients)\n14 ruptured (11.9 %)',
    fc='#C8E6C9', ec='#1B5E20', fontsize=9, fontweight='bold')
box(5.0, 6.3, 3.0, 1.15,
    'Pseudo-external cohort\nn = 70 MCA aneurysms\n(69 patients)\n30 ruptured (42.9 %)',
    fc='#C8E6C9', ec='#1B5E20', fontsize=9, fontweight='bold')

arrow(2.0, 7.7, 2.0, 6.9)
arrow(5.0, 7.7, 5.0, 6.9)

# ===== RIGHT BRANCH: CMHA =====
box(10.8, 9.6, 3.6, 0.8,
    'CMHA dataset (Song et al. 2024)\n2nd Affiliated Hospital, Anhui',
    fc='#FFF8E1', ec='#F57C00', fontsize=9, fontweight='bold')

# CMHA -> MCA subset (arrow DOWN)
arrow(10.8, 9.2, 10.8, 8.55)

# exclusion for CMHA
exclusion(10.8, 8.15, 3.0, 0.7,
          'Excluded from CMHA:\nnon-MCA locations')
arrow(10.8, 7.8, 10.8, 7.0)

# final CMHA
box(10.8, 6.3, 3.2, 1.15,
    'Out-of-domain test cohort\nn = 105 MCA aneurysms\n(99 patients)\n77 ruptured (73.3 %)',
    fc='#FFCCBC', ec='#BF360C', fontsize=9, fontweight='bold')

# ===== Design row =====
box(3.5, 4.3, 6.5, 1.2,
    '2 × 2 factorial design\nTraining regime (HUG vs pseudo-external) × '
    'Feature set (11 vs 6 non-NPS features)\nL2-penalised logistic regression',
    fc='#F3E5F5', ec='#4A148C', fontsize=9)

arrow(2.0, 5.7, 2.8, 4.95)
arrow(5.0, 5.7, 4.2, 4.95)

# CMHA feeds into evaluation
box(10.8, 4.3, 3.2, 1.2,
    'External evaluation\n(discrimination +\ncalibration)',
    fc='#F3E5F5', ec='#4A148C', fontsize=9)
arrow(10.8, 5.7, 10.8, 4.95)

# arrow from factorial -> evaluation
arrow(6.85, 4.3, 9.15, 4.3)

# ===== Outcomes =====
box(6.5, 2.3, 11.5, 1.3,
    'Outcomes: Brier score (primary), calibration slope and intercept, AUC,\n'
    'regime × morphological-extremity interaction\n'
    'Five confirmatory tests pre-specified before execution (repository commit ace1aa0)',
    fc='#FFFFFF', ec='#555', fontsize=9)
arrow(6.5, 3.65, 6.5, 2.95)

# ===== Legend =====
ax.text(0.3, 0.6, 'MCA = middle cerebral artery; NPS = neck-plane-sensitive features.',
        ha='left', va='center', fontsize=8.5, style='italic', color='#555')

plt.tight_layout()
out_png = r'Y:\doutorado_noel\03_analises\figures\Fig1_flowchart.png'
out_pdf = r'Y:\doutorado_noel\03_analises\figures\Fig1_flowchart.pdf'
plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")

# Cohort numbers frozen — M1 data freeze (2026-04-15)

Numeros reais extraidos dos CSVs. Substitui memorias baseadas em leitura de paper.

## AneuX (N=750 aneurismas, 15 com status desconhecido)
Total com status: 735 aneurismas (261R + 474UR)

### Por hospital/coorte
| Cohort | Total | Ruptured | Unruptured | Rupture rate |
|---|---|---|---|---|
| Geneva (HUG1+HUG2) | 485 | 83 | 388 | 17% |
| External (Barcelona+Milan+Sheffield) | 265 | 178 | 86 | 67% |

(HUG1=350 primario, HUG2=135 com selection bias tratamento-dirigido — usar HUG1 default por memorial v3)

### MCA-only subset (locations: "MCA bif" + "MCA M1")
| Cohort | Total MCA | R | UR | Rupture rate |
|---|---|---|---|---|
| Geneva MCA | 126 | 16 | 110 | 12.7% |
| External MCA | 59 | 28 | 31 | 47.5% |

### Excluded from primary
- ICA cavernous (n=34) — excluded per memorial (intradural/extradural distinction)
- Status=NaN (n=15)

## CMHA (N=105 aneurismas de 99 pacientes)
- **Todos MCA** (M1 n=33, M2 n=64, junction M1-M2 n=7, M3 n=1)
- 77 rupturados + 28 nao-rupturados
- **Prevalencia ruptura: 73%**
- 6 pacientes com aneurismas concomitantes (1R + 1UR no mesmo CTA)
- +44 controles (nao-aneurisma) em arquivo separado — nao usados no modelo primario

## Gradiente de prevalencia de ruptura MCA (evidencia pro-tese)
- Geneva MCA: 12.7% ruptured
- AneuX external MCA: 47.5% ruptured
- CMHA MCA: 73% ruptured

**Interpretacao:** gradiente confirma dataset shift progressivo do cohort de desenvolvimento para stress-test out-of-domain. Suporta fortemente tese de "apparent transportability" — performance no harmonized external (MCA 47%) sobre-estimara transportabilidade real para CMHA MCA (73%).

## Feature dictionary (Gate 2 resolvido)
- morpho-per-cut.csv: 173 colunas
  - 3 id/meta (source, dataset, cutToShow)
  - ~170 features morfometricos (aneurisma + vaso parente) — matches Zenodo claim
- AneuX fornece features ja computados; CMHA fornece STL + Fluent CAS → features devem ser recomputados via VMTK ou codigo AneuX publico
- Subset uniformemente disponivel: depende de reimplementacao de pipeline AneuX em CMHA STLs

## Pendencias M1
- [x] Gate 1 HUG: resolvido (HUG1=350 default)
- [x] Gate 2 features: resolvido (173 cols identificadas)
- [ ] Gate 3 K1 timing: Via A metadata NEGATIVA, aguardando Via A2 radiologia visual de Noel ou Via B email autor
- [ ] cohort_map.json (patient_id → aneurysm_id → source → split)
- [ ] OSF pre-registration

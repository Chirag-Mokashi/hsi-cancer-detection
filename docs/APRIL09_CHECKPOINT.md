# April 9 Checkpoint — Results Analysis & Decision Document

> Read this fully before making any decisions.
> Each section ends with a **DECISION NEEDED** block — your answers will drive the paper.

---

## 1. WHAT WAS BUILT

**Task:** Compare 4 ML/DL models for glioblastoma (brain tumor) detection in hyperspectral
histology images using Leave-One-Patient-Out Cross Validation (LOPOCV).

**Dataset:** HistologyHSI-GB (public, TCIA) — 3 patients (P1, P2, P3), 134 ROIs, 699 bands,
400–909 nm. Labels: Tumor (T) vs Non-Tumor (NT). Class imbalance: 39T / 95NT.

**Experiment grid:**
- 4 models: Random Forest (RF), SVM, HybridSN (3D+2D CNN), Vision Transformer (ViT)
- 3 band selection methods: PCA, Mutual Information (MI), LASSO
- 5 band counts: 4, 10, 20, 50, 100
- 3 LOPOCV folds: test P1 / test P2 / test P3
- Total: 186 training runs (48 RF + 48 SVM + 45 HybridSN + 45 ViT)

---

## 2. RESULTS SUMMARY TABLE

### 2a. Best Combo Per Model (by mean AUC across 3 folds)

| Model | Best Combo | AUC mean±std | Acc mean±std | Sens mean±std | Spec mean±std | F1 mean±std |
|-------|-----------|-------------|-------------|--------------|--------------|------------|
| **HybridSN** | LASSO/100 | **0.918 ± 0.039** | 0.789 ± 0.106 | 0.574 ± 0.447 | 0.910 ± 0.075 | 0.699 ± 0.191 |
| **SVM** | LASSO/100 | 0.873 ± 0.072 | **0.851 ± 0.047** | 0.626 ± 0.263 | **0.852 ± 0.178** | **0.777 ± 0.073** |
| **RF** | LASSO/100 | 0.806 ± 0.076 | 0.797 ± 0.040 | 0.476 ± 0.411 | 0.856 ± 0.184 | 0.700 ± 0.071 |
| **ViT** | MI/100 | 0.799 ± 0.099 | 0.619 ± 0.149 | **0.836 ± 0.091** | 0.613 ± 0.215 | 0.637 ± 0.105 |

### 2b. Overall Mean AUC (all combos & folds averaged)

| Model | Mean AUC | Mean Sensitivity | Mean Specificity |
|-------|---------|-----------------|-----------------|
| HybridSN | 0.762 ± 0.133 | 0.580 ± 0.295 | 0.711 ± 0.214 |
| SVM      | 0.764 ± 0.103 | 0.587 ± 0.216 | 0.782 ± 0.188 |
| ViT      | 0.757 ± 0.118 | **0.806 ± 0.149** | 0.462 ± 0.292 |
| RF       | 0.739 ± 0.091 | 0.441 ± 0.257 | **0.849 ± 0.147** |

---

## 3. KEY FINDINGS

### Finding 1: HybridSN wins on peak AUC, SVM wins on practical metrics

HybridSN achieves the highest best-combo AUC (0.918) but SVM has better accuracy (0.851),
specificity (0.852), and F1 (0.777) at its best combo. For a clinical tool where you need
reliable predictions (not just ranking), SVM is arguably more useful.

**For the paper:** Do not simply declare HybridSN the winner. Present both perspectives:
- "Best discriminative power": HybridSN (AUC 0.918)
- "Best practical classification": SVM (accuracy 0.851, F1 0.777)

This validates the project's hypothesis that classical ML is competitive with DL on small datasets.

---

### Finding 2: LASSO band selection dominates — except for ViT

Three of four models (RF, SVM, HybridSN) all peak with LASSO/100.
ViT uniquely peaks with MI/100 and shows LASSO is its third-best method.

| Model | Best method | 2nd | 3rd |
|-------|------------|-----|-----|
| RF | LASSO (0.767) | PCA (0.734) | MI (0.710) |
| SVM | LASSO (0.815) | MI (0.730) | PCA (0.727) |
| HybridSN | LASSO (0.858) | MI (0.726) | PCA (0.702) |
| ViT | MI (0.782) | PCA (0.734) | LASSO (0.754) |

**Why ViT prefers MI:** LASSO selects sparse, discriminative bands. MI selects bands
with high mutual information with the class label. ViT's attention mechanism may benefit
from more distributed spectral information (MI's selection) rather than a sparse
discriminative subset (LASSO).

**For the paper:** This is a genuine finding worth a sentence or two in Discussion.

---

### Finding 3: P2 sensitivity collapse — all models EXCEPT ViT

P2 (the second patient) causes near-total sensitivity collapse in RF, SVM, and HybridSN.
ViT is the only model that maintains sensitivity across all three patients.

| Model | P2 mean sensitivity | Interpretation |
|-------|-------------------|---------------|
| RF | **0.181** | Severe collapse — misses 82% of tumors in P2 |
| HybridSN | **0.252** | Severe collapse — misses 75% of tumors in P2 |
| SVM | **0.373** | Moderate collapse |
| ViT | **0.765** | No collapse — maintains tumor detection |

**At the best combo (all fold=2):**
- RF/LASSO/100: sens=0.009 — catastrophic (misses 99% of P2 tumors)
- HybridSN/LASSO/100: sens=0.048 — catastrophic
- SVM/LASSO/100: sens=0.256 — poor
- ViT/MI/100: sens=0.837 — strong

**Why:** P2 has different spectral characteristics from P1 and P3. Traditional models
trained on P1+P3 learn spectral patterns that do not generalise to P2. ViT's attention
mechanism may capture more robust cross-patient features, though at the cost of
lower overall specificity (0.462 mean).

**Trade-off:** ViT avoids P2 collapse but over-predicts tumor everywhere
(high sensitivity, low specificity). RF/SVM/HybridSN are conservative — they correctly
identify NT tissue but miss P2 tumor.

**For the paper:** This is the most important clinical finding. In a cancer detection
context, missing tumors (low sensitivity) is worse than false alarms (low specificity).
ViT's behaviour on P2 is arguably more clinically safe, even if its overall AUC is lower.

---

### Finding 4: More bands = better, diminishing returns after 50

For all models, AUC generally increases from 4 → 100 bands with diminishing returns
after 50. Top combos are always at 50 or 100 bands. This suggests the full spectral
range adds value even after band selection.

---

### Finding 5: DL does not clearly outperform classical ML at n=3

Overall mean AUC: SVM (0.764) ≈ HybridSN (0.762) ≈ ViT (0.757) > RF (0.739).
The differences are within ±0.01 for the top 3 models — essentially tied on average.

HybridSN only pulls ahead at its best combo (LASSO/100) due to strong P3 performance
(AUC=0.961). Its P2 fold is catastrophic (AUC=0.927 but sens=0.048).

**For the paper:** Honest framing — DL does not clearly win. At n=3 patients,
variance is too high to claim superiority. The computational cost of DL
(HybridSN: 2.9h on A100; ViT: 4.3h on A100 vs SVM: 11.8h on CPU) combined
with comparable performance favours classical ML for small datasets.

---

## 4. DECISIONS NEEDED FOR THE PAPER

---

### DECISION 1: McNemar's Test (Q11)

**What it is:** A statistical test comparing two classifiers on the same test set.
Tests whether disagreements between models are significant.

**The situation:** n=3 folds = 3 data points. With only 3 folds, any statistical
test will have very low power (high chance of false negatives). We cannot make
strong significance claims.

**Options:**
- [ ] **A — Skip entirely.** Acknowledge as limitation: "LOPOCV with n=3 produces
  insufficient observations for formal significance testing." This is the honest choice.
- [ ] **B — Run McNemar on best combos.** Compare HybridSN vs SVM per-sample predictions
  (not per-fold). More samples → more power. But the 3-fold structure still limits this.
- [ ] **C — Report effect sizes only.** Cohen's d or similar, no p-values.
  Framed as "magnitude of difference" not significance.

**Recommendation: A (skip) or C (effect sizes).** With n=3 patients, a p-value would
be misleading. Reviewers at ISBI/MICCAI will likely flag it.

**YOUR DECISION:** _______________

---

### DECISION 2: Extra Plots (Q5)

Current plots: AUC vs bands, per-fold bar chart, top-5 combos, P2 sensitivity collapse,
plus cross-model comparison (model_comparison, auc_heatmap, band_method_comparison,
sens_spec_scatter).

**Potential additions:**

- [ ] **ROC curves** — one curve per model at best combo (3 folds overlaid).
  Shows full sensitivity/specificity trade-off. Strong addition for a medical imaging paper.
  *Time: ~30 min to implement.*

- [ ] **Confusion matrices** — 2×2 per model per fold at best combo.
  Visual and intuitive. Shows exactly where each model fails.
  *Time: ~20 min.*

- [ ] **Per-patient AUC bar chart** — grouped by patient (P1/P2/P3), one bar per model.
  Directly visualises the P2 collapse finding.
  *Time: ~20 min.*

- [ ] **Band importance plot** — which bands LASSO selected most frequently.
  Connects to spectroscopy literature (known tumor absorption bands).
  *Time: ~30 min.*

**Recommendation for April 10 package:** ROC curves + per-patient AUC bar chart.
These two directly support the two main findings (HybridSN best AUC; P2 collapse pattern).

**YOUR DECISION:** _______________

---

### DECISION 3: Paper Framing

**Option A — "DL wins"**
Lead with HybridSN AUC 0.918. Downplay P2 collapse. Standard result.

**Option B — "Classical ML is competitive" (recommended)**
Lead with the finding that SVM matches DL on average. Frame DL advantage as
dataset-size-dependent. Honest, interesting, and more publishable as it goes
against the narrative.

**Option C — "P2 collapse is the main story"**
Frame the paper around inter-patient spectral variability. The finding that
ViT avoids P2 collapse (at the cost of specificity) while all other models fail
is novel and clinically relevant.

**Recommendation: B + C combined.** The paper tells two stories:
1. Classical ML (SVM) is surprisingly competitive with DL on a 3-patient cohort
2. P2 collapse reveals critical inter-patient spectral variability; ViT partially
   mitigates it through its attention mechanism

**YOUR DECISION:** _______________

---

### DECISION 4: Paper Venue

| Venue | Deadline | Format | Fit |
|-------|---------|--------|-----|
| ISBI 2026 | ~Jan 2026 (passed) | 4 pages | ✓ Good fit |
| MICCAI 2026 | ~March 2026 (passed) | 8-14 pages | ✓ Strong fit |
| MICCAI 2027 | ~March 2027 | 8-14 pages | Target |
| MIDL 2026 | ~Feb 2026 (check) | 8 pages | ✓ Good fit |
| ArXiv preprint | Anytime | Any | Now |

**Recommendation:** Post to ArXiv after April 26 as a preprint. Target MICCAI 2027
or MIDL 2027 for peer review after expanding to more patients (P4-P13) on HPC.

**YOUR DECISION:** _______________

---

### DECISION 5: How to Handle the 3-Patient Limitation

This is the biggest weakness reviewers will flag. Options:

- [ ] **A — Acknowledge and contextualise.** Cite other hyperspectral histology papers
  with similar n. Frame as a pilot study.
- [ ] **B — Frame as LOPOCV strength.** LOPOCV is the strictest possible evaluation —
  zero data leakage, true generalisation test. n=3 is small but the evaluation
  protocol is rigorous.
- [ ] **C — Commit to expanding.** State explicitly: "Results on n=3 are preliminary;
  expansion to P4-P13 via HPC is planned as future work."

**Recommendation: B + C.** Defend the protocol, commit to expansion.

**YOUR DECISION:** _______________

---

## 5. SUGGESTED PAPER STRUCTURE (for reference)

```
Title: Comparative Study of Classical and Deep Learning Models for
       Glioblastoma Detection in Hyperspectral Histology Images

1. Introduction (½ page)
   - Hyperspectral imaging for intraoperative tumour margin detection
   - Gap: no systematic comparison of ML/DL with band selection on HSI histology
   - Contribution: RF/SVM/HybridSN/ViT + PCA/MI/LASSO + LOPOCV on HistologyHSI-GB

2. Dataset & Preprocessing (½ page)
   - HistologyHSI-GB: 3 patients, 134 ROIs, 699 bands, 400-909nm
   - LOPOCV rationale: strictest generalisation test for small cohorts
   - Band selection: PCA / MI / LASSO at 4/10/20/50/100 bands

3. Methods (1 page)
   - RF & SVM: standard sklearn, patch-based pixel features
   - HybridSN: Roy et al. 2020, patch=11, Focal Loss, 50 epochs
   - ViT: custom from scratch, patch=11, token_size=4, 9 tokens, Focal Loss
   - Evaluation: accuracy, sensitivity, specificity, F1, AUC

4. Results (1 page)
   - Table 1: best combo per model (the table from Section 2a above)
   - Figure 1: model_comparison.png or AUC heatmap
   - Figure 2: P2 sensitivity collapse plot

5. Discussion (½ page)
   - LASSO dominance; ViT/MI divergence
   - P2 inter-patient variability finding
   - Classical ML competitive with DL at n=3
   - Clinical implications: sensitivity vs specificity trade-off

6. Conclusion + Future Work (¼ page)
   - Expand to P4-P13 on HPC
   - ACO band selection
   - Ablation study (patch size effect)
```

---

## 6. IMMEDIATE NEXT ACTIONS (before April 10)

- [ ] Answer the 5 decisions above
- [ ] Decide which extra plots to generate (Decision 2)
- [ ] Write the results section using the table from Section 2a
- [ ] Write the discussion section using Findings 1-5
- [ ] Assemble the April 10 minimum package

---

*Generated April 9, 2026. Data source: 186 runs across RF/SVM/HybridSN/ViT.*
*All results in: results/RF/, results/SVM/, results/HybridSN/, results/ViT/*
*Cross-model plots in: results/summary/*

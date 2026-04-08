# HSI Project — Session Decision Config
## Date: April 5, 2026
## Purpose: Complete decision log from Q1-Q15 planning session
## Read this at the start of every Claude Code terminal session

---

## GOVERNANCE RULES (READ FIRST — ALWAYS)

1. **Terminal is ground truth** — if browser Claude plan conflicts with what
   terminal knows from PROJECT_LOG.md, terminal wins. Always.

2. **Before any destructive action** (delete, overwrite, empty a file, clear
   a folder) — terminal must explicitly pause and state:
   "WARNING: This was not in the original plan. Confirm before proceeding."

3. **PROJECT_LOG.md is never emptied or overwritten** without explicit
   user confirmation. It is the single source of truth for project state.

4. **If browser Claude suggests something out of scope** — terminal flags it,
   pauses, asks permission before executing.

5. **After each session** — terminal updates PROJECT_LOG.md with what actually
   happened. Browser reads it next session to sync.

6. **MacBook upgrade will NOT happen before April 26, 2026** — all code must
   remain Windows 11 / PowerShell compatible until after that date.
   Terminal must never suggest Mac-specific or zsh-specific solutions
   before April 26.

---

## SYSTEM CAPABILITIES (CONFIRMED)

| Component | Detail |
|-----------|--------|
| OS | Windows 11 — PowerShell only, not cmd |
| Local RAM | 8 GB or less |
| Local GPU | Integrated GPU (ASUS VivoBook 2024) — no CUDA |
| Local training | RF/SVM via scikit-learn only (CPU) |
| Deep learning | Google Colab A100 only |
| Colab tier | Pro+ equivalent (Northeastern student) — A100 + high-RAM |
| Simultaneous Colab notebooks | Not yet confirmed — test before assuming parallelism |
| Local free disk | ~16.3 GB usable staging space (~19 GB if 3 GB freed) |
| Shell syntax | PowerShell ONLY — Remove-Item not rmdir |
| Python version | 3.14.3 — HDF5 only, no .npz, ASCII scripts only |

---

## GOOGLE DRIVE UPLOAD STATUS

| Item | Status |
|------|--------|
| Total h5 files needed | 134 |
| Uploaded as of April 5 | ~90 |
| Remaining | ~44 |
| Upload rate | ~4 files per hour |
| Upload method | Claude Code terminal (custom script) |
| RoboCopy | FAILED — do not retry |
| ETA for completion | ~11 hours from April 5 evening |
| Staging strategy | Upload one file, delete local copy immediately |
| Drive path | /content/drive/MyDrive/HSI/ |

**ACTION REQUIRED before training:**
- Confirm all 134 files uploaded
- Run audit.py in Colab against Drive files
- Only proceed to training if all 134 pass audit

---

## Q1 — DATA_DIR PATH IN NOTEBOOKS

**Status: TO BE CONFIRMED when notebooks are open**
- Check 4c_hybridSN.ipynb and 4d_vit.ipynb for hardcoded DATA_DIR
- Expected: /content/drive/MyDrive/HSI/ or similar
- Terminal must verify and update PROJECT_LOG.md with exact path
- Do not assume — open notebooks and check

---

## Q2 — COMPUTE TIER

**Colab: Pro+ equivalent (Northeastern University student account)**
- A100 GPU accessible
- High-RAM mode available (toggle between standard and high)
- Longer session limits than free tier
- Background execution available

**HPC Cluster (Northeastern Discovery):**
- Access NOT yet confirmed
- SSH test pending: ssh username@login.discovery.neu.edu
- Email: rchelp@northeastern.edu
- If access confirmed → revisit ACO band selection (all 5 counts)
- If access confirmed → SLURM job scripts for overnight unattended runs
- For now: treat as unavailable, plan around Colab only

---

## Q3 — CHECKPOINT RESUME LOGIC

**Status: UNTESTED**
- ViT was attempted but hit issues — is_done() logic never verified
  against a real interrupted session
- Checkpoint logic must be tested and verified BEFORE launching
  full 45-fold runs
- Test procedure: run 2 folds, manually interrupt, resume, confirm
  it picks up from fold 3 not fold 1
- Do not trust checkpoint logic until this test passes

---

## Q4 — ACO BAND SELECTION

**Decision: SKIP ACO for now**

**Reason:**
- ACO is CPU-bound — A100 provides no speedup
- Time estimates on Colab CPU:
  - 4 bands: 20-40 min
  - 10 bands: 45-90 min
  - 20 bands: 2-4 hours
  - 50 bands: 8-15 hours (Colab timeout risk)
  - 100 bands: 20-40 hours (will definitely die)
- Colab max session ~12 hours — 50 and 100 band ACO will not complete

**Band selection methods for this project:**
- PCA (unsupervised)
- Mutual Information (supervised)
- LASSO (supervised)
- ACO: SKIPPED

**Future ACO plan:**
- If HPC cluster access confirmed → run ACO for all 5 band counts
- ACO results for RF/SVM already done (bands_ACO.json exists or will)
- ACO for HybridSN/ViT: only if HPC available before April 26

**Band counts:** 4, 10, 20, 50, 100 (all methods except ACO)

---

## Q5 — VIT PATCH SIZE ABLATION

**Decision: patch_size=1 REMOVED from ablation**

**Reason (document in paper):**
- ViT with patch_size=1 means one token per pixel — no spatial neighbors
  in the attention mechanism
- Reduces to a spectral MLP, not a transformer
- Spatial context is fundamental to ViT's design — without it the
  architecture loses its core advantage
- This is a principled exclusion, not an oversight
- Paper methodology section must explicitly state this and explain why

**Remaining patch sizes for ViT ablation:** 6x6, 11x11, 22x22

---

## Q6 — CLASS WEIGHTS

**Decision: Per-fold automatic computation using compute_class_weight**

**Implementation (use exactly this in every model):**
```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=y_train  # training labels for THIS fold only
)
class_weight_dict = {0: weights[0], 1: weights[1]}
```

**Why:**
- Global 2.4x hardcoded is wrong — actual per-fold ratios vary:
  - Test P1: training weight ~1.96x
  - Test P2: training weight ~2.74x
  - Test P3: training weight ~2.44x
- Per-fold is methodologically correct and future-proof
- When dataset expands to more patients — same code, no changes needed
- No hardcoded numbers anywhere in codebase

**For DL models (HybridSN, ViT):**
- Use Focal Loss instead of plain class weights (see Q6-DL below)

---

## Q6-DL — FOCAL LOSS FOR DEEP LEARNING MODELS

**Decision: HybridSN and ViT use Focal Loss, not plain class weights**

**Why Focal Loss over class weights for DL:**
- Focal Loss focuses training on hard-to-classify pixels
- Better than plain weighting for CNN/ViT architectures
- Standard in medical imaging DL literature
- Works with imbalanced datasets natively

**Implementation note:**
- Use per-fold class weights inside Focal Loss alpha parameter
- alpha derived from compute_class_weight per fold
- gamma = 2.0 (standard starting point)

---

## Q7 — P2 SENSITIVITY COLLAPSE

**Decision: Document as finding, NO patient-specific fixes**

**What NOT to do:**
- No P2-specific code, thresholds, or adjustments
- No hardcoded handling for any individual patient
- No architecture changes based on one patient's behavior

**What TO do:**
- Report per-fold metrics separately (do not average-only report)
- Document P2 collapse under paper "Limitations" section
- Suggested paper language:
  "Results show inter-patient spectral variability consistent with
  glioblastoma heterogeneity. The sensitivity reduction observed when
  testing on P2 suggests this patient's tumor tissue has distinct
  spectral characteristics relative to P1 and P3. Expanding to
  additional patients (P4-P13) is expected to improve model
  generalization."
- This is a finding, not a bug

**Generalization principles applied throughout:**
- Per-fold class weights (not patient-specific)
- Focal Loss (handles hard examples generally)
- No overfitting to 3-patient dataset

---

## Q8 — MODEL RESULT REPORTING

**Decision: No bias toward any model — honest reporting**

**Rules:**
- If SVM beats HybridSN and ViT → report it exactly as-is
- No cherry-picking metrics to make DL look better
- No post-hoc threshold tuning to inflate specific model scores

**If DL does not win — paper framing:**
"Classical ML with carefully selected spectral bands is competitive
with deep learning approaches on small clinical HSI datasets. This
finding is consistent with the limited training data available (3
patients, 134 ROIs). Deep learning models are expected to demonstrate
their advantage as the dataset scales to additional patients."

**This is a publishable and honest conclusion.**

**Per-model expected behavior with 3 patients:**
- RF/SVM: Likely strongest (low data requirement)
- HybridSN: May underfit (needs more spatial examples)
- ViT: May underfit (transformers are data-hungry)

---

## Q9 — BUG HANDLING MID-RUN

**Decision: Granular per-fold checkpointing — Option C**

**Fold file naming convention:**
```
results/fold_001_PCA_4bands_testP1.json
results/fold_002_PCA_4bands_testP2.json
results/fold_003_PCA_4bands_testP3.json
results/fold_004_MI_4bands_testP1.json
...
```

**Rules:**
- Each fold saves individually immediately upon completion
- If bug found mid-run → fix bug, delete only affected fold files,
  rerun only those folds
- Never rerun all 45 folds unless catastrophic failure
- v2 tag added to any rerun fold files for audit trail
- Results aggregation script reads all fold JSONs and compiles final table

**Why this is mandatory given April 10 deadline:**
- Zero time to rerun 45 folds if something breaks
- A100 time is too valuable to waste on avoidable reruns
- Granular checkpointing is non-negotiable

---

## Q10 — STATISTICAL TESTING

**Decision: Two-phase approach**

**Phase 1 (April 10 target):**
- Report mean ± std across 3 folds per model
- No formal significance testing
- Include caveat in paper:
  "Formal significance testing was omitted due to n=3 LOPOCV folds.
  Results should be interpreted as indicative pending a larger
  patient cohort."

**Phase 2 (April 21 final paper):**
- Add McNemar's test (pixel-level comparisons)
- Works with n=3 patients because it uses ALL pixel predictions,
  not just 3 fold scores
- Provides genuine statistical power
- ~20 extra lines of code

**Why NOT Wilcoxon:**
- Wilcoxon needs minimum 5-6 data points
- With n=3 folds, p-value is statistically meaningless
- Reviewers from IEEE/MICCAI will flag this immediately

---

## Q11 — DRIVE INTEGRITY AUDIT

**Decision: Run audit.py in Colab BEFORE any training**

**Why this is non-negotiable:**
- Prior history of mismatches (P1 patient attribute wrong, corrupt
  800-byte stub file, RoboCopy failed mid-transfer)
- Audit takes 5-10 minutes
- A corrupt file discovered at fold 23 costs 8+ hours of A100 time

**Procedure:**
1. Confirm all 134 files uploaded
2. Mount Drive in Colab
3. Run audit.py as first cell
4. All 134 must pass: shape (800,1004,699), patient attribute,
   label attribute, wavelengths 699, values [0,1], no NaN/Inf
5. Any failures → reupload those specific files only
6. Only proceed to band selection and training after clean audit

---

## Q12 — TOP-LEVEL TO P1 REASSIGNMENT DOCUMENTATION

**Status: TO BE VERIFIED on next terminal session**

**What happened:**
- 9 P1 ROIs downloaded to HSI root instead of HSI/P1/
- Initially tagged patient="top-level" by preprocess.py
- Fixed with fix_patient.py: renamed files + patched HDF5 attribute
- All 134 verified via audit.py after fix

**Action required:**
- Terminal to check if paper draft mentions this correction
- If not → add paragraph to paper methodology section
- Suggested paper language:
  "During data preparation, 9 P1 ROIs were inadvertently downloaded
  to the root directory and initially tagged as top-level samples.
  Patient attribution was corrected programmatically via in-place
  HDF5 attribute patching, and all 134 files subsequently verified
  via audit.py to confirm correct patient and label assignments."
- This demonstrates rigorous data handling — include it, do not hide it

---

## Q13 — TARGET VENUE

**Current status: Course project / thesis work — no immediate submission**

**Future plan (post April 26, after dataset expansion):**
- Expand dataset to P4-P13 (more compute needed — HPC or MacBook)
- Target venues: ISBI or MICCAI (clinical HSI focus fits well)
- At that point: McNemar's test becomes mandatory
- Statistical rigor requirements increase for journal submission
  (IEEE TBME or Medical Image Analysis)

---

## Q14 — REUSABLE SKILL DOCUMENT

**Decision: Build comprehensive skill document by April 21**

**Contents to include:**
- Colab notebook template (checkpointing + focal loss + per-fold
  class weights pre-built)
- LOPOCV fold generator pattern
- Results saver pattern (per-fold JSON naming convention)
- Audit/verify pattern for Drive files
- Drive upload strategy (single file, delete local, verify)
- Band selection framework (PCA/MI/LASSO structure)
- HDF5 loading pattern
- Versioning conventions

**Purpose:**
- Career portfolio: each project adds reusable skills
- Terminal can load skill for future projects
- Next HSI project or medical imaging project starts faster
- Stacks toward AI engineering / research roles
- Future direction: AI agents — skill document will evolve

---

## Q15 — DEADLINES

| Date | Milestone |
|------|-----------|
| April 5 (today) | Planning session complete |
| April 6 | Drive upload complete, audit passes, band selection starts |
| April 7 | HybridSN training launches |
| April 8 | ViT training launches (parallel if A100 allows) |
| April 9 | Buffer — patch failed folds, collect results |
| **April 10** | **First complete results — showable target** |
| April 11-12 | Flex buffer (1-2 days acceptable) |
| April 21 | Paper draft complete, McNemar's test added |
| **April 26** | **HARD DEADLINE — Last day of Northeastern spring semester** |
| April 28 | Faculty grade submission deadline |
| April 29 | Spring degree conferral |

**MacBook upgrade: AFTER April 26 only — not a factor for this project**

---

## PARALLEL EXECUTION STRATEGY

| Track A — Colab (A100) | Track B — Local (CPU) |
|------------------------|----------------------|
| Band selection running | Git cleanup, notebook config |
| HybridSN training | ViT notebook being set up |
| ViT training | Results parsing scripts |
| Any failed fold reruns | PROJECT_LOG.md updates |

**Caution:** Two simultaneous heavy Colab sessions may compete for
A100 allocation. If that happens — stagger by 2-3 hours, not fully
parallel. Test before assuming full parallelism is available.

---

## EXPERIMENT GRID (FINAL)

**Band selection methods:** PCA, MI, LASSO (ACO skipped)
**Band counts:** 4, 10, 20, 50, 100
**LOPOCV folds:** 3 (test P1, test P2, test P3)
**Total HybridSN runs:** 3 methods x 5 counts x 3 folds = 45 folds
**Total ViT runs:** 3 methods x 5 counts x 3 folds = 45 folds
**RF/SVM runs:** handled separately (local, CPU)

**ViT patch sizes (ablation):** 6x6, 11x11, 22x22
(patch_size=1 excluded — see Q5)

---

## FUTURE ACTIONS (POST APRIL 26)

1. Verify Northeastern HPC cluster access
   - SSH: ssh username@login.discovery.neu.edu
   - Email: rchelp@northeastern.edu
   - If confirmed: run ACO band selection, expand to P4-P13
2. MacBook Apple Silicon transition
   - Update file paths (C:\... to /Users/chirag/...)
   - Update shell scripts (PowerShell to zsh/bash)
   - Update torch.device('cuda') to torch.device('mps')
   - OneDrive workarounds likely not needed
3. Dataset expansion (P4-P13)
   - Re-run full pipeline with more patients
   - LOPOCV becomes more statistically meaningful
   - DL models expected to show advantage over classical ML
4. Paper submission (ISBI or MICCAI)
   - Add McNemar's test results
   - Add full statistical analysis
   - Document P2 sensitivity collapse as key finding

---

## NOTES FOR TERMINAL

- This file was generated in browser Claude on April 5, 2026
- Terminal must cross-reference with PROJECT_LOG.md before acting
- If anything in this file conflicts with PROJECT_LOG.md — ask user
- PROJECT_LOG.md takes precedence for file paths and current state
- This file covers decisions and design choices only
- Update PROJECT_LOG.md after each session, not this file

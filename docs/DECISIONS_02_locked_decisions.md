# HSI Cancer Detection Project - Locked Decisions and Clarified Plan

This document consolidates the currently locked decisions for the hyperspectral imaging glioblastoma project so they can be passed to Claude Code or used as a single unambiguous planning reference. It is based on the uploaded project log and the decisions finalized in this discussion. The source log records the original open questions around GPU choice, ablation setup, session/runtime feasibility, significance testing, plots, presentation scope, McNemar's test, ACO, and ablation placement.[file:1]

## Purpose of this file

The goal of this file is to remove ambiguity from the active execution plan. It captures not only the final answer to each question, but also the reason for the decision, the practical consequence, and any wording that should be carried into code comments, planning notes, or paper drafting later.[file:1]

## Scope and status

These decisions apply to the current project timeline and especially to the immediate Phase 1 / April 10 milestone planning reflected in the project log.[file:1] Items explicitly deferred remain deferred; they are not deleted from long-term thinking, but they are removed from the immediate critical path unless noted otherwise.[file:1]

## Master decision table

| Question | Final decision | Operational meaning |
|---|---|---|
| Q1 | HybridSN used A100.[file:1] | Record A100 as the GPU for HybridSN. |
| Q2 | Exclude HybridSN patch size 1.[file:1] | Do not include patch size 1 in the current HybridSN ablation design. |
| Q3 | Skip ablation for now due to runtime limits.[file:1] | Do not schedule ablation in the current execution window. |
| Q4 | Remove Wilcoxon completely for Phase 1.[file:1] | Delete Wilcoxon code, plots, and CSV outputs from the current pipeline. |
| Q5 | Defer extra-plot decision until April 9 / until results are ready.[file:1] | Do not overbuild figures now; decide once results are in hand. |
| Q6 | Download results CSVs manually from Google Drive.[file:1] | Use a simple manual retrieval path after runs finish. |
| Q7 | April 10 target is a minimum showable package.[file:1] | Prioritize a compact, credible result set rather than a fully polished paper. |
| Q8 | Paper format not fully locked yet.[file:1] | Keep outputs structured so they can be adapted later. |
| Q9 | Keep P2 sensitivity collapse as a finding, not a bug fix.[file:1] | Report it honestly; do not attempt ad hoc correction. |
| Q10 | If SVM beats DL, report it honestly.[file:1] | Frame it as a legitimate small-dataset outcome if results support it. |
| Q11 | Defer McNemar decision until April 9.[file:1] | Do not implement now unless presentation needs force it. |
| Q12 | Keep ACO skipped unless explicitly reopened later.[file:1] | Remove ACO from the immediate execution path; mention as future work if needed. |
| Q13 | Put ablation in a separate section.[file:1] | If ablation is later included, it should stand as its own section rather than being buried. |

## Q1 - HybridSN GPU

### Final answer

HybridSN should be recorded as having run on **A100**.[file:1]

### What was ambiguous before

The project log shows that the HybridSN notebook originally referenced **Colab T4**, while the ViT notebook referenced **Colab A100**, and the training question existed because the live session allocation was unclear.[file:1] That ambiguity is now resolved for planning purposes by locking the final answer as **A100**.[file:1]

### What this means in practice

- Any run notes, experiment summaries, or paper-facing implementation notes should list **A100** for HybridSN.[file:1]
- No further follow-up is needed on the T4 versus A100 question unless a later audit requires raw notebook evidence.[file:1]
- The earlier T4 wording in the notebook history should be treated as outdated relative to the final locked decision.[file:1]

### Wording to preserve

Use language such as: *"HybridSN training was conducted on Google Colab with an A100 GPU."*[file:1]

## Q2 - HybridSN ablation patch sizes

### Final answer

Do **not** include **patch size 1** in the HybridSN ablation setup.[file:1]

### What the log originally contained

The project log notes that the HybridSN notebook had considered ablation patch sizes **1, 6, 11**, and the original open question was whether patch size 1 should stay because HybridSN's 3D convolutions can technically process 1x1 patches.[file:1]

### Why patch size 1 is being removed

Patch size 1 may be technically possible for HybridSN, but it weakens the intended meaning of the ablation because it largely removes the spatial neighborhood that makes a spatial patch comparison informative.[file:1] The current decision is to keep the ablation focused on meaningful spatial patch settings rather than allowing one condition that effectively collapses toward non-spatial spectral behavior.[file:1]

### What this means in practice

- If HybridSN ablation is revisited later, do not include patch size 1 in the active ablation list.[file:1]
- The working interpretation should be that the HybridSN patch-size ablation is intended to compare **spatial context scales**, not to test a degenerate no-neighborhood condition.[file:1]
- Any old code or notebook cell listing `1, 6, 11` should be updated so the current plan does not contradict itself.[file:1]

### Wording to preserve

Use a note such as: *"Patch size 1 was excluded from the HybridSN ablation to keep the comparison focused on meaningful spatial context; although HybridSN can technically operate on 1x1 patches, that setting reduces the experiment's value as a spatial-patch ablation."*[file:1]

## Q3 - Main runs plus ablation session planning

### Final answer

**Skip ablation for now.** Do not treat ablation as part of the current execution window.[file:1]

### What the original question was

The log framed Q3 as a practical Colab question: whether the **main runs plus ablation** would fit in one Colab session, or whether ablation should be done after a separate restart because the estimated runtime could exceed session limits.[file:1]

### Why the answer is now "skip ablation for now"

The current runtime experience makes the answer clear. The active timeline already shows approximately **15 hours for HybridSN** and about **18 hours for ViT**, which makes ablation unrealistic for the immediate schedule.[file:1] Because the project log already treated runtime and session length as a real constraint, the current decision formalizes that ablation is **not part of the immediate deliverable path**.[file:1]

### What this means in practice

- Do not allocate current Colab time to ablation.[file:1]
- Do not treat ablation outputs as blockers for the April 10 presentation target.[file:1]
- Do not spend time trying to optimize session packing for main-plus-ablation right now, because ablation is no longer in the immediate critical path.[file:1]

### Important nuance

Q13 still locks ablation as a **separate section** if it is included later.[file:1] That does **not** conflict with this decision: Q3 says ablation is **deferred now**, while Q13 says **how to place it later if it is eventually produced**.[file:1]

## Q4 - Wilcoxon in `5_compile_results.py`

### Final answer

Remove **all Wilcoxon-related content** from the Phase 1 pipeline.[file:1]

### Why this decision was made

The project log explicitly notes that Wilcoxon is not meaningful with the current fold count because the evaluation uses only **n = 3 folds** in LOPOCV, making a signed-rank test weak or misleading in this setting.[file:1]

### Required implementation changes

- Remove Wilcoxon code from `5_compile_results.py`.[file:1]
- Do not generate `wilcoxon_significance.csv`.[file:1]
- Do not generate Wilcoxon significance plots.[file:1]
- Remove any comments or references that make Wilcoxon appear to be part of the active Phase 1 deliverable.[file:1]

### What should remain in the plan

Do keep a short planning note stating that formal significance testing may be revisited later if the evaluation design changes or if a more appropriate test is added in a future phase.[file:1]

### Wording to preserve

Use language such as: *"Wilcoxon analysis was removed from Phase 1 because the current LOPOCV setting provides only three folds, which is insufficient for a meaningful signed-rank significance interpretation; formal statistical testing may be revisited in a later phase."*[file:1]

## Q5 - Additional plots beyond `5_compile_results.py`

### Final answer

Defer the extra-plot decision until **April 9** or until the actual result set is available for review.[file:1]

### Why this decision was made

The project log lists several possible figures beyond the current outputs, such as ROC curves, confusion matrices, and learning curves, but the necessity of each depends on what the final results look like and what the April 10 milestone actually needs to show.[file:1]

### What this means in practice

- Do not commit early to a large figure-generation workload.[file:1]
- Re-evaluate figure needs once the main training outputs are available.[file:1]
- Use April 9 as the checkpoint for deciding what is essential for presentation.[file:1]

### Current planning rule

For now, the rule is simple: generate only what is necessary to understand and present the main results, and postpone optional visuals until there is evidence they are needed.[file:1]

## Q6 - Getting result CSVs back from Drive

### Final answer

Retrieve the results **manually from Google Drive** after the runs finish.[file:1]

### Why this decision was made

The log raised multiple transfer options, including manual download, reverse `rclone`, and copying through Colab, but no stronger need was identified to justify more setup complexity for the current milestone.[file:1]

### What this means in practice

- After HybridSN and ViT runs complete, manually download the required result CSVs from Drive.[file:1]
- Once those CSVs are local, run the downstream analysis scripts locally as planned.[file:1]
- Do not spend extra time building an automated transfer solution unless it becomes necessary later.[file:1]

### Files that should remain on the retrieval checklist

The original log specifically called out the main DL result CSVs and both ablation CSVs as desired downstream analysis inputs.[file:1] Because ablation is currently skipped, the immediate retrieval priority is the **main result CSVs**, while ablation CSVs become relevant only if ablation is later revisited.[file:1]

## Q7 - What must be showable by April 10

### Final answer

The April 10 milestone should be treated as a **minimum showable package**, not a full polished paper package.[file:1]

### What this means

The target is to have a compact but credible set of results ready for discussion or review, centered on the key model outcomes and a few essential visuals.[file:1] The plan should not assume that every ideal figure, every statistical test, or every fully drafted paper section must be complete by that date.[file:1]

### Operational interpretation

For April 10, prioritize:
- core results tables,[file:1]
- the most useful plots,[file:1]
- concise supporting interpretation,[file:1]
- and a clean explanation of what is done, what is deferred, and why.[file:1]

### What not to assume

Do not interpret April 10 as requiring a complete conference-style paper, exhaustive ablation coverage, or fully finalized significance testing.[file:1]

## Q8 - Paper format and length

### Final answer

The final paper format is **not yet fully locked**.[file:1]

### What this means in practice

Because the log raises alternatives such as course report versus conference-style paper and does not yet settle the output format, all current deliverables should be organized in a way that can adapt to multiple final write-up structures.[file:1]

### Operational rule

- Keep metrics tables clean and exportable.[file:1]
- Keep figure captions and notes organized.[file:1]
- Keep methodological justifications written in a reusable way.[file:1]
- Avoid over-customizing outputs to a specific template before the template is confirmed.[file:1]

### Ambiguity status

This item is intentionally **not resolved** yet, but that ambiguity is controlled: the action is to preserve flexibility rather than forcing a premature format decision.[file:1]

## Q9 - Handling P2 sensitivity collapse

### Final answer

Treat the **P2 sensitivity collapse** as a real empirical finding and **not** as a bug to be patched away.[file:1]

### Why this matters

The project log already recognizes the poor P2 behavior as a cross-model generalization finding rather than a data-processing defect, and that interpretation remains the correct one for the current plan.[file:1]

### What this means in practice

- Do not introduce patient-specific fixes just to improve P2.[file:1]
- Do not hide the poor P2 performance in result summaries.[file:1]
- Preserve it as evidence of inter-patient difficulty and limited generalization in this setting.[file:1]

### Writing status

The exact polished paper language can still be drafted later, but the underlying decision is already locked: it is a result to be discussed, not a bug to be quietly repaired.[file:1]

## Q10 - If SVM beats HybridSN and ViT

### Final answer

If SVM outperforms the deep learning models, report that outcome **honestly and directly**.[file:1]

### Why this is acceptable

The project log already includes the framing that classical machine learning can remain competitive on small datasets, and the current data size and LOPOCV structure make that a legitimate interpretation rather than an embarrassment.[file:1]

### What this means in practice

- Do not bias the reporting toward DL just because DL was expected to be more sophisticated.[file:1]
- If SVM wins, present it as an evidence-based outcome consistent with limited sample size and strong handcrafted/classical baselines.[file:1]
- Keep the framing balanced and scientific rather than defensive.[file:1]

### Wording to preserve

Use language such as: *"On this small LOPOCV setting, classical ML remained highly competitive and in some cases outperformed the tested deep learning configurations."*[file:1]

## Q11 - McNemar's test

### Final answer

Defer the decision on McNemar's test until **April 9**.[file:1]

### What this means in practice

- Do not implement McNemar now.[file:1]
- Revisit only if the April 10 presentation requires a stronger pairwise comparison story.[file:1]
- Until then, the working plan remains descriptive reporting rather than expanded inferential testing.[file:1]

### Ambiguity status

This is a deliberate deferred decision, not an overlooked one. The ambiguity is controlled by the checkpoint date.[file:1]

## Q12 - ACO band selection

### Final answer

Keep **ACO skipped** unless it is explicitly reopened later.[file:1]

### Why this decision was made

The project log already treated ACO as deferred unless HPC access became available, and your current instruction is to keep it out of the active plan until you say otherwise.[file:1]

### What this means in practice

- Do not allocate current effort to ACO.[file:1]
- Do not let ACO block training, analysis, or presentation milestones.[file:1]
- Keep one short note that ACO was considered, deferred, and can be described as future work or a possible extension later.[file:1]

### Wording to preserve

Use language such as: *"ACO-based band selection was considered in project planning but deferred from the current submission path due to infrastructure and timeline constraints; it remains a potential future extension."*[file:1]

## Q13 - Ablation report placement

### Final answer

If ablation is included later, it should appear in a **separate ablation section**.[file:1]

### Why this decision matters

The project log explicitly raised whether ablation should be integrated into the main results or presented separately.[file:1] The locked decision is that it deserves its own subsection or section, which keeps the main model comparison cleaner and makes the design-validation discussion easier to follow.[file:1]

### Important interaction with Q3

Q13 does **not** mean ablation must be completed now.[file:1] Q3 already established that ablation is skipped for now due to runtime constraints.[file:1] Q13 only defines the structure to use **if ablation is later produced**.[file:1]

## Cross-decision consistency check

The following points are important to keep the project plan internally consistent and free of hidden contradictions.[file:1]

### Ablation status is deferred, not deleted

Ablation is currently **not part of the active runtime plan** because of time constraints.[file:1] However, if ablation is revisited later, it should be presented in a **separate section**, and HybridSN patch size 1 should remain excluded from that future ablation design.[file:1]

### Statistical testing is minimized in Phase 1

Wilcoxon is removed now, and McNemar is deferred until April 9.[file:1] This means the Phase 1 analysis should remain focused on descriptive summaries and core results rather than forcing weak or premature significance claims.[file:1]

### April 10 is a showable milestone, not a maximal milestone

The immediate plan is to make the work **presentable and defensible**, not exhaustive in every possible dimension.[file:1] That is why ablation is deferred, ACO is skipped, Wilcoxon is removed, and extra plots are postponed until the actual results can justify them.[file:1]

## Required implementation changes checklist

Use this checklist when editing notebooks, scripts, and planning notes.[file:1]

- Update any HybridSN notes so the recorded GPU is **A100**.[file:1]
- Remove HybridSN patch size 1 from the current ablation plan and comment why it is excluded.[file:1]
- Do not schedule ablation in the current runtime plan.[file:1]
- Remove Wilcoxon code and outputs from `5_compile_results.py`.[file:1]
- Leave a note that formal statistical testing may be revisited later.[file:1]
- Defer extra-plot selection until April 9 / until results are available.[file:1]
- Use manual Google Drive download for result CSV retrieval.[file:1]
- Keep April 10 expectations limited to a minimum showable results package.[file:1]
- Preserve P2 collapse as an observed finding.[file:1]
- Preserve honest reporting if SVM beats HybridSN or ViT.[file:1]
- Do not implement McNemar yet.[file:1]
- Keep ACO marked as future work unless explicitly reopened.[file:1]
- If ablation is later added, place it in a dedicated section.[file:1]

## Suggested planning language for Claude Code terminal

The following block is written as a direct implementation note with reduced ambiguity.[file:1]

```text
LOCKED PROJECT DECISIONS

1. HybridSN GPU is A100.
2. HybridSN ablation excludes patch size 1. Keep a note explaining that patch size 1 was removed because it weakens the intended spatial-context comparison.
3. Skip ablation for now. Current runtime is already too high (roughly 15h HybridSN, 18h ViT), so ablation is not in the immediate execution window.
4. Remove all Wilcoxon-related code, CSVs, and plots from 5_compile_results.py. Keep only a planning note that formal significance testing may be revisited later.
5. Defer extra-plot decisions until April 9 or until main results are available.
6. Retrieve result CSVs manually from Google Drive after runs finish.
7. April 10 target is a minimum showable package: core results plus essential figures, not a fully polished paper.
8. Final paper format is not fully locked; keep outputs organized so they can be adapted later.
9. Treat P2 sensitivity collapse as a real finding, not a bug fix target.
10. If SVM beats the DL models, report it honestly and frame it as classical ML being competitive on a small dataset.
11. Defer McNemar implementation/decision until April 9.
12. Keep ACO skipped unless explicitly reopened later; mention as future work if needed.
13. If ablation is later included, put it in a separate ablation section.
```

## Final ambiguity review

At this point, the plan is materially unambiguous on the active items that affect implementation and analysis.[file:1] The only remaining open items are intentionally deferred ones, such as the April 9 decision on extra plots and McNemar, and the still-flexible final paper format; these are not accidental gaps, but explicitly scheduled later decisions.[file:1]

In other words, nothing in this document should leave uncertainty about what to do **now**: run the main experiments, skip ablation for the moment, remove Wilcoxon, avoid premature statistical testing, keep ACO out of scope, and prepare a minimum but credible April 10 result package.[file:1]

# MV-SuStaIn: A Preliminary Simulation Validation

**Status: preliminary / observational. Full-power validation is ongoing.**

This report describes a small simulation study comparing MV-SuStaIn (a
longitudinal, joint patient-level likelihood extension of SuStaIn) against
classic SuStaIn (independent-visit training) on synthetic data. It is
released as a snapshot of ongoing work, not as a completed validation —
see [Limitations](#limitations) before drawing conclusions from it.

---

## 1. Background

SuStaIn (Subtype and Stage Inference) jointly infers, from cross-sectional
data, a small number of distinct disease progression sequences
("subtypes") and each individual's position along their subtype's
sequence ("stage") [1]. Classic SuStaIn scores every observation
independently, which discards information when a cohort instead has
multiple visits per patient: knowing that several visits belong to the
same person constrains which subtype and stage they can plausibly occupy.

MV-SuStaIn addresses this by aggregating a patient's visits into a single
joint likelihood *before* inferring subtype and stage, so repeated
observations reinforce one another instead of being treated as unrelated
data points. The implementation is available at
[github.com/\<org\>/mv-sustain](../README.md) (code, MIT license).

## 2. Simulation Design

Two-subtype synthetic cohorts (60 subjects, 4 biomarkers) were generated
under two likelihoods used by SuStaIn: **ordinal** (discrete clinical
rating-scale data, e.g. resembling MDS-UPDRS/MoCA-style items) and
**z-score** (continuous biomarker data). For each likelihood, cohorts were
simulated at four visit counts per patient (1, 2, 3, 6) and fit two ways —
classic (stacked, independent-visit) and MV-SuStaIn (longitudinal,
joint-likelihood) — on the *same* simulated data, so the only thing that
differs between the two fits is the training likelihood mechanism, not the
data.

Each of the 16 (likelihood × visit-count × mode) conditions was repeated
across **12 seeds**, with the classic and MV-SuStaIn fit of a given
condition sharing a seed so the comparison is paired (same simulated
cohort, two training mechanisms). Fitting used `N_startpoints=5`,
`N_mcmc=8000` per fit. Exact parameters and the driver script are
reproducible from the source repository's `scripts/jul2026/` (not included
in this public release, since it depends on the private research harness
— the method implementation itself, in this repository, is what matters
for reproducing the mechanism).

Sequence recovery is scored against the known ground-truth generative
sequence (Kendall's tau); subtype recovery is scored against the known
ground-truth subtype label (ARI, permutation accuracy). Per the project's
own methodology (classic SuStaIn's post-hoc visit-combination is a valid,
necessary step for it, but is not the correct comparison point for
MV-SuStaIn, whose default output already reflects the full joint-visit
posterior — using it there would inflate confidence), classic SuStaIn is
scored on its post-hoc metric and MV-SuStaIn on its unconstrained metric,
except at 1 visit, where the two are mathematically identical and no
post-hoc combination applies to either model.

## 3. Results

![Headline comparison figure](headline_figure.png)

*Mean ± SD across 12 paired repeats. Dashed grey: classic SuStaIn. Solid
blue: MV-SuStaIn.*

| Likelihood | Visits | ARI (MV) | ARI (classic) | p | Kendall-τ (MV) | Kendall-τ (classic) | p |
|---|---|---|---|---|---|---|---|
| Ordinal | 1 | 0.323 | 0.323 | — | 0.931 | 0.931 | — |
| Ordinal | 2 | 0.731 | 0.719 | 0.82 | 0.957 | 0.953 | 0.45 |
| Ordinal | 3 | 0.905 | 0.800 | 0.16 | 0.970 | 0.955 | 0.38 |
| Ordinal | 6 | 0.989 | 0.941 | 0.50 | 0.980 | 0.960 | 0.13 |
| Z-score | 1 | 0.074 | 0.074 | — | 0.744 | 0.744 | — |
| Z-score | 2 | 0.161 | 0.160 | 0.58 | 0.766 | 0.740 | 0.31 |
| Z-score | 3 | 0.336 | 0.345 | 0.70 | 0.775 | 0.759 | 0.41 |
| Z-score | 6 | 0.311 | 0.237 | 0.23 | 0.778 | 0.754 | 0.14 |

(p = two-sided Wilcoxon signed-rank, paired across the 12 shared-seed
repeats; full table with all four metrics in `aggregate_summary.csv`.)

**Sanity check:** at 1 visit, MV-SuStaIn and classic SuStaIn produce
identical results on every metric, exactly as expected — with a single
visit there is nothing for the joint-likelihood mechanism to aggregate
across, so the two training procedures are mathematically the same. This
is a basic correctness check on the implementation, not a scientific
finding.

**Ordinal likelihood:** MV-SuStaIn's mean is at or above classic SuStaIn
on both ARI and Kendall's tau at every visit count ≥2, and the Kendall's
tau gap widens with more visits (2 → 6 visits: p drops from 0.45 to 0.13).
This is directionally consistent with the hypothesis that joint-visit
training helps more as more visits become available, and it is the
cleanest single trend in this dataset.

**Z-score likelihood:** the pattern is less consistent — ARI is
essentially tied at 2-3 visits and only diverges in MV-SuStaIn's favor at
6 visits; direction is not monotonic across visit counts. This does not
replicate an earlier, informal finding (from before a June 2026 fix to the
simulation's noise handling) that MV-SuStaIn was *worse* on z-score
staging — the current, corrected picture is a wash rather than a
disadvantage, which is itself worth noting but is not a strong claim
either way.

## 4. Limitations

- **No result in this report reaches conventional statistical
  significance** (all Wilcoxon p ≥ 0.125). With only 12 paired repeats,
  this study is underpowered to detect the effect sizes observed here at
  p < 0.05. The trends above should be read as consistent with the
  hypothesis, not as proof of it.
- Simulated data only, at one set of noise/simulation settings. Real-cohort
  behavior (the subject of ongoing work) may differ.
- A larger-repeat follow-up (targeting statistical significance on the
  ordinal Kendall's-tau result specifically) and real-cohort validation
  are both in progress.

## Data & Code Availability

Code: this repository (MIT license). Raw per-repeat results and full
metric table: `validation/aggregate_summary.csv`. Simulated data only; no
real patient data was used in this study.

## References

1. Young AL, Marinescu RV, Oxtoby NP, et al. Uncovering the heterogeneity
   and temporal complexity of neurodegenerative diseases with Subtype and
   Stage Inference. *Nat Commun*. 2018;9(1):4273.

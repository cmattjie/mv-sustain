# mv-sustain

A longitudinal, multi-visit extension of the **SuStaIn** (Subtype and Stage
Inference) algorithm, built on top of
[pySuStaIn](https://github.com/ucl-pond/pySuStaIn).

SuStaIn jointly infers, from cross-sectional data, (a) a small number of
distinct progression sequences ("subtypes") and (b) each individual's
position along their subtype's sequence ("stage") — separating *which
pattern* someone follows from *how far along* they are, without needing
longitudinal follow-up.

Classic SuStaIn scores each observation (each visit) independently. When a
cohort instead has multiple visits per patient, that independence assumption
discards information: knowing that several visits belong to the same person
constrains which subtype and stage they can plausibly occupy. **MV-SuStaIn**
addresses this by aggregating a patient's visits into a single joint
likelihood *before* inferring subtype and stage, so repeated observations
reinforce one another instead of being treated as unrelated data points.

This repository provides:

- **`Stacked*Sustain`** classes — classic, independent-visit SuStaIn, provided
  as the fair baseline for comparison.
- **`Longitudinal*Sustain`** classes — the joint patient-level likelihood
  extension (MV-SuStaIn), for both z-score and ordinal likelihoods.
- **`SustainRunner`** — a single entry point that routes to the correct model
  class given a likelihood type and a `use_longitudinal_likelihood` flag, so
  most users won't need to touch the model classes directly.

## Status

This is research software under active development. The code here is a
periodically-updated extract of a larger, private research codebase where
the full validation study, simulation harness, and clinical application
work live. This repository is kept intentionally minimal: it's the reusable
algorithm layer, not the research pipeline built on top of it.

## Installation

```bash
git clone https://github.com/<your-org-or-username>/mv-sustain.git
cd mv-sustain
pip install -e .
```

This pulls in [pySuStaIn](https://github.com/ucl-pond/pySuStaIn) directly
from its GitHub repository (it is not published on PyPI).

## Quickstart

```bash
python examples/quickstart.py
```

This simulates a small multi-visit synthetic cohort with two known
progression subtypes and fits both the classic and longitudinal models on
it, to demonstrate the API end-to-end. It is intentionally small and fast —
see the script's docstring for details, and treat its output as a sanity
check rather than a performance benchmark.

Minimal usage sketch:

```python
from mv_sustain import SustainRunner

runner = SustainRunner(
    likelihood="zscore",
    n_subtypes=2,
    biomarker_labels=["biomarker_0", "biomarker_1", "biomarker_2"],
    dataset_name="my_cohort",
    output_folder="/path/to/output",
    N_startpoints=10,
    N_iterations_MCMC=10000,
    use_parallel_startpoints=False,
    use_longitudinal_likelihood=True,   # False for classic SuStaIn
    longitudinal_patient_ids=patient_ids,
    seed=0,
)
runner.fit(X_train, Z_vals=Z_vals, Z_max=Z_max, sigma_noise=1.0, patient_ids=patient_ids)
print(runner.fit_result_.ml_subtype, runner.fit_result_.ml_stage)
```

## Citing this work

If you use this package, please cite the original SuStaIn papers (per
pySuStaIn's own citation request):

1. Young AL, Marinescu RV, Oxtoby NP, et al. Uncovering the heterogeneity and
   temporal complexity of neurodegenerative diseases with Subtype and Stage
   Inference. *Nat Commun*. 2018;9(1):4273.
   https://doi.org/10.1038/s41467-018-05892-0
2. Aksman LM, Wijeratne PA, Oxtoby NP, et al. pySuStaIn: A Python
   implementation of the Subtype and Stage Inference algorithm. *SoftwareX*.
   2021;16:100811. https://doi.org/10.1016/j.softx.2021.100811
3. If using the ordinal likelihood: Young AL, Vogel JW, Robinson JL, et al.
   Ordinal SuStaIn: Subtype and Stage Inference for Clinical Rating Scale and
   Ordinal Data. *Front Artif Intell*. 2021;4:613261.
   https://doi.org/10.3389/frai.2021.613261

A citable reference for the longitudinal (MV-SuStaIn) extension itself will
be added here once available (manuscript/thesis in preparation). See
`CITATION.cff` for a machine-readable citation of this software as it
currently stands.

## License

MIT — see `LICENSE`. This project adapts and extends pySuStaIn (also MIT);
see `THIRD_PARTY_NOTICES.md` for pySuStaIn's original license text.

"""
Minimal end-to-end demo: simulate a small multi-visit z-score cohort with two known progression subtypes, then fit both the classic (stacked, independent-visit) and the longitudinal (joint patient-level likelihood) SuStaIn variants on the *same* data, and compare subtype-assignment accuracy against ground truth.

This is deliberately small and fast (a few dozen patients, short MCMC chains) so it runs on a laptop in well under a minute. It is meant to demonstrate the API, not to be a benchmark — see the accompanying paper/thesis chapter for the full validation study.

Usage:
    python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np

from mv_sustain import SustainRunner

RNG = np.random.default_rng(0)

N_BIOMARKERS = 3
N_THRESHOLDS = 3  # z-score event thresholds per biomarker (1, 2, 3 std devs)
Z_MAX_VALUE = 5.0
N_PATIENTS = 40
N_VISITS = 3
NOISE_SIGMA = 1.0

# Two subtypes defined by the *order* in which biomarkers reach abnormality.
# Subtype 0: biomarker 0 -> 1 -> 2.  Subtype 1: biomarker 2 -> 1 -> 0.
SUBTYPE_ONSET_STAGE = {
    0: np.array([1, 4, 7]),  # biomarker i reaches Z_max by this stage
    1: np.array([7, 4, 1]),
}
N_STAGES = 9  # total disease stages spanned by the simulation


def simulate_patient(subtype: int, true_stages: np.ndarray) -> np.ndarray:
    """Generate noisy z-scored observations for one patient across visits."""
    onset = SUBTYPE_ONSET_STAGE[subtype]
    observed = np.zeros((len(true_stages), N_BIOMARKERS))
    for v, stage in enumerate(true_stages):
        expected = np.minimum(stage / onset, 1.0) * Z_MAX_VALUE
        observed[v] = expected + RNG.normal(0, NOISE_SIGMA, size=N_BIOMARKERS)
    return observed


def main() -> None:
    Z_vals = np.tile(np.arange(1, N_THRESHOLDS + 1), (N_BIOMARKERS, 1)).astype(float)
    Z_max = np.full(N_BIOMARKERS, Z_MAX_VALUE)

    true_subtypes = RNG.integers(0, 2, size=N_PATIENTS)
    X_rows, patient_ids, stage_rows = [], [], []
    for p in range(N_PATIENTS):
        onset_stage = RNG.integers(1, N_STAGES, size=1)[0]
        visit_stages = np.clip(
            np.sort(onset_stage + RNG.integers(-2, 3, size=N_VISITS)), 0, N_STAGES
        )
        X_rows.append(simulate_patient(true_subtypes[p], visit_stages))
        patient_ids.extend([p] * N_VISITS)
        stage_rows.append(visit_stages)

    X_train = np.vstack(X_rows)
    patient_ids = np.array(patient_ids)

    common_kwargs = dict(
        likelihood="zscore",
        n_subtypes=2,
        biomarker_labels=[f"biomarker_{i}" for i in range(N_BIOMARKERS)],
        dataset_name="quickstart_demo",
        N_startpoints=3,
        N_iterations_MCMC=1000,
        use_parallel_startpoints=False,
        seed=0,
    )

    print("Fitting classic (stacked, independent-visit) SuStaIn...")
    stacked = SustainRunner(output_folder="/tmp/mv_sustain_quickstart/stacked", **common_kwargs)
    stacked.fit(X_train, Z_vals=Z_vals, Z_max=Z_max, sigma_noise=NOISE_SIGMA, plot=True)

    print("Fitting MV-SuStaIn (joint patient-level likelihood)...")
    longitudinal = SustainRunner(
        output_folder="/tmp/mv_sustain_quickstart/longitudinal",
        use_longitudinal_likelihood=True,
        longitudinal_patient_ids=patient_ids,
        **common_kwargs,
    )
    longitudinal.fit(
        X_train,
        Z_vals=Z_vals,
        Z_max=Z_max,
        sigma_noise=NOISE_SIGMA,
        patient_ids=patient_ids,
        plot=True,
    )

    # One ground-truth subtype label per patient (repeat to match the per-visit rows).
    true_subtypes_per_visit = np.repeat(true_subtypes, N_VISITS)

    def subtype_accuracy(runner: SustainRunner) -> float:
        pred = runner.fit_result_.ml_subtype
        # SuStaIn subtype labels are arbitrary permutations of {0, 1}; try both.
        acc_direct = np.mean(pred == true_subtypes_per_visit)
        acc_flipped = np.mean((1 - pred) == true_subtypes_per_visit)
        return max(acc_direct, acc_flipped)

    print(f"\nClassic SuStaIn subtype accuracy:      {subtype_accuracy(stacked):.2f}")
    print(f"MV-SuStaIn subtype accuracy:           {subtype_accuracy(longitudinal):.2f}")
    print(
        "\n(With only ~40 patients and short MCMC chains, treat this as a sanity "
        "check that both models run end-to-end, not as a performance claim.)"
    )


if __name__ == "__main__":
    main()

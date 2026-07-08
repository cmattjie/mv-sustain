# sustain_utils.py
# Generic SuStaIn runner for zscore, mixture, ordinal, using pySuStaIn classes directly.
# No silent fallbacks. If API mismatches, raise clearly with CHECK notes.
"""
Thin wrapper around pySuStaIn models.

The goal is to provide strict, explicit wiring for simulation/fitting/prediction without guessing APIs, while keeping outputs consistent across likelihood types.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Literal, Optional, Sequence, Tuple

import os
import pickle
import numpy as np

# Stacked (independent-visit) and MV (joint-patient) model classes.
from .stacked_sustain_override import StackedZscoreSustain, StackedOrdinalSustain  # type: ignore
from .mixture_override import MixtureSustain  # type: ignore

from .longitudinal_override import (
    LongitudinalMixtureSustain,
    LongitudinalOrdinalSustain,
    LongitudinalZscoreSustain,
)
from .sustain_helpers import apply_longitudinal_subtype_constraint, orient_p_subtype_stage

Likelihood = Literal["zscore", "mixture", "ordinal"]


def _most_probable_sequences_from_mcmc(samples_sequence: np.ndarray) -> np.ndarray:
    """
    Derive a "most probable" sequence per subtype by ordering events
    by their mean position across MCMC samples.

    samples_sequence: (n_subtypes, n_events, n_samples)
    Returns: (n_subtypes, n_events) sequence of event IDs.
    """
    samples_sequence = np.asarray(samples_sequence, dtype=int)
    if samples_sequence.ndim != 3:
        raise ValueError("samples_sequence must be 3D (n_subtypes, n_events, n_samples).")

    # Convert each sample's ordering into event positions, then average.
    n_subtypes, n_events, n_samples = samples_sequence.shape
    est_sequences = np.empty((n_subtypes, n_events), dtype=int)
    for s in range(n_subtypes):
        seq_samples = samples_sequence[s].T  # (n_samples, n_events)
        positions = np.empty((n_samples, n_events), dtype=float)
        for i in range(n_samples):
            positions[i, seq_samples[i]] = np.arange(n_events)
        mean_pos = positions.mean(axis=0)
        est_sequences[s] = np.argsort(mean_pos)
    return est_sequences


def compute_n_stages(
    *,
    likelihood: Likelihood,
    Z_vals: Optional[np.ndarray] = None,
    score_vals: Optional[np.ndarray] = None,
    n_features: Optional[int] = None,
) -> int:
    """
    Determine n_stages in a way consistent with common pySuStaIn conventions.

    zscore:
      n_stages = number of (feature, threshold) events = count(Z_vals > 0)

    mixture:
      n_stages = number of features (one event per feature) (CHECK if your version differs)

    ordinal:
      n_stages = number of (feature, score) events = count(score_vals > 0)

    No silent defaults: if required inputs are missing, raise.
    """
    # Normalize likelihood name and dispatch by model type.
    likelihood = str(likelihood).lower().strip()  # type: ignore
    if likelihood == "zscore":
        if Z_vals is None:
            raise ValueError("Z_vals is required to compute n_stages for zscore.")
        Z_vals = np.asarray(Z_vals)
        return int(np.sum(Z_vals > 0))

    if likelihood == "mixture":
        if n_features is None:
            raise ValueError("n_features is required to compute n_stages for mixture.")
        return int(n_features)

    if likelihood == "ordinal":
        if score_vals is None:
            raise ValueError("score_vals is required to compute n_stages for ordinal.")
        score_vals = np.asarray(score_vals)
        return int(np.sum(score_vals > 0))

    raise ValueError(f"Unknown likelihood: {likelihood}")


@dataclass
class SustainFitResult:
    samples_sequence: Any
    samples_f: Any
    ml_subtype: np.ndarray
    prob_ml_subtype: Any
    ml_stage: np.ndarray
    prob_ml_stage: Any
    prob_subtype_stage: Any
    est_sequences: np.ndarray
    mcmc_sequences: np.ndarray


class SustainRunner:
    """
    Thin, strict wrapper around pySuStaIn models.

    Supported likelihoods:
      - zscore: ZscoreSustain
      - mixture: MixtureSustain
      - ordinal: OrdinalSustain

    The goal is compatibility with standard pySuStaIn outputs, without silently guessing APIs.
    """

    def __init__(
        self,
        *,
        likelihood: Likelihood,
        n_subtypes: int,
        biomarker_labels: Sequence[str],
        output_folder: Optional[str] = None,
        out_pickle_folder: Optional[str] = None,
        dataset_name: str,
        N_startpoints: int,
        N_iterations_MCMC: int,
        use_parallel_startpoints: bool,
        seed: Optional[int] = None,
        use_longitudinal_likelihood: bool = False,
        longitudinal_patient_ids: Optional[Sequence[object]] = None,
        use_cache_recomputation: bool = True,
    ) -> None:
        # Store configuration and validate inputs up front.
        self.likelihood: Likelihood = str(likelihood).lower().strip()  # type: ignore
        if self.likelihood not in ("zscore", "mixture", "ordinal"):
            raise ValueError("likelihood must be one of: zscore, mixture, ordinal.")

        self.n_subtypes = int(n_subtypes)
        if self.n_subtypes <= 0:
            raise ValueError("n_subtypes must be > 0.")

        if out_pickle_folder is None and output_folder is None:
            raise ValueError("out_pickle_folder or output_folder must be provided.")
        if out_pickle_folder is not None and output_folder is not None:
            if str(out_pickle_folder) != str(output_folder):
                raise ValueError("out_pickle_folder and output_folder must match when both are provided.")

        self.biomarker_labels = list(biomarker_labels)
        self.out_pickle_folder = str(out_pickle_folder or output_folder)
        # Backward-compatible attribute name used throughout the codebase.
        self.output_folder = self.out_pickle_folder
        self.dataset_name = str(dataset_name)
        self.N_startpoints = int(N_startpoints)
        self.N_iterations_MCMC = int(N_iterations_MCMC)
        self.use_parallel_startpoints = bool(use_parallel_startpoints)
        self.seed = None if seed is None else int(seed)
        self.use_longitudinal_likelihood = bool(use_longitudinal_likelihood)
        self.longitudinal_patient_ids = None if longitudinal_patient_ids is None else list(longitudinal_patient_ids)
        self.use_cache_recomputation = bool(use_cache_recomputation)

        # Runtime objects (populated after fit).
        self.model_: Any = None
        self.fit_result_: Optional[SustainFitResult] = None

        self.labels_: Optional[np.ndarray] = None
        self.stages_: Optional[np.ndarray] = None
        self.prob_subtype_stage_: Any = None
        self.est_sequences_: Optional[np.ndarray] = None
        self.mcmc_sequences_: Optional[np.ndarray] = None

    def initialize_model(
        self,
        X_train: Any,
        *,
        Z_vals: Optional[np.ndarray] = None,
        Z_max: Optional[np.ndarray] = None,
        score_vals: Optional[np.ndarray] = None,
        sigma_noise: float = 1.0,
        sigma_mode: str = "fixed",
        cov: Optional[np.ndarray] = None,
        cov_blocks: Optional[Sequence[Sequence[int]]] = None,
        cov_kind: str = "abs",
        cov_scale: Optional[float] = None,
        cov_ridge: float = 1e-6,
        missing_policy: str = "error",
        patient_ids: Optional[Sequence[object]] = None,
        p_correct_mode: str = "fixed",
        X_obs_raw: Optional[np.ndarray] = None,
    ) -> "SustainRunner":
        # Ensure output folder exists for pySuStaIn pickles/plots.
        os.makedirs(self.output_folder, exist_ok=True)

        longitudinal_ids = patient_ids if patient_ids is not None else self.longitudinal_patient_ids
        if self.use_longitudinal_likelihood and longitudinal_ids is None:
            raise ValueError("patient_ids must be provided when use_longitudinal_likelihood=True.")

        if self.likelihood == "zscore":
            # Validate inputs and build ZscoreSustain model.
            X = np.asarray(X_train, dtype=float)
            if X.ndim != 2:
                raise ValueError("zscore fit expects X_train as 2D (n_subjects, n_features).")

            if Z_vals is None or Z_max is None:
                raise ValueError("zscore fit requires Z_vals and Z_max (no defaults).")

            Z_vals = np.asarray(Z_vals, dtype=float)
            Z_max = np.asarray(Z_max, dtype=float)

            if Z_vals.ndim != 2:
                raise ValueError("Z_vals must be 2D (n_features, n_thresholds).")
            if Z_max.ndim != 1:
                raise ValueError("Z_max must be 1D (n_features,).")
            if Z_vals.shape[0] != X.shape[1] or Z_max.shape[0] != X.shape[1]:
                raise ValueError("Z_vals/Z_max first dimension must match X_train n_features.")
            if len(self.biomarker_labels) != X.shape[1]:
                raise ValueError("biomarker_labels length must match X_train n_features (no silent relabel).")

            if self.use_longitudinal_likelihood:
                # MV path: joint patient-level likelihood during training.
                self.model_ = LongitudinalZscoreSustain(
                    X,
                    Z_vals,
                    Z_max,
                    self.biomarker_labels,
                    self.N_startpoints,
                    self.n_subtypes,
                    self.N_iterations_MCMC,
                    self.output_folder,
                    self.dataset_name,
                    self.use_parallel_startpoints,
                    seed=self.seed,
                    patient_ids=longitudinal_ids,
                    use_cache_recomputation=self.use_cache_recomputation,
                    sigma_noise=sigma_noise,
                    sigma_mode=sigma_mode,
                    cov=cov,
                    cov_blocks=cov_blocks,
                    cov_kind=cov_kind,
                    cov_scale=cov_scale,
                    cov_ridge=cov_ridge,
                    missing_policy=missing_policy,
                )
            else:
                # Stacked path: each visit is independent during training.
                self.model_ = StackedZscoreSustain(
                    X,
                    Z_vals,
                    Z_max,
                    self.biomarker_labels,
                    self.N_startpoints,
                    self.n_subtypes,
                    self.N_iterations_MCMC,
                    self.output_folder,
                    self.dataset_name,
                    self.use_parallel_startpoints,
                    seed=self.seed,
                    use_cache_recomputation=self.use_cache_recomputation,
                    sigma_noise=sigma_noise,
                    sigma_mode=sigma_mode,
                    cov=cov,
                    cov_blocks=cov_blocks,
                    cov_kind=cov_kind,
                    cov_scale=cov_scale,
                    cov_ridge=cov_ridge,
                    missing_policy=missing_policy,
                )

        elif self.likelihood == "mixture":
            # Validate inputs and build MixtureSustain model.
            if not (isinstance(X_train, (tuple, list)) and len(X_train) == 2):
                raise ValueError("mixture fit expects X_train=(L_yes, L_no).")
            L_yes = np.asarray(X_train[0], dtype=float)
            L_no = np.asarray(X_train[1], dtype=float)
            if L_yes.ndim != 2 or L_no.ndim != 2:
                raise ValueError("L_yes and L_no must be 2D.")
            if L_yes.shape != L_no.shape:
                raise ValueError("L_yes and L_no must have the same shape.")
            if len(self.biomarker_labels) != L_yes.shape[1]:
                raise ValueError("biomarker_labels length must match n_features.")

            if self.use_longitudinal_likelihood or self.use_cache_recomputation:
                self.model_ = LongitudinalMixtureSustain(
                    L_yes,
                    L_no,
                    self.biomarker_labels,
                    self.N_startpoints,
                    self.n_subtypes,
                    self.N_iterations_MCMC,
                    self.output_folder,
                    self.dataset_name,
                    self.use_parallel_startpoints,
                    seed=self.seed,
                    patient_ids=longitudinal_ids if self.use_longitudinal_likelihood else None,
                    use_cache_recomputation=self.use_cache_recomputation,
                )
            else:
                self.model_ = MixtureSustain(
                    L_yes,
                    L_no,
                    self.biomarker_labels,
                    self.N_startpoints,
                    self.n_subtypes,
                    self.N_iterations_MCMC,
                    self.output_folder,
                    self.dataset_name,
                    self.use_parallel_startpoints,
                    seed=self.seed,
                )

        elif self.likelihood == "ordinal":
            # Validate inputs and build OrdinalSustain model.
            if not (isinstance(X_train, (tuple, list)) and len(X_train) == 2):
                raise ValueError("ordinal fit expects X_train=(prob_nl, prob_score).")
            prob_nl = np.asarray(X_train[0], dtype=float)
            prob_score = np.asarray(X_train[1], dtype=float)
            if prob_nl.ndim != 2:
                raise ValueError("prob_nl must be 2D (n_subjects, n_biomarkers).")
            if prob_score.ndim != 3:
                raise ValueError("prob_score must be 3D (n_subjects, n_biomarkers, N_scores).")

            n_features = int(prob_nl.shape[1])
            if prob_score.shape[1] != n_features:
                raise ValueError("prob_score second dim must match prob_nl n_biomarkers.")
            if len(self.biomarker_labels) != n_features:
                raise ValueError("biomarker_labels length must match n_biomarkers.")

            if score_vals is None:
                raise ValueError("ordinal fit requires score_vals (no defaults).")
            score_vals = np.asarray(score_vals, dtype=int)
            if score_vals.ndim != 2 or score_vals.shape[0] != n_features:
                raise ValueError("score_vals must be 2D (n_biomarkers, N_scores).")
            if score_vals.shape[1] != prob_score.shape[2]:
                raise ValueError("score_vals N_scores must match prob_score last dimension.")

            if self.use_longitudinal_likelihood:
                # MV path: joint patient-level likelihood during training.
                self.model_ = LongitudinalOrdinalSustain(
                    prob_nl,
                    prob_score,
                    score_vals,
                    self.biomarker_labels,
                    self.N_startpoints,
                    self.n_subtypes,
                    self.N_iterations_MCMC,
                    self.output_folder,
                    self.dataset_name,
                    self.use_parallel_startpoints,
                    seed=self.seed,
                    patient_ids=longitudinal_ids,
                    use_cache_recomputation=self.use_cache_recomputation,
                    p_correct_mode=str(p_correct_mode),
                    X_obs_raw=X_obs_raw,
                )
            else:
                # Stacked path: each visit is independent during training.
                self.model_ = StackedOrdinalSustain(
                    prob_nl,
                    prob_score,
                    score_vals,
                    self.biomarker_labels,
                    self.N_startpoints,
                    self.n_subtypes,
                    self.N_iterations_MCMC,
                    self.output_folder,
                    self.dataset_name,
                    self.use_parallel_startpoints,
                    seed=self.seed,
                    use_cache_recomputation=self.use_cache_recomputation,
                    p_correct_mode=str(p_correct_mode),
                    X_obs_raw=X_obs_raw,
                )

        else:
            raise ValueError(f"Unknown likelihood: {self.likelihood}")

        return self

    def _cache_fit_result(self, fit_result: SustainFitResult) -> "SustainRunner":
        self.fit_result_ = fit_result
        self.labels_ = np.asarray(fit_result.ml_subtype, dtype=int).ravel()
        self.stages_ = np.asarray(fit_result.ml_stage, dtype=int).ravel()
        self.prob_subtype_stage_ = fit_result.prob_subtype_stage
        self.est_sequences_ = np.asarray(fit_result.est_sequences, dtype=int)
        self.mcmc_sequences_ = np.asarray(fit_result.mcmc_sequences, dtype=int)
        return self

    def fit(
        self,
        X_train: Any,
        *,
        Z_vals: Optional[np.ndarray] = None,
        Z_max: Optional[np.ndarray] = None,
        score_vals: Optional[np.ndarray] = None,
        sigma_noise: float = 1.0,
        sigma_mode: str = "fixed",
        cov: Optional[np.ndarray] = None,
        cov_blocks: Optional[Sequence[Sequence[int]]] = None,
        cov_kind: str = "abs",
        cov_scale: Optional[float] = None,
        cov_ridge: float = 1e-6,
        missing_policy: str = "error",
        patient_ids: Optional[Sequence[object]] = None,
        p_correct_mode: str = "fixed",
        X_obs_raw: Optional[np.ndarray] = None,
        plot: bool = False,
        plot_format: str = "png",
    ) -> "SustainRunner":
        self.initialize_model(
            X_train,
            Z_vals=Z_vals,
            Z_max=Z_max,
            score_vals=score_vals,
            sigma_noise=sigma_noise,
            sigma_mode=sigma_mode,
            cov=cov,
            cov_blocks=cov_blocks,
            cov_kind=cov_kind,
            cov_scale=cov_scale,
            cov_ridge=cov_ridge,
            missing_policy=missing_policy,
            patient_ids=patient_ids,
            p_correct_mode=p_correct_mode,
            X_obs_raw=X_obs_raw,
        )

        # Run EM + MCMC in pySuStaIn and unpack standard outputs.
        out = self.model_.run_sustain_algorithm(plot=plot, plot_format=plot_format)
        if not (isinstance(out, (tuple, list)) and len(out) == 7):
            raise RuntimeError(
                "Unexpected run_sustain_algorithm output. CHECK pySuStaIn version and expected tuple length."
            )

        (samples_sequence, samples_f, ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype_stage) = out
        # Store both raw samples and derived "most probable" sequence.
        samples_sequence_arr = np.asarray(samples_sequence)
        if samples_sequence_arr.ndim != 3:
            raise RuntimeError(
                f"Expected samples_sequence to be 3D (n_subtypes, n_events, n_samples), got {samples_sequence_arr.shape}."
            )
        mcmc_sequences = np.transpose(samples_sequence_arr, (0, 2, 1))
        est_sequences = _most_probable_sequences_from_mcmc(samples_sequence_arr)

        self.fit_result_ = SustainFitResult(
            samples_sequence=samples_sequence_arr,
            samples_f=samples_f,
            ml_subtype=np.asarray(ml_subtype, dtype=int).ravel(),
            prob_ml_subtype=prob_ml_subtype,
            ml_stage=np.asarray(ml_stage, dtype=int).ravel(),
            prob_ml_stage=prob_ml_stage,
            prob_subtype_stage=prob_subtype_stage,
            est_sequences=est_sequences,
            mcmc_sequences=mcmc_sequences,
        )
        return self._cache_fit_result(self.fit_result_)

    def load_posterior_state(
        self,
        *,
        mcmc_sequences_event_ids: np.ndarray,
        samples_f: np.ndarray,
        est_sequences_event_ids: Optional[np.ndarray] = None,
        ml_subtype: Optional[np.ndarray] = None,
        prob_ml_subtype: Any = None,
        ml_stage: Optional[np.ndarray] = None,
        prob_ml_stage: Any = None,
        prob_subtype_stage: Any = None,
    ) -> "SustainRunner":
        """
        Load a previously fit posterior state so predict() can score unseen data without refitting.

        Expected saved artifacts:
          - mcmc_sequences_event_ids: (n_subtypes, n_samples, n_events)
          - samples_f: subtype-fraction samples compatible with pySuStaIn predict
        """
        if self.model_ is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")

        mcmc_sequences = np.asarray(mcmc_sequences_event_ids, dtype=int)
        if mcmc_sequences.ndim != 3:
            raise ValueError(
                "mcmc_sequences_event_ids must be 3D (n_subtypes, n_samples, n_events)."
            )
        if mcmc_sequences.shape[0] != self.n_subtypes:
            raise ValueError(
                f"Subtype mismatch in mcmc_sequences_event_ids: got {mcmc_sequences.shape[0]} expected {self.n_subtypes}."
            )

        samples_sequence = np.transpose(mcmc_sequences, (0, 2, 1))
        est_sequences = (
            np.asarray(est_sequences_event_ids, dtype=int)
            if est_sequences_event_ids is not None
            else _most_probable_sequences_from_mcmc(samples_sequence)
        )
        if est_sequences.ndim != 2 or est_sequences.shape[0] != self.n_subtypes:
            raise ValueError("est_sequences_event_ids must be 2D with n_subtypes rows.")

        samples_f_arr = np.asarray(samples_f, dtype=float)
        if samples_f_arr.size == 0:
            raise ValueError("samples_f must contain at least one subtype-fraction sample.")

        n_subjects = 0
        if ml_subtype is not None:
            n_subjects = int(np.asarray(ml_subtype).size)
        elif ml_stage is not None:
            n_subjects = int(np.asarray(ml_stage).size)
        elif prob_subtype_stage is not None:
            n_subjects = int(np.asarray(prob_subtype_stage).shape[0])

        fit_result = SustainFitResult(
            samples_sequence=samples_sequence,
            samples_f=samples_f_arr,
            ml_subtype=np.zeros(n_subjects, dtype=int) if ml_subtype is None else np.asarray(ml_subtype, dtype=int).ravel(),
            prob_ml_subtype=prob_ml_subtype,
            ml_stage=np.zeros(n_subjects, dtype=int) if ml_stage is None else np.asarray(ml_stage, dtype=int).ravel(),
            prob_ml_stage=prob_ml_stage,
            prob_subtype_stage=prob_subtype_stage,
            est_sequences=np.asarray(est_sequences, dtype=int),
            mcmc_sequences=mcmc_sequences,
        )
        return self._cache_fit_result(fit_result)

    def predict(
        self,
        X_new: Any,
        *,
        score_vals: Optional[np.ndarray] = None,
        patient_ids: Optional[Sequence[object]] = None,
    ) -> np.ndarray:
        if self.model_ is None or self.fit_result_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # pySuStaIn uses subtype_and_stage_individuals_newData for predictions.
        fn = getattr(self.model_, "subtype_and_stage_individuals_newData", None)
        if fn is None:
            raise AttributeError("pySuStaIn model missing subtype_and_stage_individuals_newData. CHECK version.")

        ss = self.fit_result_.samples_sequence
        sf = self.fit_result_.samples_f
        n_predict_samples = int(ss.shape[2])

        if self.likelihood == "zscore":
            # Z-score predict uses continuous inputs.
            X = np.asarray(X_new, dtype=float)
            if X.ndim != 2:
                raise ValueError("zscore predict expects 2D array.")
            fn_params = inspect.signature(fn).parameters
            if "patient_ids" in fn_params and patient_ids is not None:
                out = fn(X, ss, sf, n_predict_samples, patient_ids=np.asarray(patient_ids))
            else:
                out = fn(X, ss, sf, n_predict_samples)

        elif self.likelihood == "mixture":
            # Mixture predict uses likelihood pairs.
            if not (isinstance(X_new, (tuple, list)) and len(X_new) == 2):
                raise ValueError("mixture predict expects (L_yes, L_no).")
            L_yes = np.asarray(X_new[0], dtype=float)
            L_no = np.asarray(X_new[1], dtype=float)
            fn_params = inspect.signature(fn).parameters
            if "patient_ids" in fn_params and patient_ids is not None:
                out = fn(
                    L_yes,
                    L_no,
                    ss,
                    sf,
                    n_predict_samples,
                    patient_ids=np.asarray(patient_ids),
                )
            else:
                out = fn(L_yes, L_no, ss, sf, n_predict_samples)

        elif self.likelihood == "ordinal":
            # Ordinal predict uses prob_nl/prob_score and sometimes score_vals.
            if not (isinstance(X_new, (tuple, list)) and len(X_new) == 2):
                raise ValueError("ordinal predict expects (prob_nl, prob_score).")
            prob_nl = np.asarray(X_new[0], dtype=float)
            prob_score = np.asarray(X_new[1], dtype=float)
            if prob_nl.ndim != 2 or prob_score.ndim != 3:
                raise ValueError("ordinal predict expects prob_nl 2D and prob_score 3D.")

            # Some pySuStaIn versions accept score_vals here, some do not.
            # We do not guess. We inspect parameter order and optional patient_ids support.
            fn_params = inspect.signature(fn).parameters
            param_names = list(fn_params.keys())
            has_patient_ids = "patient_ids" in fn_params

            kwargs = {}
            if has_patient_ids and patient_ids is not None:
                kwargs["patient_ids"] = np.asarray(patient_ids)

            # Standard order starts with (prob_nl, prob_score, samples_sequence, ...)
            if len(param_names) >= 5 and param_names[2] in ("samples_sequence",):
                out = fn(prob_nl, prob_score, ss, sf, n_predict_samples, **kwargs)
            elif len(param_names) >= 6:
                if score_vals is None:
                    raise ValueError("This pySuStaIn version requires score_vals in predict; pass it explicitly.")
                out = fn(prob_nl, prob_score, np.asarray(score_vals), ss, sf, n_predict_samples, **kwargs)
            else:
                raise TypeError(
                    f"Unexpected ordinal predict signature (params={param_names}). CHECK your pySuStaIn version."
                )
        else:
            raise ValueError(f"Unknown likelihood: {self.likelihood}")

        if not (isinstance(out, (tuple, list)) and len(out) == 7):
            raise RuntimeError("Unexpected predict output. CHECK pySuStaIn subtype_and_stage_individuals_newData output.")

        # Cache hard assignments and posterior for downstream usage.
        ml_subtype, _, ml_stage, _, _, _, prob_subtype_stage = out
        self.labels_ = np.asarray(ml_subtype, dtype=int).ravel()
        self.stages_ = np.asarray(ml_stage, dtype=int).ravel()
        self.prob_subtype_stage_ = prob_subtype_stage
        return self.labels_

    def predict_proba_subtype(self, *, n_subtypes: int, stage_axis: int = 1, subtype_axis: int = 2) -> np.ndarray:
        """
        Return marginal P(subtype | data) by summing prob_subtype_stage over stage.
        Assumes prob_subtype_stage is 3D.

        No guessing: caller must provide n_subtypes and ensure axes align with their pySuStaIn output.
        """
        # Sum over stage axis to marginalize P(subtype | data).
        pst = self.prob_subtype_stage_
        if pst is None:
            raise ValueError("No stored prob_subtype_stage. Call fit() or predict() first.")

        pst = np.asarray(pst)
        if pst.ndim != 3:
            raise ValueError(f"prob_subtype_stage must be 3D, got {pst.shape}.")

        stage_axis = int(stage_axis)
        subtype_axis = int(subtype_axis)
        if stage_axis == subtype_axis:
            raise ValueError("stage_axis and subtype_axis cannot be the same.")

        axes = [0, 1, 2]
        if stage_axis not in axes or subtype_axis not in axes:
            raise ValueError(f"stage_axis and subtype_axis must be in {axes}.")

        sample_axis = [a for a in axes if a not in (stage_axis, subtype_axis)]
        if len(sample_axis) != 1:
            raise ValueError("Could not infer sample axis.")
        sample_axis = sample_axis[0]

        pst_std = np.moveaxis(pst, (sample_axis, stage_axis, subtype_axis), (0, 1, 2))
        if pst_std.shape[2] != int(n_subtypes):
            raise ValueError(f"Subtype dim mismatch: got {pst_std.shape[2]} expected {int(n_subtypes)}.")

        return np.sum(pst_std, axis=1)

    def apply_longitudinal_constraint(self, patient_ids: Sequence[object], *, eps: float = 1e-12) -> dict:
        """
        Enforce a shared subtype per patient across repeated visits.

        This updates cached labels/stages/posteriors in-place and returns the adjusted outputs.
        """
        if self.prob_subtype_stage_ is None:
            raise ValueError("No stored prob_subtype_stage. Call fit() or predict() first.")
        if self.est_sequences_ is None:
            raise ValueError("No stored sequences. Call fit() first.")
        n_events = int(self.est_sequences_.shape[1])
        pst_raw = np.asarray(self.prob_subtype_stage_)
        if pst_raw.ndim != 3:
            raise ValueError("prob_subtype_stage must be 3D.")
        if pst_raw.shape[1:] == (n_events + 1, int(self.n_subtypes)):
            original_order = "stage_subtype"
        elif pst_raw.shape[1:] == (int(self.n_subtypes), n_events + 1):
            original_order = "subtype_stage"
        else:
            raise ValueError("Unexpected prob_subtype_stage shape for longitudinal constraint.")

        pst_oriented = orient_p_subtype_stage(
            pst_raw,
            n_events=n_events,
            n_subtypes=int(self.n_subtypes),
        )
        out = apply_longitudinal_subtype_constraint(pst_oriented, patient_ids, eps=eps)
        pst_adj = np.asarray(out["prob_subtype_stage"])
        if original_order == "stage_subtype":
            pst_adj = np.transpose(pst_adj, (0, 2, 1))
            out["prob_subtype_stage"] = pst_adj
        self.labels_ = np.asarray(out["pred_subtype"], dtype=int)
        self.stages_ = np.asarray(out["pred_stage"], dtype=int)
        self.prob_subtype_stage_ = out["prob_subtype_stage"]
        return out

    def get_fit_outputs(self) -> dict:
        """
        Return fit outputs required for evaluation (hard assignments, posteriors, sequences).
        """
        if self.fit_result_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        # Standardized output used by evaluation.
        return {
            "pred_subtype": np.asarray(self.labels_, dtype=int),
            "pred_stage": np.asarray(self.stages_, dtype=int),
            "p_subtype_stage": self.prob_subtype_stage_,
            "est_sequences": np.asarray(self.est_sequences_, dtype=int),
            "mcmc_sequences": np.asarray(self.mcmc_sequences_, dtype=int),
        }

    def _normalize_f_samples(self, samples_f: np.ndarray) -> np.ndarray | None:
        """
        Normalize MCMC subtype fraction samples into shape (n_subtypes, n_samples).
        Returns None if shape cannot be inferred.
        """
        sf = np.asarray(samples_f, dtype=float)
        if sf.size == 0:
            return None
        sf = np.squeeze(sf)
        if sf.ndim == 1:
            if sf.size == self.n_subtypes:
                return sf.reshape(self.n_subtypes, 1)
            return None
        if sf.ndim == 2:
            if sf.shape[0] == self.n_subtypes:
                return sf
            if sf.shape[1] == self.n_subtypes:
                return sf.T
        # Try to find subtype axis in higher-rank arrays.
        if sf.ndim >= 3:
            for axis in range(sf.ndim):
                if sf.shape[axis] == self.n_subtypes:
                    sf2 = np.moveaxis(sf, axis, 0)
                    return sf2.reshape(self.n_subtypes, -1)
        return None

    def get_mcmc_subtype_fractions(self) -> np.ndarray | None:
        """
        Return MCMC subtype fraction samples as (n_subtypes, n_samples), if available.
        """
        if self.fit_result_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._normalize_f_samples(np.asarray(self.fit_result_.samples_f))

    def get_samples_f(self) -> np.ndarray | None:
        """
        Return pySuStaIn ``samples_f`` in normalized (n_subtypes, n_samples) form.
        """
        return self.get_mcmc_subtype_fractions()

    def get_mcmc_subtype_fractions_mean(self) -> np.ndarray | None:
        """
        Return mean subtype fractions from MCMC samples, if available.
        """
        samples = self.get_mcmc_subtype_fractions()
        if samples is None:
            return None
        return np.mean(samples, axis=1)

    def _load_pickle_variables(self, *, n_subtypes: Optional[int] = None) -> dict:
        # pySuStaIn stores MCMC samples in per-subtype pickle files.
        n_subtypes = int(self.n_subtypes if n_subtypes is None else n_subtypes)
        pickle_dir = os.path.join(self.output_folder, "pickle_files")
        fname = f"{self.dataset_name}_subtype{n_subtypes - 1}.pickle"
        path = os.path.join(pickle_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"pickle file not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_samples_likelihood(self, *, n_subtypes: Optional[int] = None) -> np.ndarray:
        # Convenience accessor for MCMC likelihood samples.
        vars_ = self._load_pickle_variables(n_subtypes=n_subtypes)
        samples = np.asarray(vars_.get("samples_likelihood"))
        if samples.ndim != 1:
            samples = samples.ravel()
        return samples

    def get_parallel_startpoints_info(self) -> dict[str, Any]:
        requested = bool(self.use_parallel_startpoints)
        if self.model_ is None:
            return {
                "requested_parallel_startpoints": requested,
                "used_parallel_startpoints": requested,
                "parallel_startpoints_fallback_reason": None,
            }
        return {
            "requested_parallel_startpoints": bool(getattr(self.model_, "requested_parallel_startpoints", requested)),
            "used_parallel_startpoints": bool(getattr(self.model_, "used_parallel_startpoints", requested)),
            "parallel_startpoints_fallback_reason": getattr(self.model_, "parallel_startpoints_fallback_reason", None),
        }

    def get_loglikelihoods(self, *, tail_n: int = 100) -> dict:
        """
        Return log-likelihood summaries:
          - max over MCMC samples
          - mean of last N MCMC samples
          - log-likelihood for EM ML sequence/f
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Pull samples and ML estimates from pySuStaIn pickles.
        vars_ = self._load_pickle_variables()
        samples_likelihood = np.asarray(vars_.get("samples_likelihood"))
        if samples_likelihood.ndim != 1:
            samples_likelihood = samples_likelihood.ravel()
        if samples_likelihood.size == 0:
            raise ValueError("samples_likelihood is empty.")

        loglike_max_mcmc = float(np.max(samples_likelihood))
        tail_n = int(tail_n)
        tail_n = tail_n if tail_n > 0 else 1
        tail_n = min(tail_n, samples_likelihood.size)
        loglike_mean_tail_mcmc = float(np.mean(samples_likelihood[-tail_n:]))

        # Compute EM log-likelihood if EM outputs are present.
        ml_sequence_em = vars_.get("ml_sequence_EM")
        ml_f_em = vars_.get("ml_f_EM")
        if ml_sequence_em is None or ml_f_em is None:
            loglike_ml_em = float("nan")
        else:
            sustain_data = getattr(self.model_, "_AbstractSustain__sustainData", None)
            if sustain_data is None:
                raise AttributeError("Could not access sustainData on pySuStaIn model.")
            loglike_ml_em = float(self.model_._calculate_likelihood(sustain_data, ml_sequence_em, ml_f_em)[0])

        return {
            "loglike_max_mcmc": loglike_max_mcmc,
            "loglike_mean_tail_mcmc": loglike_mean_tail_mcmc,
            "loglike_ml_em": loglike_ml_em,
            "loglike_tail_n": int(tail_n),
        }

    def get_ml_subtype_fractions_em(self) -> np.ndarray | None:
        """
        Return EM ML subtype fractions from pySuStaIn pickles, if present.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        vars_ = self._load_pickle_variables()
        ml_f_em = vars_.get("ml_f_EM")
        if ml_f_em is None:
            return None
        arr = np.asarray(ml_f_em, dtype=float).reshape(-1)
        if arr.size == self.n_subtypes:
            return arr
        # Fallback: try to coerce any array with matching size.
        raw = np.asarray(ml_f_em, dtype=float)
        if raw.size == self.n_subtypes:
            return raw.reshape(self.n_subtypes)
        return None

    def get_ml_f_em(self) -> np.ndarray | None:
        """
        Return pySuStaIn ``ml_f_EM`` as a flat subtype-fraction vector.
        """
        return self.get_ml_subtype_fractions_em()

    def get_ml_sequence_em(self) -> np.ndarray | None:
        """
        Return pySuStaIn ``ml_sequence_EM`` as shape (n_subtypes, n_events), if available.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        vars_ = self._load_pickle_variables()
        ml_sequence_em = vars_.get("ml_sequence_EM")
        if ml_sequence_em is None:
            return None
        arr = np.asarray(ml_sequence_em, dtype=int)
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            return None
        if arr.shape[0] != self.n_subtypes and arr.size == self.n_subtypes * arr.shape[-1]:
            arr = arr.reshape(self.n_subtypes, -1)
        if arr.shape[0] != self.n_subtypes:
            return None
        return arr.astype(int)

    def cross_validate(self, *, test_idxs: Sequence[np.ndarray], select_fold: Optional[Sequence[int]] = None, plot: bool = False) -> tuple[Any, Any]:
        """Wrapper for pySuStaIn cross_validate_sustain_model."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        fn = getattr(self.model_, "cross_validate_sustain_model", None)
        if fn is None:
            raise AttributeError("pySuStaIn model missing cross_validate_sustain_model. CHECK version.")

        select_fold_arg: list[int] = []
        if select_fold is not None:
            select_fold_arg = [int(x) for x in select_fold]

        return fn(test_idxs, select_fold=select_fold_arg, plot=plot)

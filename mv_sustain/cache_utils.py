from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .payload_utils import _hash_payload


def _build_pickle_cache_dir(
    *,
    fit_mode: str,
    sim_mode: str,
    sim_backend: str,
    args: argparse.Namespace,
    seed: int,
    fit_n_subtypes: int,
    sim_n_subtypes: int,
    n_visits_per_patient: int | str,
    sim_max_visits_per_patient: int,
    visit_grouping: str,
    min_stage_gap_frac: float,
    subtype_fractions: np.ndarray,
    zscore_from_ordinal: bool,
) -> Path:
    default_cache_root = Path(__file__).resolve().parent.parent / "sustain_results" / "_pickle_cache"
    cache_root_override = getattr(args, "pickle_cache_root", None)
    if cache_root_override is None or str(cache_root_override).strip() == "":
        cache_root = default_cache_root
    else:
        cache_root = Path(str(cache_root_override)).expanduser()
    n_visits_field: int | str
    n_visits_key = str(n_visits_per_patient).lower().strip()
    if n_visits_key in ("mixed", "mixed-1", "empirical"):
        n_visits_field = n_visits_key
    else:
        n_visits_field = int(n_visits_per_patient)

    payload = {
        "fit_mode": str(fit_mode),
        "sim_mode": str(sim_mode),
        "sim_backend": str(sim_backend),
        "seed": int(seed),
        "n_biomarkers": int(args.n_biomarkers),
        "n_subjects": int(args.n_subjects),
        # Cache scope is intentionally keyed by simulated/ground-truth K.
        # This allows reusing pySuStaIn pickles across runs that only change
        # fitted K (e.g., K-range analysis) while isolating different simulators.
        "sim_n_subtypes": int(sim_n_subtypes),
        "subtype_fraction_scheme": str(args.subtype_fraction_scheme),
        "subtype_fractions": np.asarray(subtype_fractions, dtype=float).tolist(),
        "misdiagnosed_fraction": float(args.misdiagnosed_fraction),
        "n_visits_per_patient": n_visits_field,
        "sim_max_visits_per_patient": int(sim_max_visits_per_patient),
        "visit_grouping": str(visit_grouping),
        "min_stage_gap_frac": float(min_stage_gap_frac),
        "n_visits_w_missing": int(getattr(args, "n_visits_w_missing", 0)),
        "missing_proportion": getattr(args, "missing_proportion", None),
        "N_startpoints": int(args.N_startpoints),
        "N_mcmc": int(args.N_mcmc),
        "parallel_startpoints": bool(args.parallel_startpoints),
        "use_longitudinal_likelihood": bool(args.use_longitudinal_likelihood),
        "cv_folds": int(getattr(args, "cv_folds", 0)),
        "cv_seed": getattr(args, "cv_seed", None),
        "cv_parallel_folds": int(getattr(args, "cv_parallel_folds", 0)),
        "tsustain_sustain_init": bool(getattr(args, "tsustain_sustain_init", False)),
        "tsustain_sustain_init_longitudinal": bool(getattr(args, "tsustain_sustain_init_longitudinal", False)),
        "tsustain_init_n_startpoints": getattr(args, "tsustain_init_n_startpoints", None),
        "tsustain_init_n_mcmc": getattr(args, "tsustain_init_n_mcmc", None),
        "sim_method": str(args.sim_method),
        "sim_missing_rate": args.sim_missing_rate,
        "sim_missing_by_biomarker": args.sim_missing_by_biomarker,
        "ordinal_missing_policy": getattr(args, "ordinal_missing_policy", None),
        "missing_policy": getattr(args, "missing_policy", None),
        "sim_ordinal_cov_path": args.sim_ordinal_cov_path,
        "sim_ordinal_cov_groups": args.sim_ordinal_cov_groups,
        "sim_ordinal_cov_kind": str(args.sim_ordinal_cov_kind)
        if (args.sim_ordinal_cov_path is not None or args.sim_ordinal_cov_groups is not None)
        else None,
        "sim_ordinal_cov_block": args.sim_ordinal_cov_block,
        "sim_ordinal_cov_ridge": float(args.sim_ordinal_cov_ridge),
        "zscore_from_ordinal": bool(zscore_from_ordinal),
        "zscore_sigma_strategy": str(args.zscore_sigma_strategy),
        "zscore_cov_path": args.zscore_cov_path,
        "zscore_cov_kind": str(args.zscore_cov_kind) if args.zscore_cov_path is not None else None,
        "zscore_cov_block": args.zscore_cov_block,
        "zscore_cov_ridge": float(args.zscore_cov_ridge),
        "zscore_cov_scale": args.zscore_cov_scale,
        "zscore_cov_from_sim": bool(args.zscore_cov_from_sim),
        "zscore_cov_from_sim_rho_min": float(args.zscore_cov_from_sim_rho_min),
        "zscore_cov_shrinkage": str(args.zscore_cov_shrinkage),
        "zscore_cov_shrinkage_alpha": float(args.zscore_cov_shrinkage_alpha),
        "zscore_cov_block_max_size": int(args.zscore_cov_block_max_size),
        "sim_zscore_cov_path": args.sim_zscore_cov_path,
        "sim_zscore_cov_groups": args.sim_zscore_cov_groups,
        "sim_zscore_cov_kind": str(args.sim_zscore_cov_kind)
        if (args.sim_zscore_cov_path is not None or args.sim_zscore_cov_groups is not None)
        else None,
        "sim_zscore_cov_block": args.sim_zscore_cov_block,
        "sim_zscore_cov_ridge": float(args.sim_zscore_cov_ridge),
        "sim_zscore_cov_scale": args.sim_zscore_cov_scale,
    }
    sim_sequence_overlap_frac = float(getattr(args, "sim_sequence_overlap_frac", 0.0) or 0.0)
    if sim_sequence_overlap_frac > 0.0:
        payload["sim_sequence_overlap_frac"] = sim_sequence_overlap_frac
    sim_sequence_template_kendall_tau = getattr(args, "sim_sequence_template_kendall_tau", None)
    if sim_sequence_template_kendall_tau is not None:
        payload["sim_sequence_template_kendall_tau"] = float(sim_sequence_template_kendall_tau)
        payload["sim_sequence_template_kendall_tolerance"] = float(
            getattr(args, "sim_sequence_template_kendall_tolerance", 0.05)
        )
        payload["sim_sequence_template_max_tries"] = int(
            getattr(args, "sim_sequence_template_max_tries", 200)
        )
    visit_count_distribution_hash = getattr(args, "_effective_visit_count_distribution_hash", None)
    if visit_count_distribution_hash is not None:
        payload["visit_count_distribution_hash"] = str(visit_count_distribution_hash)
        payload["visit_count_distribution_source"] = getattr(
            args,
            "_effective_visit_count_distribution_source",
            None,
        )
    ts_alpha = getattr(args, "tsustain_alpha", None)
    if ts_alpha is not None:
        if isinstance(ts_alpha, (list, tuple, np.ndarray)):
            ts_alpha = [float(x) for x in ts_alpha]
        else:
            ts_alpha = [float(ts_alpha)]
        payload.update(
            {
                "tsustain_sequence_preset": str(getattr(args, "tsustain_sequence_preset", "")),
                "tsustain_alpha": ts_alpha,
            }
        )
    if sim_mode == "ordinal" or fit_mode == "ordinal":
        payload.update(
            {
                "N_scores": int(args.N_scores),
                "p_correct": float(args.p_correct),
                "p_correct_mode": str(getattr(args, "p_correct_mode", "fixed")),
                "noise_logic": str(args.noise_logic),
                "sim_p_correct": float(args.sim_p_correct),
                "sim_noise_logic": str(args.sim_noise_logic),
            }
        )
    if sim_mode == "zscore" or fit_mode == "zscore":
        payload.update(
            {
                "z_thresholds": list(args.z_thresholds),
                "z_max": float(args.z_max),
                "sigma_noise": float(args.sigma_noise),
                "sigma_mode": str(getattr(args, "sigma_mode", "fixed")),
                "sim_z_thresholds": list(args.sim_z_thresholds) if args.sim_z_thresholds is not None else None,
                "sim_z_max": float(args.sim_z_max) if args.sim_z_max is not None else None,
                "sim_sigma_noise": float(args.sim_sigma_noise),
            }
        )
    cache_key = _hash_payload(payload)
    return cache_root / cache_key

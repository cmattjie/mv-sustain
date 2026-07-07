from __future__ import annotations

import argparse
import hashlib
import json
import math
from typing import Any

import numpy as np


_EXTRA_PAYLOAD_KEYS = {
    "args",
    "sim_hash",
    "true_sequences_raw",
    "true_sequences_event_ids",
    "est_sequences_event_ids",
    "est_sequences_raw",
    "visit_sampling_strategy",
    "visit_sampling_seed",
    "visit_split_groups_per_patient",
    "visit_grouping_patient_ids",
    "visit_grouping_visit_index",
    "visit_split_source_patient_ids",
    "zscore_thresholds_source",
    "zscore_ordinal_mode",
    "zscore_sigma_strategy",
    "zscore_sigma_replaced_count",
    "zscore_sigma_replacement",
    "zscore_cov_path",
    "zscore_cov_kind",
    "zscore_cov_blocks",
    "zscore_cov_scale",
    "zscore_cov_ridge",
    "zscore_cov_from_sim",
    "zscore_cov_from_sim_rho_min",
    "zscore_cov_shrinkage",
    "zscore_cov_shrinkage_alpha",
    "zscore_cov_block_max_size",
    "sim_zscore_cov_path",
    "sim_zscore_cov_groups",
    "sim_zscore_cov_kind",
    "sim_ordinal_cov_path",
    "sim_ordinal_cov_groups",
    "sim_ordinal_cov_kind",
    "sim_ordinal_cov_block",
    "sim_ordinal_cov_ridge",
    "zscore_cov_est_blocks",
    "zscore_cov_est_min_eig",
    "zscore_cov_est_ridge",
    "repeat_summaries",
    "tsustain_uncertainty",
    "visit_count_distribution_targets",
}


def filter_args_for_payload(
    args: argparse.Namespace,
    *,
    fit_mode: str,
    sim_mode: str,
    use_longitudinal: bool,
) -> dict:
    args_dict = {k: v for k, v in vars(args).items() if not str(k).startswith("_")}
    drop_prefixes: list[str] = []
    if fit_mode != "tsustain":
        drop_prefixes.append("tsustain_")
    if fit_mode != "zscore":
        drop_prefixes.append("zscore_cov_")
    if sim_mode != "zscore":
        drop_prefixes.append("sim_zscore_cov_")
    if sim_mode != "ordinal":
        drop_prefixes.append("sim_ordinal_cov_")

    for key in list(args_dict.keys()):
        if any(key.startswith(prefix) for prefix in drop_prefixes):
            args_dict.pop(key, None)

    if fit_mode != "zscore":
        args_dict.pop("missing_policy", None)
    if fit_mode != "ordinal":
        args_dict.pop("ordinal_missing_policy", None)
    if not use_longitudinal:
        args_dict.pop("use_longitudinal_likelihood", None)
    visit_count_preset = str(args_dict.get("visit_count_distribution_preset", "none") or "none").strip().lower()
    visit_count_json = args_dict.get("visit_count_distribution_json")
    if visit_count_preset == "none" and (visit_count_json is None or str(visit_count_json).strip() == ""):
        args_dict.pop("visit_count_distribution_preset", None)
        args_dict.pop("visit_count_distribution_json", None)

    test_visit_count_preset = args_dict.get("test_visit_count_distribution_preset")
    test_visit_count_json = args_dict.get("test_visit_count_distribution_json")
    preset_inactive = (
        test_visit_count_preset is None
        or str(test_visit_count_preset).strip().lower() == "none"
    )
    json_inactive = test_visit_count_json is None or str(test_visit_count_json).strip() == ""
    if preset_inactive and json_inactive:
        args_dict.pop("test_visit_count_distribution_preset", None)
        args_dict.pop("test_visit_count_distribution_json", None)
    return args_dict


def split_payload(payload: dict) -> tuple[dict, dict]:
    default_payload = dict(payload)
    extra_payload: dict = {}

    for key in _EXTRA_PAYLOAD_KEYS:
        if key in default_payload:
            extra_payload[key] = default_payload.pop(key)

    if "cv_loglike_matrix" in default_payload:
        extra_payload["cv_loglike_matrix"] = default_payload.pop("cv_loglike_matrix")

    test_summary = default_payload.get("test_summary")
    if isinstance(test_summary, dict) and "test_grouping_summary" in test_summary:
        extra_payload["test_grouping_summary"] = test_summary.pop("test_grouping_summary")

    t_test_summary = default_payload.get("t-test_summary")
    if isinstance(t_test_summary, dict) and "t-test_grouping_summary" in t_test_summary:
        extra_payload["t-test_grouping_summary"] = t_test_summary.pop("t-test_grouping_summary")

    ts_cvic = default_payload.get("tsustain_cvic")
    if isinstance(ts_cvic, dict) and "results" in ts_cvic:
        extra_payload["tsustain_cvic"] = {"results": ts_cvic.pop("results")}

    return default_payload, extra_payload


def _normalize_fraction_vector(vec: np.ndarray | None) -> np.ndarray | None:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=float).ravel()
    if arr.size == 0:
        return None
    total = float(np.sum(arr))
    if not np.isfinite(total) or total <= 0:
        return None
    return arr / total


def _normalize_hash_value(val: Any) -> Any:
    if isinstance(val, np.ndarray):
        return _normalize_hash_value(val.tolist())
    if isinstance(val, (list, tuple)):
        return [_normalize_hash_value(v) for v in val]
    if isinstance(val, dict):
        return {k: _normalize_hash_value(val[k]) for k in sorted(val)}
    if isinstance(val, float):
        if math.isnan(val):
            return "nan"
        if math.isinf(val):
            return "inf" if val > 0 else "-inf"
    return val


def _hash_payload(payload: dict) -> str:
    normalized = _normalize_hash_value(payload)
    text = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()

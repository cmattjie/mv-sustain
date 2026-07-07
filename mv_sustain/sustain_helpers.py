from __future__ import annotations

"""Small helpers shared across SuStaIn simulation and evaluation."""

import argparse
from typing import Sequence
from itertools import permutations

import numpy as np

def resolve_simulation_overrides(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve sim_* overrides to fit values and normalize numeric inputs."""
    if args.sim_noise_logic is None:
        args.sim_noise_logic = args.noise_logic
    if args.sim_p_correct is None:
        args.sim_p_correct = args.p_correct
    if args.sim_z_thresholds is None:
        args.sim_z_thresholds = list(args.z_thresholds)
    if args.sim_z_max is None:
        args.sim_z_max = args.z_max
    if args.sim_sigma_noise is None:
        args.sim_sigma_noise = args.sigma_noise

    args.noise_logic = str(args.noise_logic)
    args.sim_noise_logic = str(args.sim_noise_logic)
    args.p_correct = float(args.p_correct)
    args.sim_p_correct = float(args.sim_p_correct)

    args.z_thresholds = [float(x) for x in args.z_thresholds]
    args.sim_z_thresholds = [float(x) for x in args.sim_z_thresholds]
    args.z_max = float(args.z_max)
    args.sim_z_max = float(args.sim_z_max)
    args.sigma_noise = float(args.sigma_noise)
    args.sim_sigma_noise = float(args.sim_sigma_noise)
    return args

def best_label_permutation_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> tuple[float, dict]:
    """
    Compute best accuracy over all permutations of label mapping.
    Strict: for n_classes > 8, raises (avoid accidental huge runtime).
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    K = int(n_classes)
    if K <= 0:
        raise ValueError("n_classes must be > 0.")
    if K > 8:
        raise ValueError("n_classes > 8 not supported by brute permutation matcher. Implement Hungarian if needed.")

    # Brute-force permutation mapping for small K.
    best_acc = -1.0
    best_map = None
    labels = list(range(K))
    for p in permutations(labels):
        mapping = {pred: true for pred, true in enumerate(p)}
        y_mapped = np.array([mapping[int(a)] for a in y_pred], dtype=int)
        acc = float(np.mean(y_mapped == y_true))
        if acc > best_acc:
            best_acc = acc
            best_map = mapping

    if best_map is None:
        raise RuntimeError("Internal error: no permutation evaluated.")
    return best_acc, best_map


def zscore_to_ordinal(Z_pd: np.ndarray, *, N_scores: int) -> np.ndarray:
    """Round z-scores to ordinal bins and clip to [0, N_scores]."""
    Z_pd = np.asarray(Z_pd, dtype=float)
    X_obs = np.rint(Z_pd).astype(np.int32)
    return np.clip(X_obs, 0, int(N_scores)).astype(np.int32, copy=False)


def ordinal_to_zscore(X_obs: np.ndarray, *, z_max: float | None) -> np.ndarray:
    """Treat ordinal scores as z-scores (optionally clipped)."""
    Z_pd = np.asarray(X_obs, dtype=np.float32)
    if z_max is not None:
        Z_pd = np.clip(Z_pd, -float(z_max), float(z_max)).astype(np.float32, copy=False)
    return Z_pd


def sequences_to_event_ids(sequences: np.ndarray, *, n_biomarkers: int, mode: str) -> np.ndarray:
    """Convert (biomarker, level) pairs into event indices."""
    sequences = np.asarray(sequences, dtype=int)
    if sequences.ndim == 2:
        return sequences
    if sequences.ndim != 3 or sequences.shape[2] != 2:
        raise ValueError("sequences must be (n_subtypes, n_events) or (n_subtypes, n_events, 2).")
    biomarker_idx = sequences[:, :, 0]
    level_idx = sequences[:, :, 1]
    if mode == "ordinal":
        return biomarker_idx + (level_idx - 1) * int(n_biomarkers)
    if mode == "zscore":
        return biomarker_idx + level_idx * int(n_biomarkers)
    raise ValueError("mode must be 'ordinal' or 'zscore'.")


def orient_p_subtype_stage(p_subtype_stage: np.ndarray, *, n_events: int, n_subtypes: int) -> np.ndarray:
    """Standardize p_subtype_stage to shape (n_subjects, n_subtypes, n_stages)."""
    pst = np.asarray(p_subtype_stage)
    if pst.ndim != 3:
        raise ValueError("p_subtype_stage must be 3D.")
    if pst.shape[1:] == (n_events + 1, n_subtypes):
        return np.transpose(pst, (0, 2, 1))
    if pst.shape[1:] == (n_subtypes, n_events + 1):
        return pst
    raise ValueError(
        "Unexpected p_subtype_stage shape. Expected (n_subjects, n_stages+1, n_subtypes) "
        "or (n_subjects, n_subtypes, n_stages+1)."
    )

def apply_longitudinal_subtype_constraint(
    p_subtype_stage: np.ndarray,
    patient_ids: Sequence[object],
    *,
    eps: float = 1e-12,
) -> dict:
    """
    Enforce shared subtype per patient across repeated visits, while staging per visit.

    Inputs:
      p_subtype_stage: (n_visits, n_subtypes, n_stages)
      patient_ids: length n_visits, labels grouping visits by patient
    Returns:
      dict with adjusted posteriors + hard assignments.
    """
    pst = np.asarray(p_subtype_stage, dtype=float)
    if pst.ndim != 3:
        raise ValueError("p_subtype_stage must be 3D (n_visits, n_subtypes, n_stages).")
    patient_ids = np.asarray(patient_ids)
    if patient_ids.shape[0] != pst.shape[0]:
        raise ValueError("patient_ids length must match number of visits.")

    n_visits, n_subtypes, _ = pst.shape
    p_subtype = np.sum(pst, axis=2)  # (n_visits, n_subtypes)

    unique_ids, inverse = np.unique(patient_ids, return_inverse=True)
    # Combine visit-level subtype evidence per patient (product in probability space).
    log_p = np.zeros((unique_ids.size, n_subtypes), dtype=float)
    for i in range(n_visits):
        log_p[inverse[i]] += np.log(p_subtype[i] + float(eps))

    log_p -= np.max(log_p, axis=1, keepdims=True)
    p_patient = np.exp(log_p)
    p_patient /= np.sum(p_patient, axis=1, keepdims=True)

    p_stage_given_subtype = pst / (p_subtype[:, :, None] + float(eps))
    p_patient_visit = p_patient[inverse]

    pst_adj = p_stage_given_subtype * p_patient_visit[:, :, None]
    pst_sum = np.sum(pst_adj, axis=(1, 2), keepdims=True)
    pst_adj /= (pst_sum + float(eps))

    pred_subtype = np.argmax(p_patient_visit, axis=1)
    pred_stage = np.empty(n_visits, dtype=int)
    for i in range(n_visits):
        pred_stage[i] = int(np.argmax(p_stage_given_subtype[i, pred_subtype[i]]))

    return {
        "pred_subtype": pred_subtype,
        "pred_stage": pred_stage,
        "prob_subtype_stage": pst_adj,
        "patient_ids_unique": unique_ids,
        "patient_subtype_posterior": p_patient,
    }

# ------------------------------------------------------------
# Visit grouping helpers
# ------------------------------------------------------------

def _unique_in_order(values: Sequence[object]) -> np.ndarray:
    """Return unique values in order of first appearance."""
    arr = np.asarray(values)
    if arr.size == 0:
        return arr
    _, first_idx = np.unique(arr, return_index=True)
    return arr[np.sort(first_idx)]


def apply_visit_grouping(
    *,
    rng: np.random.Generator,
    patient_ids: np.ndarray,
    visit_index: np.ndarray | None,
    n_visits_target: int | str,
    mode: str,
    n_visits_target_by_source_patient: np.ndarray | None = None,
    permute: bool = True,
) -> dict:
    """
    Subsample or split visits per patient with a per-patient permutation.

    - subsample: permute visits within each patient, take first n_visits_target.
    - split: permute visits within each patient, chunk into groups of n_visits_target.
    - mixed target: cycle k = 1..(V-1) across source patients (V = source visits).
      * subsample keeps k visits and drops the rest
      * split creates two groups: k and V-k visits
    - mixed-1 target: same as mixed, then drop any resulting 1-visit groups.

    Returns ordered indices to keep plus new patient_ids and visit_index.
    """
    patient_ids = np.asarray(patient_ids)
    if patient_ids.ndim != 1:
        raise ValueError("patient_ids must be 1D.")
    if visit_index is not None:
        visit_index = np.asarray(visit_index)
        if visit_index.shape[0] != patient_ids.shape[0]:
            raise ValueError("visit_index length must match patient_ids.")

    target_by_source_patient = None
    if n_visits_target_by_source_patient is not None:
        target_by_source_patient = np.asarray(n_visits_target_by_source_patient, dtype=np.int32).ravel()
        if target_by_source_patient.size == 0:
            raise ValueError("n_visits_target_by_source_patient must be non-empty.")
        if np.any(target_by_source_patient <= 0):
            raise ValueError("n_visits_target_by_source_patient must contain only positive integers.")

    n_visits_target_raw = str(n_visits_target).lower().strip()
    mixed_target = n_visits_target_raw in ("mixed", "mixed-1")
    mixed_drop_singletons = n_visits_target_raw == "mixed-1"
    n_visits_target_int = None
    if not mixed_target and target_by_source_patient is None:
        n_visits_target_int = int(n_visits_target)
        if n_visits_target_int <= 0:
            raise ValueError("n_visits_target must be > 0.")

    mode = str(mode).lower().strip()
    if mode not in ("subsample", "split"):
        raise ValueError("mode must be 'subsample' or 'split'.")
    if target_by_source_patient is not None:
        if mode != "subsample":
            raise ValueError("n_visits_target_by_source_patient is only supported with mode='subsample'.")
        if mixed_target:
            raise ValueError("n_visits_target_by_source_patient cannot be combined with mixed visit targets.")

    unique_ids = _unique_in_order(patient_ids)
    if target_by_source_patient is not None and target_by_source_patient.shape[0] != unique_ids.shape[0]:
        raise ValueError("n_visits_target_by_source_patient must have one entry per source patient.")
    selected: list[int] = []
    new_patient_ids: list[object] = []
    new_visit_index: list[int] = []
    source_patient_ids: list[object] = []
    split_groups_per_patient: list[int] = []
    mixed_k_by_source_patient: list[int] = []
    mixed_group_sizes: list[int] = []

    next_pid = 0
    for src_i, pid in enumerate(unique_ids):
        idx = np.where(patient_ids == pid)[0]
        if idx.size == 0:
            continue
        if visit_index is not None:
            idx = idx[np.argsort(visit_index[idx], kind="mergesort")]
        if permute:
            idx = idx[rng.permutation(idx.size)]

        mixed_k = None
        if mixed_target:
            if idx.size < 2:
                raise ValueError("n_visits_target in {'mixed', 'mixed-1'} requires at least 2 visits per source patient.")
            mixed_k = 1 + (src_i % (idx.size - 1))
            mixed_k_by_source_patient.append(int(mixed_k))

        if mode == "subsample":
            if target_by_source_patient is not None:
                keep_count = int(target_by_source_patient[src_i])
            else:
                keep_count = int(mixed_k) if mixed_target else int(n_visits_target_int)
            if idx.size < keep_count:
                raise ValueError("n_visits_target exceeds available visits for a patient.")
            chosen = idx[:keep_count]
            if visit_index is not None:
                chosen = chosen[np.argsort(visit_index[chosen], kind="mergesort")]
            else:
                chosen = np.sort(chosen)
            if mixed_drop_singletons and chosen.size == 1:
                continue
            selected.extend(chosen.tolist())
            new_patient_ids.extend([pid] * int(chosen.size))
            new_visit_index.extend(list(range(int(chosen.size))))
            source_patient_ids.append(pid)
            if mixed_target:
                mixed_group_sizes.append(int(chosen.size))
        else:
            if mixed_target:
                kept_groups = 0
                chunks = [idx[: int(mixed_k)], idx[int(mixed_k) :]]
                for chunk in chunks:
                    if visit_index is not None:
                        chunk = chunk[np.argsort(visit_index[chunk], kind="mergesort")]
                    else:
                        chunk = np.sort(chunk)
                    if chunk.size == 0:
                        raise ValueError("mixed split produced an empty chunk.")
                    if mixed_drop_singletons and chunk.size == 1:
                        continue
                    selected.extend(chunk.tolist())
                    new_patient_ids.extend([next_pid] * int(chunk.size))
                    new_visit_index.extend(list(range(int(chunk.size))))
                    source_patient_ids.append(pid)
                    mixed_group_sizes.append(int(chunk.size))
                    next_pid += 1
                    kept_groups += 1
                if kept_groups > 0:
                    split_groups_per_patient.append(int(kept_groups))
            else:
                if idx.size % int(n_visits_target_int) != 0:
                    raise ValueError("visit_grouping='split' requires visit count divisible by n_visits_target.")
                n_groups = int(idx.size // int(n_visits_target_int))
                split_groups_per_patient.append(n_groups)
                for g in range(n_groups):
                    chunk = idx[g * int(n_visits_target_int) : (g + 1) * int(n_visits_target_int)]
                    if visit_index is not None:
                        chunk = chunk[np.argsort(visit_index[chunk], kind="mergesort")]
                    else:
                        chunk = np.sort(chunk)
                    selected.extend(chunk.tolist())
                    new_patient_ids.extend([next_pid] * int(chunk.size))
                    new_visit_index.extend(list(range(int(chunk.size))))
                    source_patient_ids.append(pid)
                    next_pid += 1

    split_groups = None
    if mode == "split" and split_groups_per_patient:
        unique_groups = sorted(set(split_groups_per_patient))
        if len(unique_groups) == 1:
            split_groups = unique_groups[0]

    if mixed_target:
        if mode == "subsample":
            strategy = (
                "permute_then_take_mixed_cycle_drop_singletons"
                if mixed_drop_singletons
                else "permute_then_take_mixed_cycle"
            )
        else:
            strategy = (
                "permute_then_split_mixed_cycle_drop_singletons"
                if mixed_drop_singletons
                else "permute_then_split_mixed_cycle"
            )
    elif target_by_source_patient is not None:
        strategy = "permute_then_take_per_patient_targets"
    else:
        strategy = "permute_then_take" if mode == "subsample" else "permute_then_chunk"
    return {
        "indices": np.asarray(selected, dtype=int),
        "patient_ids": np.asarray(new_patient_ids),
        "visit_index": np.asarray(new_visit_index, dtype=np.int32),
        "source_patient_ids": np.asarray(source_patient_ids),
        "strategy": strategy,
        "split_groups_per_patient": split_groups,
        "mixed_cycle_enabled": bool(mixed_target),
        "mixed_drop_singletons": bool(mixed_drop_singletons),
        "mixed_k_by_source_patient": np.asarray(mixed_k_by_source_patient, dtype=np.int32)
        if mixed_target
        else None,
        "mixed_group_sizes": np.asarray(mixed_group_sizes, dtype=np.int32) if mixed_target else None,
        "n_visits_target_by_source_patient": target_by_source_patient,
    }

# ------------------------------------------------------------
# Easy-read helpers for sequences and small utilities
# ------------------------------------------------------------

def to_list(arr: np.ndarray | None) -> list | None:
    """Return arr.tolist() or None (safe for JSON payloads)."""
    if arr is None:
        return None
    return np.asarray(arr).tolist()


def _bm_label(idx: int) -> str:
    """Map biomarker index -> 'A','B','C',... then 'X{n}' after Z."""
    i = int(idx)
    if 0 <= i < 26:
        return chr(65 + i)
    return f"X{i+1}"


def sequences_pairs_to_easyread(
    seqs_pairs: np.ndarray | None,
    *,
    mode: str,
    n_biomarkers: int,
) -> list | None:
    """Convert (biomarker, level) sequence pairs to tokens like 'A1','B2'.

    - For zscore mode: levels are 0-based thresholds -> display as 1-based.
    - For ordinal mode: levels are already 1..N.
    """
    if seqs_pairs is None:
        return None
    arr = np.asarray(seqs_pairs)
    if arr.ndim != 3 or arr.shape[2] != 2:
        return None
    tokens: list[list[str]] = []
    is_z = str(mode).lower().strip() == "zscore"
    for s in range(arr.shape[0]):
        toks: list[str] = []
        for (b, lvl) in arr[s]:
            b = int(b)
            lvl = int(lvl)
            lvl_disp = (lvl + 1) if is_z else lvl
            toks.append(f"{_bm_label(b)}{lvl_disp}")
        tokens.append(toks)
    return tokens


def sequences_eids_to_easyread(
    seqs_eids: np.ndarray | None,
    *,
    mode: str,
    n_biomarkers: int,
) -> list | None:
    """Convert event-id sequences to easy-read tokens like 'A1','B2'."""
    if seqs_eids is None:
        return None
    arr = np.asarray(seqs_eids)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        return None
    tokens: list[list[str]] = []
    m = int(n_biomarkers)
    is_z = str(mode).lower().strip() == "zscore"
    for s in range(arr.shape[0]):
        toks: list[str] = []
        for eid in arr[s]:
            eid = int(eid)
            b = eid % m
            lvl_idx = eid // m
            lvl_disp = lvl_idx + 1
            toks.append(f"{_bm_label(b)}{lvl_disp}")
        tokens.append(toks)
    return tokens


def fmt_float(val: float) -> str:
    """Format floats for filesystem tags (1.0 -> 1p0)."""
    return str(val).replace(".", "p")


def fmt_list(vals: list[float]) -> str:
    """Format a list of floats for tag strings."""
    return "-".join(fmt_float(float(v)) for v in vals)


def build_outdir_tag(
    *,
    args: argparse.Namespace,
    fit_mode: str,
    sim_mode: str,
    fit_n_subtypes: int,
    sim_n_subtypes: int,
    sim_noise_logic: str,
) -> str:
    # Compose a deterministic output folder name from experiment settings.
    parts: list[str] = []
    parts.append(f"fit-{fit_mode}")
    parts.append(f"sim-{sim_mode}")
    parts.append(f"fitK{int(fit_n_subtypes)}")
    parts.append(f"simK{int(sim_n_subtypes)}")
    parts.append(f"bm{int(args.n_biomarkers)}")

    parts.append(f"n{int(args.n_subjects)}")

    if fit_mode == "ordinal" or sim_mode == "ordinal":
        parts.append(f"scores{int(args.N_scores)}")
        parts.append(f"pc{fmt_float(float(args.p_correct))}")
        sim_p_correct = float(getattr(args, "sim_p_correct", args.p_correct))
        if sim_p_correct != float(args.p_correct):
            parts.append(f"simp{fmt_float(sim_p_correct)}")
        parts.append(f"fitnl-{str(args.noise_logic)}")
        if sim_noise_logic != str(args.noise_logic):
            parts.append(f"simnl-{sim_noise_logic}")

    if fit_mode == "zscore" or sim_mode == "zscore":
        parts.append(f"zt{fmt_list([float(x) for x in args.z_thresholds])}")
        parts.append(f"zmax{fmt_float(float(args.z_max))}")
        sim_z_thresholds = getattr(args, "sim_z_thresholds", None)
        if sim_z_thresholds is not None:
            sim_list = [float(x) for x in sim_z_thresholds]
            fit_list = [float(x) for x in args.z_thresholds]
            if sim_list != fit_list:
                parts.append(f"simzt{fmt_list(sim_list)}")
        sim_z_max = getattr(args, "sim_z_max", None)
        if sim_z_max is not None and float(sim_z_max) != float(args.z_max):
            parts.append(f"simzmax{fmt_float(float(sim_z_max))}")
    if sim_mode == "zscore":
        sim_sigma_noise = float(getattr(args, "sim_sigma_noise", args.sigma_noise))
        parts.append(f"sig{fmt_float(sim_sigma_noise)}")

    parts.append(f"seed{int(args.seed)}")
    parts.append(f"sp{int(args.N_startpoints)}")
    parts.append(f"mcmc{int(args.N_mcmc)}")
    parts.append(f"backend{str(args.sim_backend)}")

    # Optional tags for experimental variants.
    sim_method = getattr(args, "sim_method", "default")
    if sim_method and str(sim_method) != "default":
        parts.append(f"sim_method-{sim_method}")
    frac_scheme = getattr(args, "subtype_fraction_scheme", "uniform")
    if frac_scheme and str(frac_scheme) != "uniform":
        parts.append(f"frac-{frac_scheme}")
    misdiag = float(getattr(args, "misdiagnosed_fraction", 0.0) or 0.0)
    if misdiag > 0:
        parts.append(f"misdiag{fmt_float(misdiag)}")
    seq_overlap = float(getattr(args, "sim_sequence_overlap_frac", 0.0) or 0.0)
    if seq_overlap > 0.0:
        parts.append(f"seqov{fmt_float(seq_overlap)}")
    seq_template_tau = getattr(args, "sim_sequence_template_kendall_tau", None)
    if seq_template_tau is not None:
        parts.append(f"seqtau{fmt_float(float(seq_template_tau))}")
        parts.append(
            f"seqtautol{fmt_float(float(getattr(args, 'sim_sequence_template_kendall_tolerance', 0.05)))}"
        )
        parts.append(
            f"seqtautry{int(getattr(args, 'sim_sequence_template_max_tries', 200))}"
        )
    visit_count_distribution_source = getattr(args, "_effective_visit_count_distribution_source", None)
    if visit_count_distribution_source is not None:
        vdist_map = {
            "ppmi_with_genomics": "vdist-ppmiwg",
            "ppmi_without_genomics": "vdist-ppmiong",
            "json": "vdist-json",
        }
        parts.append(vdist_map.get(str(visit_count_distribution_source), f"vdist-{visit_count_distribution_source}"))
    sigma_strategy = str(getattr(args, "zscore_sigma_strategy", "auto"))
    if sigma_strategy not in ("auto", "error"):
        parts.append(f"zsig-{sigma_strategy}")
    n_repeats = int(getattr(args, "n_repeats", 1) or 1)
    if n_repeats > 1:
        parts.append(f"rep{n_repeats}")
    cv_folds = int(getattr(args, "cv_folds", 0) or 0)
    if cv_folds > 1:
        parts.append(f"cv{cv_folds}")
    n_visits_raw = getattr(args, "n_visits_per_patient", 1)
    n_visits_text = str(n_visits_raw).lower().strip()
    n_visits_mixed = n_visits_text in ("mixed", "mixed-1")
    n_visits_mixed_drop_singletons = n_visits_text == "mixed-1"
    n_visits = None if n_visits_mixed else int(n_visits_raw or 1)
    if n_visits_mixed:
        parts.append("npmixed-1" if n_visits_mixed_drop_singletons else "npmixed")
        min_gap = getattr(args, "min_stage_gap_frac", None)
        if min_gap is not None:
            parts.append(f"gap{fmt_float(float(min_gap))}")
    elif n_visits is not None and n_visits > 1:
        parts.append(f"np{n_visits}")
        min_gap = getattr(args, "min_stage_gap_frac", None)
        if min_gap is not None:
            parts.append(f"gap{fmt_float(float(min_gap))}")
    sim_max_visits = getattr(args, "sim_max_visits_per_patient", None)
    if sim_max_visits is not None and (n_visits_mixed or int(sim_max_visits) != int(n_visits)):
        parts.append(f"max{int(sim_max_visits)}")
    visit_grouping = getattr(args, "visit_grouping", None)
    if visit_grouping and str(visit_grouping) != "subsample":
        parts.append(f"vg-{visit_grouping}")
    if bool(getattr(args, "use_longitudinal_likelihood", False)):
        parts.append("longlik")
    return "_".join(parts)


def subtype_fractions_from_scheme(
    *,
    n_subtypes: int,
    scheme: str,
    fractions: list[float] | None = None,
) -> np.ndarray:
    """Resolve subtype fractions from scheme or explicit list."""
    scheme = str(scheme).lower().strip()
    k = int(n_subtypes)
    if k <= 0:
        raise ValueError("n_subtypes must be > 0.")
    if fractions is not None:
        frac = np.asarray([float(x) for x in fractions], dtype=float)
        if frac.shape != (k,):
            raise ValueError("subtype_fractions must have shape (n_subtypes,).")
        if np.any(frac < 0) or not np.isclose(frac.sum(), 1.0):
            raise ValueError("subtype_fractions must be non-negative and sum to 1.")
        return frac
    if scheme == "uniform":
        return np.full(k, 1.0 / k, dtype=float)
    if scheme == "paper":
        weights = np.arange(k + 1, 1, -1, dtype=float)
        return weights / np.sum(weights)
    raise ValueError(f"Unknown subtype_fraction_scheme: {scheme}")


def derive_zscore_from_ordinal(
    *,
    X_obs: np.ndarray,
    stage_true: np.ndarray,
    score_vals: np.ndarray,
    min_sigma: float = 1e-6,
    ddof: int = 1,
    sigma_strategy: str = "error",
) -> dict:
    """
    Derive z-score inputs using stage-0 subjects as a baseline.
    sigma_strategy controls how near-zero baseline std values are handled.
    """
    X_obs = np.asarray(X_obs, dtype=float)
    stage_true = np.asarray(stage_true, dtype=int)
    score_vals = np.asarray(score_vals, dtype=float)
    if X_obs.ndim != 2:
        raise ValueError("X_obs must be 2D.")
    if stage_true.ndim != 1 or stage_true.shape[0] != X_obs.shape[0]:
        raise ValueError("stage_true must be 1D and match X_obs rows.")
    if score_vals.ndim != 2 or score_vals.shape[0] != X_obs.shape[1]:
        raise ValueError("score_vals must be 2D with n_biomarkers rows.")

    # Use stage-0 subjects as a baseline reference.
    baseline_mask = stage_true == 0
    if not np.any(baseline_mask):
        raise ValueError("No stage-0 subjects available for baseline z-scoring.")

    # Here we compute baseline mean/std for each biomarker.
    baseline_obs = X_obs[baseline_mask]
    mu = np.mean(baseline_obs, axis=0)
    sigma = np.std(baseline_obs, axis=0, ddof=int(ddof))
    if not np.all(np.isfinite(sigma)):
        raise ValueError("Baseline std has non-finite values.")
    min_sigma = float(min_sigma)
    sigma_strategy = str(sigma_strategy).lower().strip()
    if sigma_strategy not in ("error", "floor", "global"):
        raise ValueError("sigma_strategy must be one of: error, floor, global.")

    small = sigma < min_sigma
    sigma_replaced_count = 0
    sigma_replacement = None
    if np.any(small):
        if sigma_strategy == "error":
            idx = np.where(small)[0]
            raise ValueError(f"{idx.size} baseline std values < min_sigma. First indices: {idx[:10].tolist()}")
        if sigma_strategy == "floor":
            sigma = sigma.copy()
            sigma[small] = min_sigma
            sigma_replaced_count = int(np.sum(small))
            sigma_replacement = float(min_sigma)
        elif sigma_strategy == "global":
            global_sigma = float(np.std(baseline_obs, ddof=int(ddof)))
            if not np.isfinite(global_sigma) or global_sigma < min_sigma:
                raise ValueError("Global baseline std is non-finite or < min_sigma.")
            sigma = sigma.copy()
            sigma[small] = global_sigma
            sigma_replaced_count = int(np.sum(small))
            sigma_replacement = global_sigma

    # Transform scores into z-space for fitting.
    Z_pd = (X_obs - mu) / sigma
    Z_vals = (score_vals - mu[:, None]) / sigma[:, None]
    Z_max = (np.max(score_vals, axis=1) - mu) / sigma
    return {
        "Z_pd": Z_pd.astype(np.float32),
        "Z_vals": Z_vals.astype(np.float32),
        "Z_max": Z_max.astype(np.float32),
        "mu": mu.astype(np.float32),
        "sigma": sigma.astype(np.float32),
        "sigma_strategy": sigma_strategy,
        "sigma_replaced_count": int(sigma_replaced_count),
        "sigma_replacement": sigma_replacement,
    }


def make_cv_folds(n_samples: int, n_folds: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Create random CV test folds (indices only)."""
    n_samples = int(n_samples)
    n_folds = int(n_folds)
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")
    if n_folds <= 1:
        raise ValueError("n_folds must be > 1.")
    indices = rng.permutation(n_samples)
    folds = np.array_split(indices, n_folds)
    return [np.asarray(f, dtype=int) for f in folds]


def loglike_ci(samples_likelihood: np.ndarray, alpha: float = 0.05) -> dict:
    """Compute a symmetric (1-alpha) CI from likelihood samples."""
    samples = np.asarray(samples_likelihood, dtype=float).ravel()
    if samples.size == 0:
        raise ValueError("samples_likelihood is empty.")
    lower = float(np.quantile(samples, alpha / 2))
    upper = float(np.quantile(samples, 1.0 - alpha / 2))
    return {
        "mean": float(np.mean(samples)),
        "lower": lower,
        "upper": upper,
    }


def ci_overlap(ci_a: dict, ci_b: dict) -> bool:
    """Return True if two CIs overlap."""
    return not (ci_a["upper"] < ci_b["lower"] or ci_b["upper"] < ci_a["lower"])


def aggregate_numeric_dicts(dicts: list[dict]) -> dict:
    """Aggregate numeric keys into mean/std (ignores non-numeric)."""
    if not dicts:
        return {}
    keys = set().union(*[d.keys() for d in dicts])
    out = {}
    for key in keys:
        vals = []
        for d in dicts:
            val = d.get(key)
            if isinstance(val, (int, float, np.number)) and np.isfinite(val):
                vals.append(float(val))
        if vals:
            arr = np.asarray(vals, dtype=float)
            out[key] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
    return out

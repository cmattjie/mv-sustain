from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .progress_log import ProgressLog
from .sustain_utils import SustainRunner

_THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _normalize_test_idxs(test_idxs: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(fold, dtype=np.int32) for fold in test_idxs]


def _prepare_parallel_output_dir(output_folder: str) -> None:
    output_path = Path(str(output_folder))
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "pickle_files").mkdir(parents=True, exist_ok=True)


def _run_classic_cvic_fold_worker(
    *,
    fold_idx: int,
    runner_kwargs: Mapping[str, Any],
    fit_inputs: Any,
    fit_kwargs: Mapping[str, Any],
    patient_ids: Sequence[object] | None,
    test_idxs: Sequence[np.ndarray],
    base_seed: int,
    fold_log_dir: str | None = None,
) -> int:
    fold_log = None if fold_log_dir is None else ProgressLog(Path(fold_log_dir) / f"fold_{int(fold_idx) + 1:02d}.log")
    for key, value in _THREAD_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)

    local_runner_kwargs = dict(runner_kwargs)
    local_runner_kwargs["seed"] = int(base_seed) + int(fold_idx)
    local_runner_kwargs["use_parallel_startpoints"] = False

    if fold_log is not None:
        fold_log.log(
            f"start fold={int(fold_idx) + 1} seed={int(local_runner_kwargs['seed'])}"
        )

    try:
        runner = SustainRunner(**local_runner_kwargs)
        runner.initialize_model(
            fit_inputs,
            patient_ids=patient_ids,
            **dict(fit_kwargs),
        )
        runner.cross_validate(
            test_idxs=_normalize_test_idxs(test_idxs),
            select_fold=[int(fold_idx)],
            plot=False,
        )
    except Exception as exc:
        if fold_log is not None:
            fold_log.log(f"failed fold={int(fold_idx) + 1} error={type(exc).__name__}: {exc}")
        raise

    if fold_log is not None:
        fold_log.log(f"completed fold={int(fold_idx) + 1}")
    return int(fold_idx)


def run_classic_cvic(
    *,
    runner: SustainRunner,
    runner_kwargs: Mapping[str, Any],
    fit_inputs: Any,
    fit_kwargs: Mapping[str, Any],
    patient_ids: Sequence[object] | None,
    test_idxs: Sequence[np.ndarray],
    base_seed: int,
    cv_parallel_folds: int,
    progress_callback: Callable[[int, int], None] | None = None,
    fold_log_dir: str | Path | None = None,
) -> tuple[Any, Any]:
    normalized_test_idxs = _normalize_test_idxs(test_idxs)
    total_folds = len(normalized_test_idxs)
    effective_workers = min(len(normalized_test_idxs), max(1, int(cv_parallel_folds)))
    if effective_workers <= 1 or len(normalized_test_idxs) <= 1:
        result = runner.cross_validate(test_idxs=normalized_test_idxs, plot=False)
        if progress_callback is not None and total_folds > 0:
            progress_callback(total_folds, total_folds)
        return result

    output_folder = getattr(runner, "output_folder", None)
    if output_folder is None:
        output_folder = runner_kwargs.get("out_pickle_folder", runner_kwargs.get("output_folder"))
    if output_folder is None:
        raise ValueError("Classic CVIC parallel helper requires a runner output folder.")
    _prepare_parallel_output_dir(str(output_folder))

    worker_kwargs = dict(runner_kwargs)
    worker_kwargs.pop("seed", None)

    with ProcessPoolExecutor(
        max_workers=effective_workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = [
            executor.submit(
                _run_classic_cvic_fold_worker,
                fold_idx=int(fold_idx),
                runner_kwargs=worker_kwargs,
                fit_inputs=fit_inputs,
                fit_kwargs=dict(fit_kwargs),
                patient_ids=None if patient_ids is None else list(patient_ids),
                test_idxs=normalized_test_idxs,
                base_seed=int(base_seed),
                fold_log_dir=None if fold_log_dir is None else str(fold_log_dir),
            )
            for fold_idx in range(len(normalized_test_idxs))
        ]
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total_folds)

    return runner.cross_validate(test_idxs=normalized_test_idxs, plot=False)

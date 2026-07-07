"""
Optimized wrappers around pySuStaIn ZscoreSustain.
"""

from __future__ import annotations

import numpy as np
import warnings
from functools import partial

from pySuStaIn.ZscoreSustain import ZscoreSustain as _ZscoreSustain  # type: ignore
from .gpu_utils import get_torch_device


class ZscoreSustain(_ZscoreSustain):
    """Z-score SuStaIn with reduced redundant recomputation."""

    def __init__(
        self,
        data,
        Z_vals,
        Z_max,
        biomarker_labels,
        N_startpoints,
        N_S_max,
        N_iterations_MCMC,
        output_folder,
        dataset_name,
        use_parallel_startpoints,
        seed=None,
        *,
        sigma_noise: float = 1.0,
        sigma_mode: str = "fixed",
        cov=None,
        cov_blocks=None,
        cov_kind: str = "abs",
        cov_scale: float | None = None,
        cov_ridge: float = 1e-6,
        cov_mode: str | None = None,
        missing_policy: str = "error",
    ) -> None:
        super().__init__(
            data,
            Z_vals,
            Z_max,
            biomarker_labels,
            N_startpoints,
            N_S_max,
            N_iterations_MCMC,
            output_folder,
            dataset_name,
            use_parallel_startpoints,
            seed=seed,
        )
        # Override pySuStaIn's hardcoded std_biomarker_zscore=[1]*N so the likelihood
        # uses the same noise level as the data simulation.
        self.std_biomarker_zscore = [float(sigma_noise)] * int(
            self._ZscoreSustain__sustainData.getNumBiomarkers()
        )

        sigma_mode = str(sigma_mode).lower().strip()
        if sigma_mode not in ("fixed", "global", "per_biomarker"):
            raise ValueError("sigma_mode must be one of: fixed, global, per_biomarker.")
        self.sigma_mode = sigma_mode

        self.missing_policy = str(missing_policy).lower().strip()
        if self.missing_policy not in ("error", "skip", "uniform", "marginal"):
            raise ValueError("missing_policy must be one of: error, skip, uniform, marginal.")

        self._mcmc_opt_iters = max(1, int(N_iterations_MCMC))

        self.cov_mode = "diag"
        self.cov_blocks = None
        self._cov_cache = None
        self._cov_kind = str(cov_kind).lower().strip()
        self._cov_scale = cov_scale
        self._cov_ridge = float(cov_ridge)

        if cov is None:
            if cov_mode not in (None, "diag"):
                raise ValueError("cov_mode specified but cov is None.")
            return

        if cov_mode is None:
            cov_mode = "block" if cov_blocks is not None else "full"
        cov_mode = str(cov_mode).lower().strip()
        if cov_mode not in ("diag", "full", "block"):
            raise ValueError("cov_mode must be one of: diag, full, block.")
        if cov_mode == "diag":
            self.cov_mode = "diag"
            return

        cov = np.asarray(cov, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError(f"cov must be square 2D, got {cov.shape}.")
        n_biomarkers = int(self._ZscoreSustain__sustainData.getNumBiomarkers())
        if cov.shape[0] != n_biomarkers:
            raise ValueError("cov shape must match number of biomarkers.")

        if self._cov_kind not in ("abs", "corr"):
            raise ValueError("cov_kind must be 'abs' or 'corr'.")
        if self._cov_kind == "corr":
            scale = 1.0 if self._cov_scale is None else float(self._cov_scale)
            cov = cov * (scale ** 2)

        if cov_mode == "full":
            blocks = [list(range(n_biomarkers))]
        else:
            if cov_blocks is None:
                raise ValueError("cov_blocks must be provided for cov_mode='block'.")
            blocks = []
            seen = set()
            for block in cov_blocks:
                clean = []
                for idx in block:
                    ii = int(idx)
                    if ii < 0 or ii >= n_biomarkers:
                        raise ValueError(f"cov_block index {ii} out of range.")
                    if ii in seen:
                        raise ValueError(f"cov_block index {ii} appears in multiple blocks.")
                    seen.add(ii)
                    clean.append(ii)
                if len(clean) > 0:
                    blocks.append(clean)
            for i in range(n_biomarkers):
                if i not in seen:
                    blocks.append([i])

        self.cov_mode = cov_mode
        self.cov_blocks = blocks
        self._cov_cache = self._prepare_cov_cache(cov, blocks, self._cov_ridge)

    def _compute_stage_value(self, S: np.ndarray) -> np.ndarray:
        """Return stage_value (n_biomarkers, n_stages+1) for sequence S.

        Extracted from _calculate_likelihood_stage so sigma update can use expected
        values without rerunning the full likelihood computation.
        """
        N = self.stage_biomarker_index.shape[1]
        S_inv = np.array([0] * N)
        S_inv[S.astype(int)] = np.arange(N)
        possible_biomarkers = np.unique(self.stage_biomarker_index)
        B = len(possible_biomarkers)
        point_value = np.zeros((B, N + 2))
        arange_N = np.arange(N + 2)
        for i in range(B):
            b = possible_biomarkers[i]
            event_location = np.concatenate([[0], S_inv[(self.stage_biomarker_index == b)[0]], [N]])
            event_value = np.concatenate(
                [[self.min_biomarker_zscore[i]], self.stage_zscore[self.stage_biomarker_index == b], [self.max_biomarker_zscore[i]]]
            )
            for j in range(len(event_location) - 1):
                if j == 0:
                    temp = arange_N[event_location[j] : (event_location[j + 1] + 2)]
                    N_j = event_location[j + 1] - event_location[j] + 2
                    point_value[i, temp] = ZscoreSustain.linspace_local2(
                        event_value[j], event_value[j + 1], N_j, arange_N[0:N_j]
                    )
                else:
                    temp = arange_N[(event_location[j] + 1) : (event_location[j + 1] + 2)]
                    N_j = event_location[j + 1] - event_location[j] + 1
                    point_value[i, temp] = ZscoreSustain.linspace_local2(
                        event_value[j], event_value[j + 1], N_j, arange_N[0:N_j]
                    )
        return 0.5 * point_value[:, :-1] + 0.5 * point_value[:, 1:]

    def _update_sigma(self, data: np.ndarray, S_opt: np.ndarray, p_perm_k_norm: np.ndarray) -> None:
        """M-step update of std_biomarker_zscore from responsibility-weighted residuals.

        p_perm_k_norm[i, k, s] = p(stage=k, subtype=s | data_i), shape (M, N+1, N_S).
        Updates self.std_biomarker_zscore in place.
        NaN positions in data contribute 0 to numerator and denominator.

        Future upgrade: propagate sigma uncertainty through MCMC by treating sigma as a
        sampled parameter with a Half-Normal or Inverse-Gamma prior in _perform_mcmc.
        This plug-in approach (sigma fixed at EM point estimate during MCMC) does not
        propagate sigma posterior variance to sequence/subtype uncertainty.
        """
        if self.sigma_mode == "fixed":
            return

        data = np.asarray(data, dtype=float)
        M, B = data.shape[0], data.shape[1]
        N_S = S_opt.shape[0]

        valid = ~np.isnan(data)                          # (M, B)
        data_clean = np.where(valid, data, 0.0)          # NaN -> 0 residual

        if self.sigma_mode == "global":
            num = 0.0
            den = 0.0
            for s in range(N_S):
                sv = self._compute_stage_value(S_opt[s])             # (B, N+1)
                rs = p_perm_k_norm[:, :, s]                          # (M, N+1)
                delta = data_clean[:, :, None] - sv[None, :, :]      # (M, B, N+1)
                delta_sq = np.where(valid[:, :, None], delta ** 2, 0.0)
                # Σ_{i,k} rs[i,k] * Σ_j delta_sq[i,j,k]
                num += float(np.sum(rs * np.sum(delta_sq, axis=1)))
                # Σ_{i,k} rs[i,k] * n_valid_biomarkers[i]
                den += float(np.sum(rs * valid.sum(axis=1, keepdims=True).astype(float)))
            if den < 1e-12:
                return
            sigma = float(np.sqrt(max(num / den, 1e-4)))
            self.std_biomarker_zscore = [sigma] * B

        elif self.sigma_mode == "per_biomarker":
            num_j = np.zeros(B, dtype=float)
            den_j = np.zeros(B, dtype=float)
            for s in range(N_S):
                sv = self._compute_stage_value(S_opt[s])             # (B, N+1)
                rs = p_perm_k_norm[:, :, s]                          # (M, N+1)
                delta = data_clean[:, :, None] - sv[None, :, :]      # (M, B, N+1)
                delta_sq = np.where(valid[:, :, None], delta ** 2, 0.0)
                # For each j: Σ_{i,k} rs[i,k] * valid[i,j] * delta_sq[i,j,k]
                num_j += np.sum(rs[:, None, :] * delta_sq, axis=(0, 2))
                den_j += np.sum(rs[:, None, :] * valid[:, :, None].astype(float), axis=(0, 2))
            den_j = np.maximum(den_j, 1e-12)
            sigma_j = np.sqrt(np.maximum(num_j / den_j, 0.01))      # floor at 0.1 std
            self.std_biomarker_zscore = list(sigma_j)

    def _find_ml_iteration(self, sustainData, seed_seq):
        """Override to return sigma as 4th element so _find_ml can propagate it."""
        rng = np.random.default_rng(seed_seq)
        seq_init = self._initialise_sequence(sustainData, rng)
        f_init = [1]
        this_ml_sequence, this_ml_f, this_ml_likelihood, _, _, _ = self._perform_em(
            sustainData, seq_init, f_init, rng
        )
        return this_ml_sequence, this_ml_f, this_ml_likelihood, list(self.std_biomarker_zscore)

    def _find_ml(self, sustainData):
        """Override to propagate sigma from the best EM startpoint back to the main model.

        The parent _find_ml runs startpoints via pool.map() (separate worker processes).
        Each worker calls _optimise_parameters → _update_sigma, updating self.std_biomarker_zscore
        in the worker's copy only. That update is lost when the worker exits. MCMC then runs
        in the main process with the original sigma (typically 1.0). This override captures
        the sigma returned by our _find_ml_iteration override and applies the winning
        startpoint's sigma to the main model before returning.
        """
        if self.sigma_mode == 'fixed':
            return super()._find_ml(sustainData)

        partial_iter = partial(self._find_ml_iteration, sustainData)
        seed_sequences = np.random.SeedSequence(self.global_rng.integers(1e10))
        pool_output_list = self.pool.map(partial_iter, seed_sequences.spawn(self.N_startpoints))

        if not isinstance(pool_output_list, list):
            pool_output_list = list(pool_output_list)

        ml_sequence_mat = np.zeros((1, sustainData.getNumStages(), self.N_startpoints))
        ml_f_mat = np.zeros((1, self.N_startpoints))
        ml_likelihood_mat = np.zeros(self.N_startpoints)

        for i in range(self.N_startpoints):
            ml_sequence_mat[:, :, i] = pool_output_list[i][0]
            ml_f_mat[:, i] = pool_output_list[i][1]
            ml_likelihood_mat[i] = pool_output_list[i][2]

        ix = int(np.argmax(ml_likelihood_mat))
        ml_sequence = ml_sequence_mat[:, :, ix]
        ml_f = ml_f_mat[:, ix]
        ml_likelihood = ml_likelihood_mat[ix]

        # Apply sigma from the winning startpoint to the main process model.
        self.std_biomarker_zscore = list(pool_output_list[ix][3])

        return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat

    def _find_ml_split_iteration(self, sustainData, seed_seq):
        """Override to return sigma as 4th element so _find_ml_split can propagate it."""
        rng = np.random.default_rng(seed_seq)
        N_S = 2
        min_N_cluster = 0
        while min_N_cluster == 0:
            vals = rng.random(sustainData.getNumSamples())
            cluster_assignment = np.ceil(N_S * vals).astype(int)
            cluster_sizes = np.bincount(cluster_assignment, minlength=3)[1:]
            min_N_cluster = cluster_sizes.min()

        seq_init = np.zeros((N_S, sustainData.getNumStages()))
        for s in range(N_S):
            index_s = cluster_assignment.reshape(cluster_assignment.shape[0],) == (s + 1)
            temp_sustainData = sustainData.reindex(index_s)
            temp_seq_init = self._initialise_sequence(sustainData, rng)
            seq_init[s, :], _, _, _, _, _ = self._perform_em(temp_sustainData, temp_seq_init, [1], rng)

        f_init = np.array([1.] * N_S) / float(N_S)
        this_ml_sequence, this_ml_f, this_ml_likelihood, _, _, _ = self._perform_em(
            sustainData, seq_init, f_init, rng
        )
        return this_ml_sequence, this_ml_f, this_ml_likelihood, list(self.std_biomarker_zscore)

    def _find_ml_split(self, sustainData):
        """Override to propagate sigma from the best split startpoint back to the main model."""
        if self.sigma_mode == 'fixed':
            return super()._find_ml_split(sustainData)

        N_S = 2
        partial_iter = partial(self._find_ml_split_iteration, sustainData)
        seed_sequences = np.random.SeedSequence(self.global_rng.integers(1e10))
        pool_output_list = self.pool.map(partial_iter, seed_sequences.spawn(self.N_startpoints))

        if not isinstance(pool_output_list, list):
            pool_output_list = list(pool_output_list)

        ml_sequence_mat = np.zeros((N_S, sustainData.getNumStages(), self.N_startpoints))
        ml_f_mat = np.zeros((N_S, self.N_startpoints))
        ml_likelihood_mat = np.zeros((self.N_startpoints, 1))

        for i in range(self.N_startpoints):
            ml_sequence_mat[:, :, i] = pool_output_list[i][0]
            ml_f_mat[:, i] = pool_output_list[i][1]
            ml_likelihood_mat[i] = pool_output_list[i][2]

        ix = [np.where(ml_likelihood_mat == max(ml_likelihood_mat))[0][0]]
        ml_sequence = ml_sequence_mat[:, :, ix]
        ml_f = ml_f_mat[:, ix]
        ml_likelihood = ml_likelihood_mat[ix]

        # Apply sigma from the winning startpoint to the main process model.
        self.std_biomarker_zscore = list(pool_output_list[ix[0]][3])

        return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat

    def _optimise_mcmc_settings(self, sustainData, seq_init, f_init):
        n_iterations_MCMC_optimisation = int(self._mcmc_opt_iters)
        if n_iterations_MCMC_optimisation < 1:
            n_iterations_MCMC_optimisation = 1

        n_passes_optimisation = 3
        seq_sigma_currentpass = 1
        f_sigma_currentpass = 0.01

        N_S = seq_init.shape[0]

        for _ in range(n_passes_optimisation):
            _, _, _, samples_sequence_currentpass, samples_f_currentpass, _ = self._perform_mcmc(
                sustainData,
                seq_init,
                f_init,
                n_iterations_MCMC_optimisation,
                seq_sigma_currentpass,
                f_sigma_currentpass,
            )

            samples_position_currentpass = np.zeros(samples_sequence_currentpass.shape)
            for s in range(N_S):
                for sample in range(n_iterations_MCMC_optimisation):
                    temp_seq = samples_sequence_currentpass[s, :, sample]
                    temp_inv = np.array([0] * samples_sequence_currentpass.shape[1])
                    temp_inv[temp_seq.astype(int)] = np.arange(samples_sequence_currentpass.shape[1])
                    samples_position_currentpass[s, :, sample] = temp_inv

            seq_sigma_currentpass = np.std(samples_position_currentpass, axis=2, ddof=1)
            seq_sigma_currentpass[seq_sigma_currentpass < 0.01] = 0.01
            f_sigma_currentpass = np.std(samples_f_currentpass, axis=1, ddof=1)

        return seq_sigma_currentpass, f_sigma_currentpass

    @staticmethod
    def _cholesky_with_ridge(sigma: np.ndarray, *, ridge: float, max_tries: int = 6) -> tuple[np.ndarray, float]:
        ridge = float(ridge)
        if ridge < 0:
            ridge = 0.0
        jitter = ridge if ridge > 0 else 0.0
        eye = np.eye(int(sigma.shape[0]), dtype=float)
        for _ in range(int(max_tries)):
            try:
                return np.linalg.cholesky(sigma + jitter * eye), jitter
            except np.linalg.LinAlgError:
                jitter = 10.0 * (jitter if jitter > 0 else 1e-12)
        raise ValueError("Cholesky failed after multiple ridge attempts.")

    def _prepare_cov_cache(self, cov: np.ndarray, blocks: list[list[int]], cov_ridge: float) -> list[dict]:
        cache: list[dict] = []
        for block in blocks:
            idx = np.asarray(block, dtype=int)
            sigma_block = cov[np.ix_(idx, idx)]
            L, jitter = self._cholesky_with_ridge(sigma_block, ridge=float(cov_ridge))
            sigma_eff = sigma_block + float(jitter) * np.eye(int(idx.size), dtype=float)
            logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
            cache.append({"idx": idx, "L": L, "logdet": logdet, "d": int(idx.size), "Sigma": sigma_eff})
        return cache

    def _calculate_likelihood_stage(self, sustainData, S):
        """
        Compute likelihood of a single subtype sequence.

        - diag mode: assumes conditional independence across biomarkers.
        - full/block mode: multivariate normal with correlated biomarkers.
        """
        N = self.stage_biomarker_index.shape[1]
        S_inv = np.array([0] * N)
        S_inv[S.astype(int)] = np.arange(N)
        possible_biomarkers = np.unique(self.stage_biomarker_index)
        B = len(possible_biomarkers)
        point_value = np.zeros((B, N + 2))

        arange_N = np.arange(N + 2)

        for i in range(B):
            b = possible_biomarkers[i]
            event_location = np.concatenate([[0], S_inv[(self.stage_biomarker_index == b)[0]], [N]])
            event_value = np.concatenate(
                [[self.min_biomarker_zscore[i]], self.stage_zscore[self.stage_biomarker_index == b], [self.max_biomarker_zscore[i]]]
            )
            for j in range(len(event_location) - 1):
                if j == 0:
                    temp = arange_N[event_location[j] : (event_location[j + 1] + 2)]
                    N_j = event_location[j + 1] - event_location[j] + 2
                    point_value[i, temp] = ZscoreSustain.linspace_local2(
                        event_value[j], event_value[j + 1], N_j, arange_N[0:N_j]
                    )
                else:
                    temp = arange_N[(event_location[j] + 1) : (event_location[j + 1] + 2)]
                    N_j = event_location[j + 1] - event_location[j] + 1
                    point_value[i, temp] = ZscoreSustain.linspace_local2(
                        event_value[j], event_value[j + 1], N_j, arange_N[0:N_j]
                    )

        stage_value = 0.5 * point_value[:, : point_value.shape[1] - 1] + 0.5 * point_value[:, 1:]

        M = sustainData.getNumSamples()
        p_perm_k = np.zeros((M, N + 1))
        data = np.asarray(sustainData.data, dtype=float)

        if self.cov_mode == "diag":
            if self.missing_policy == "error":
                if np.isnan(data).any():
                    raise ValueError("missing_policy='error' does not allow NaNs in sustainData.data.")

                sigmat = np.array(self.std_biomarker_zscore)
                factor = np.log(1.0 / (np.sqrt(np.pi * 2.0) * sigmat))
                coeff = np.log(1.0 / float(N + 1))
                x = (data[:, :, None] - stage_value) / sigmat[None, :, None]
                p_perm_k = coeff + np.sum(factor[None, :, None] - 0.5 * np.square(x), 1)
                p_perm_k = np.exp(p_perm_k)
                return p_perm_k

            if self.missing_policy in ("skip", "marginal"):
                sigmat = np.array(self.std_biomarker_zscore)
                factor = np.log(1.0 / (np.sqrt(np.pi * 2.0) * sigmat))
                coeff = np.log(1.0 / float(N + 1))
                mask = ~np.isnan(data)
                x = (data[:, :, None] - stage_value) / sigmat[None, :, None]
                x = np.where(mask[:, :, None], x, 0.0)
                factor_masked = factor[None, :, None] * mask[:, :, None]
                p_perm_k = coeff + np.sum(factor_masked - 0.5 * np.square(x), 1)
                p_perm_k = np.exp(p_perm_k)
                return p_perm_k

            if self.missing_policy == "uniform":
                # Matches pySuStaIn ZScoreSustainMissingData behavior.
                p_missingdata = np.ones((1, B)) / (np.asarray(self.max_biomarker_zscore) - np.asarray(self.min_biomarker_zscore))
                p_missingdata = np.tile(p_missingdata, (M, 1))
                sigmat = np.tile(self.std_biomarker_zscore, (M, 1))
                factor = np.log(1.0 / (np.sqrt(np.pi * 2.0) * sigmat))
                coeff = np.log(1.0 / float(N + 1))

                stage_value_tiled = np.tile(stage_value, (M, 1))
                N_biomarkers = stage_value.shape[0]
                for j in range(N + 1):
                    stage_value_tiled_j = stage_value_tiled[:, j].reshape(M, N_biomarkers)
                    x_hasdata = (data - stage_value_tiled_j) / sigmat
                    p = np.log(p_missingdata)
                    p[~np.isnan(data)] = x_hasdata[~np.isnan(data)]
                    p_perm_k[:, j] = coeff + np.sum(factor - 0.5 * np.square(p), 1)
                p_perm_k = np.exp(p_perm_k)
                return p_perm_k

            raise ValueError("missing_policy must be one of: error, skip, uniform, marginal.")

        if np.isnan(data).any():
            if self.missing_policy == "error":
                raise ValueError("missing_policy='error' does not allow NaNs in sustainData.data.")
            if self.missing_policy == "skip":
                warnings.warn(
                    "missing_policy='skip' with correlated likelihood performs exact MVN marginalization. "
                    "Consider using missing_policy='marginal' for clarity.",
                    RuntimeWarning,
                )
        else:
            coeff = np.log(1.0 / float(N + 1))
            log2pi = np.log(2.0 * np.pi)
            logp = np.zeros((M, N + 1), dtype=float)
            for block in self._cov_cache:
                idx = block["idx"]
                L = block["L"]
                logdet = block["logdet"]
                d = block["d"]
                X_block = data[:, idx]
                mu_block = stage_value[idx, :]
                delta = X_block[:, :, None] - mu_block[None, :, :]
                delta_2d = np.transpose(delta, (1, 0, 2)).reshape(int(d), M * (N + 1))
                y = np.linalg.solve(L, delta_2d)
                quad = np.sum(y * y, axis=0).reshape(M, N + 1)
                logp += -0.5 * (quad + logdet + d * log2pi)

            p_perm_k = np.exp(logp + coeff)
            return p_perm_k
        # Missing data: exact MVN marginalization over observed dimensions.
        coeff = np.log(1.0 / float(N + 1))
        log2pi = np.log(2.0 * np.pi)
        logp = np.zeros((M, N + 1), dtype=float)
        if self.missing_policy == "uniform":
            p_missing = 1.0 / (np.asarray(self.max_biomarker_zscore) - np.asarray(self.min_biomarker_zscore))
            log_p_missing = np.log(p_missing)
        else:
            log_p_missing = None

        for block in self._cov_cache:
            idx = block["idx"]
            X_block = data[:, idx]
            mask_block = ~np.isnan(X_block)
            if np.all(mask_block):
                L = block["L"]
                logdet = block["logdet"]
                d = block["d"]
                mu_block = stage_value[idx, :]
                delta = X_block[:, :, None] - mu_block[None, :, :]
                delta_2d = np.transpose(delta, (1, 0, 2)).reshape(int(d), M * (N + 1))
                y = np.linalg.solve(L, delta_2d)
                quad = np.sum(y * y, axis=0).reshape(M, N + 1)
                logp += -0.5 * (quad + logdet + d * log2pi)
                continue

            rows_by_mask: dict[tuple[bool, ...], list[int]] = {}
            for r in range(M):
                key = tuple(mask_block[r].tolist())
                rows_by_mask.setdefault(key, []).append(r)

            sigma_block = block["Sigma"]
            for key, rows in rows_by_mask.items():
                obs_mask = np.array(key, dtype=bool)
                d_obs = int(np.sum(obs_mask))
                if d_obs == 0:
                    if log_p_missing is not None:
                        miss_penalty = float(np.sum(log_p_missing[idx]))
                        logp[rows, :] += miss_penalty
                    continue

                sigma_obs = sigma_block[np.ix_(obs_mask, obs_mask)]
                L_obs, _ = self._cholesky_with_ridge(sigma_obs, ridge=float(self._cov_ridge))
                logdet_obs = 2.0 * float(np.sum(np.log(np.diag(L_obs))))

                X_obs = X_block[rows][:, obs_mask]
                mu_obs = stage_value[idx[obs_mask], :]
                delta = X_obs[:, :, None] - mu_obs[None, :, :]
                delta_2d = np.transpose(delta, (1, 0, 2)).reshape(d_obs, len(rows) * (N + 1))
                y = np.linalg.solve(L_obs, delta_2d)
                quad = np.sum(y * y, axis=0).reshape(len(rows), N + 1)
                logp_block = -0.5 * (quad + logdet_obs + d_obs * log2pi)
                logp[rows, :] += logp_block

                if log_p_missing is not None and d_obs < int(idx.size):
                    miss_penalty = float(np.sum(log_p_missing[idx[~obs_mask]]))
                    logp[rows, :] += miss_penalty

        p_perm_k = np.exp(logp + coeff)
        return p_perm_k

    def _calculate_likelihood_stage_batch(self, sustainData, sequences):
        """Compute _calculate_likelihood_stage for multiple sequences simultaneously.

        sequences: (n_pos, N) array-like; each row is a full event ordering.
        Returns: (M, N+1, n_pos) array.

        Replaces the per-position serial loop from the original pySuStaIn
        _optimise_parameters. Instead of calling _calculate_likelihood_stage
        once per candidate position (Python loop, ~3-4 ms each), all positions
        are batched into a single (M, n_pos, B, N+1) broadcast — see
        docs/em_position_batch_optimization.md for rationale and benchmarks.

        Fast path covers diag cov_mode with error/skip/marginal missing policies.
        Falls back to serial for uniform missing policy or non-diag cov modes.
        """
        sequences = np.asarray(sequences)
        n_pos = len(sequences)
        if n_pos == 1:
            return self._calculate_likelihood_stage(sustainData, sequences[0])[:, :, np.newaxis]

        if self.cov_mode == "diag" and self.missing_policy in ("error", "skip", "marginal"):
            stage_values_all = np.stack(
                [self._compute_stage_value(seq) for seq in sequences], axis=0
            )  # (n_pos, B, N+1)
            data = np.asarray(sustainData.data, dtype=float)  # (M, B)
            sigmat = np.array(self.std_biomarker_zscore)  # (B,)
            factor = np.log(1.0 / (np.sqrt(np.pi * 2.0) * sigmat))
            N = self.stage_biomarker_index.shape[1]
            coeff = np.log(1.0 / float(N + 1))

            if self.missing_policy in ("skip", "marginal"):
                if not hasattr(self, '_nan_mask') or self._nan_mask.shape != data.shape:
                    self._nan_mask = ~np.isnan(data)

            _torch, _device = get_torch_device()
            if _torch is not None:
                _sigma_key = tuple(self.std_biomarker_zscore)
                _n_samples = data.shape[0]
                if not hasattr(self, '_gpu_cache') or self._gpu_cache.get('device') != _device or self._gpu_cache.get('sigma_key') != _sigma_key or self._gpu_cache.get('n_samples') != _n_samples:
                    self._gpu_cache = {
                        'device': _device,
                        'sigma_key': _sigma_key,
                        'n_samples': _n_samples,
                        't_data':   _torch.as_tensor(data,   dtype=_torch.float64, device=_device),
                        't_sigmat': _torch.as_tensor(sigmat, dtype=_torch.float64, device=_device),
                        't_factor': _torch.log(1.0 / (_torch.sqrt(
                                        _torch.tensor(2.0 * _torch.pi, dtype=_torch.float64, device=_device)
                                    ) * _torch.as_tensor(sigmat, dtype=_torch.float64, device=_device))),
                    }
                t_data   = self._gpu_cache['t_data']
                t_sigmat = self._gpu_cache['t_sigmat']
                t_factor = self._gpu_cache['t_factor']
                t_sv = _torch.as_tensor(stage_values_all, dtype=_torch.float64, device=_device)
                x = (t_data[:, None, :, None] - t_sv[None]) / t_sigmat[None, None, :, None]
                if self.missing_policy in ("skip", "marginal"):
                    t_nan_mask = _torch.as_tensor(self._nan_mask, device=_device)
                    x = _torch.where(t_nan_mask[:, None, :, None], x, _torch.zeros_like(x))
                    factor_bcast = t_factor[None, None, :, None] * t_nan_mask[:, None, :, None].to(_torch.float64)
                else:
                    factor_bcast = t_factor[None, None, :, None]
                p = _torch.exp(coeff + _torch.sum(factor_bcast - 0.5 * x.square(), dim=2))
                return p.permute(0, 2, 1).cpu().numpy()

            # x: (M, n_pos, B, N+1)
            x = (data[:, None, :, None] - stage_values_all[None, :, :, :]) / sigmat[None, None, :, None]
            if self.missing_policy in ("skip", "marginal"):
                x = np.where(self._nan_mask[:, None, :, None], x, 0.0)
                factor_bcast = factor[None, None, :, None] * self._nan_mask[:, None, :, None]
            else:
                factor_bcast = factor[None, None, :, None]
            # sum over B (axis=2): (M, n_pos, N+1)
            p = np.exp(coeff + np.sum(factor_bcast - 0.5 * np.square(x), axis=2))
            return p.transpose(0, 2, 1)  # (M, N+1, n_pos)

        # Fall back to serial for uniform missing policy or non-diag cov modes
        M = sustainData.getNumSamples()
        N = self.stage_biomarker_index.shape[1]
        result = np.zeros((M, N + 1, n_pos))
        for i, seq in enumerate(sequences):
            result[:, :, i] = self._calculate_likelihood_stage(sustainData, seq)
        return result

    def _optimise_parameters(self, sustainData, S_init, f_init, rng):
        # Optimise the parameters of the SuStaIn model

        M                                   = sustainData.getNumSamples()   #data_local.shape[0]
        N_S                                 = S_init.shape[0]
        N                                   = self.stage_zscore.shape[1]

        S_opt                               = S_init.copy()  # have to copy or changes will be passed to S_init
        f_opt                               = np.array(f_init).reshape(N_S, 1, 1)
        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))
        p_perm_k                            = np.zeros((M, N + 1, N_S))

        for s in range(N_S):
            p_perm_k[:, :, s]               = self._calculate_likelihood_stage(sustainData, S_opt[s])

        p_perm_k_weighted                   = p_perm_k * f_val_mat
        p_perm_k_norm                       = p_perm_k_weighted / np.sum(p_perm_k_weighted + 1e-250, axis=(1, 2), keepdims=True)
        f_opt                               = (np.squeeze(sum(sum(p_perm_k_norm))) / sum(sum(sum(p_perm_k_norm)))).reshape(N_S, 1, 1)
        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))
        order_seq                           = rng.permutation(N_S)  # this will produce different random numbers to Matlab

        for s in order_seq:
            other_prob_stage                = np.sum(p_perm_k * f_val_mat, 2) - p_perm_k[:, :, s] * f_val_mat[:, :, s]
            order_bio                       = rng.permutation(N)  # this will produce different random numbers to Matlab
            for i in order_bio:
                current_sequence            = S_opt[s]
                current_location            = np.array([0] * len(current_sequence))
                current_location[current_sequence.astype(int)] = np.arange(len(current_sequence))

                selected_event              = i

                move_event_from             = current_location[selected_event]

                this_stage_zscore           = self.stage_zscore[0, selected_event]
                selected_biomarker          = self.stage_biomarker_index[0, selected_event]
                possible_zscores_biomarker  = self.stage_zscore[self.stage_biomarker_index == selected_biomarker]

                # slightly different conditional check to matlab version to protect python from calling min,max on an empty array
                min_filter                  = possible_zscores_biomarker < this_stage_zscore
                max_filter                  = possible_zscores_biomarker > this_stage_zscore
                events                      = np.array(range(N))
                if np.any(min_filter):
                    min_zscore_bound        = max(possible_zscores_biomarker[min_filter])
                    min_zscore_bound_event  = events[((self.stage_zscore[0] == min_zscore_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                    move_event_to_lower_bound = current_location[min_zscore_bound_event] + 1
                else:
                    move_event_to_lower_bound = 0
                if np.any(max_filter):
                    max_zscore_bound        = min(possible_zscores_biomarker[max_filter])
                    max_zscore_bound_event  = events[((self.stage_zscore[0] == max_zscore_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                    move_event_to_upper_bound = current_location[max_zscore_bound_event]
                else:
                    move_event_to_upper_bound = N
                if move_event_to_lower_bound == move_event_to_upper_bound:
                    possible_positions      = np.array([move_event_to_lower_bound])
                else:
                    possible_positions      = np.arange(move_event_to_lower_bound, move_event_to_upper_bound)
                possible_sequences          = np.zeros((len(possible_positions), N))
                possible_likelihood         = np.zeros((len(possible_positions), 1))
                possible_p_perm_k           = np.zeros((M, N + 1, len(possible_positions)))
                for index in range(len(possible_positions)):
                    current_sequence        = S_opt[s]

                    #choose a position in the sequence to move an event to
                    move_event_to           = possible_positions[index]

                    # move this event in its new position
                    current_sequence        = np.delete(current_sequence, move_event_from, 0)  # this is different to the Matlab version, which call current_sequence(move_event_from) = []
                    new_sequence            = np.concatenate([current_sequence[np.arange(move_event_to)], [selected_event], current_sequence[np.arange(move_event_to, N - 1)]])
                    possible_sequences[index, :] = new_sequence

                    possible_p_perm_k[:, :, index] = self._calculate_likelihood_stage(sustainData, new_sequence)

                    p_perm_k[:, :, s]       = possible_p_perm_k[:, :, index]
                    total_prob_stage        = other_prob_stage + p_perm_k[:, :, s] * f_val_mat[:, :, s]
                    total_prob_subj         = np.sum(total_prob_stage, 1)
                    possible_likelihood[index] = np.sum(np.log(total_prob_subj + 1e-250))

                possible_likelihood         = possible_likelihood.reshape(possible_likelihood.shape[0])
                max_likelihood              = max(possible_likelihood)
                this_S                      = possible_sequences[possible_likelihood == max_likelihood, :]
                this_S                      = this_S[0, :]
                S_opt[s]                    = this_S
                this_p_perm_k               = possible_p_perm_k[:, :, possible_likelihood == max_likelihood]
                p_perm_k[:, :, s]           = this_p_perm_k[:, :, 0]

            S_opt[s]                        = this_S

        p_perm_k_weighted                   = p_perm_k * f_val_mat
        #adding 1e-250 fixes divide by zero problem that happens rarely
        #p_perm_k_norm                       = p_perm_k_weighted / np.tile(np.sum(np.sum(p_perm_k_weighted, 1), 1).reshape(M, 1, 1), (1, N + 1, N_S))  # the second summation axis is different to Matlab version
        p_perm_k_norm                       = p_perm_k_weighted / np.sum(p_perm_k_weighted + 1e-250, axis=(1, 2), keepdims=True)

        self._update_sigma(np.asarray(sustainData.data), S_opt, p_perm_k_norm)

        f_opt                               = (np.squeeze(sum(sum(p_perm_k_norm))) / sum(sum(sum(p_perm_k_norm)))).reshape(N_S, 1, 1)
        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))

        f_opt                               = f_opt.reshape(N_S)
        total_prob_stage                    = np.sum(p_perm_k * f_val_mat, 2)
        total_prob_subj                     = np.sum(total_prob_stage, 1)

        likelihood_opt                      = np.sum(np.log(total_prob_subj + 1e-250))

        return S_opt, f_opt, likelihood_opt

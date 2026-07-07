"""
Optimized wrappers around pySuStaIn OrdinalSustain.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from .abstract_override import AbstractSustain  # type: ignore
from pySuStaIn.OrdinalSustain import OrdinalSustain as _OrdinalSustain, OrdinalSustainData  # type: ignore
from .gpu_utils import get_torch_device


class OrdinalSustain(_OrdinalSustain):
    """Ordinal SuStaIn with reduced redundant recomputation and optional p_correct calibration."""

    def __init__(
        self,
        prob_nl,
        prob_score,
        score_vals,
        biomarker_labels,
        N_startpoints,
        N_S_max,
        N_iterations_MCMC,
        output_folder,
        dataset_name,
        use_parallel_startpoints,
        seed=None,
        *,
        p_correct: float = 0.9,
        p_correct_mode: str = "fixed",
        X_obs_raw: np.ndarray | None = None,
    ):
        self._mcmc_opt_iters = max(1, int(N_iterations_MCMC))

        N                               = prob_nl.shape[1]  # number of biomarkers
        assert (len(biomarker_labels) == N), "number of labels should match number of biomarkers"

        stage_score            = np.array([y for x in score_vals.T for y in x])
        stage_score            = stage_score.reshape(1, len(stage_score))
        IX_select              = stage_score > 0
        stage_score            = stage_score[IX_select]
        stage_score            = stage_score.reshape(1, len(stage_score))

        num_scores              = score_vals.shape[1]
        IX_vals                 = np.array([[x for x in range(N)]] * num_scores).T
        stage_biomarker_index   = np.array([y for x in IX_vals.T for y in x])
        stage_biomarker_index   = stage_biomarker_index.reshape(1, len(stage_biomarker_index))
        stage_biomarker_index   = stage_biomarker_index[IX_select]
        stage_biomarker_index   = stage_biomarker_index.reshape(1, len(stage_biomarker_index))

        prob_score = prob_score.transpose(0, 2, 1)
        prob_score = prob_score.reshape(prob_score.shape[0], prob_score.shape[1] * prob_score.shape[2])
        prob_score = prob_score[:, IX_select[0, :]]
        prob_score = prob_score.reshape(prob_nl.shape[0], stage_score.shape[1])

        self.IX_select                  = IX_select

        self.score_vals                 = score_vals
        self.stage_score                = stage_score
        self.stage_biomarker_index      = stage_biomarker_index

        self.biomarker_labels           = biomarker_labels

        numStages                       = stage_score.shape[1]
        self.__sustainData              = OrdinalSustainData(prob_nl, prob_score, numStages)

        AbstractSustain.__init__(
            self,
            self.__sustainData,
            N_startpoints,
            N_S_max,
            N_iterations_MCMC,
            output_folder,
            dataset_name,
            use_parallel_startpoints,
            seed,
        )

        # p_correct calibration (analogous to sigma_mode for z-score).
        p_correct_mode = str(p_correct_mode).lower().strip()
        if p_correct_mode not in ("fixed", "global", "per_biomarker"):
            raise ValueError("p_correct_mode must be one of: fixed, global, per_biomarker.")
        self.p_correct_mode = p_correct_mode
        self.p_correct = float(p_correct)

        # Per-biomarker p_correct array (always maintained; single shared value for fixed/global modes).
        B_orig = int(N)
        self._p_correct_arr = np.full(B_orig, float(p_correct))

        # Raw integer observations — required for EM calibration of p_correct.
        # Only stored when p_correct_mode != 'fixed'; None means calibration unavailable.
        if p_correct_mode != "fixed" and X_obs_raw is not None:
            self._X_obs_raw = np.asarray(X_obs_raw, dtype=float)
        else:
            self._X_obs_raw = None

    # ------------------------------------------------------------------
    # Expected-score computation
    # ------------------------------------------------------------------

    def _compute_expected_scores(self, S: np.ndarray) -> np.ndarray:
        """Return (B, N+1) array of expected integer ordinal scores for each biomarker at each stage.

        At stage 0 all biomarkers are at score 0 (normal/not-yet-reached).
        Each subsequent stage advances the sequence by one event.
        """
        N = int(self.stage_score.shape[1])
        B = int(np.max(self.stage_biomarker_index) + 1)
        expected = np.zeros((B, N + 1), dtype=float)
        for k in range(N):
            expected[:, k + 1] = expected[:, k]
            event_idx = int(S[k])
            b = int(self.stage_biomarker_index[0, event_idx])
            expected[b, k + 1] = float(self.stage_score[0, event_idx])
        return expected

    # ------------------------------------------------------------------
    # Likelihood stage — override to use calibrated p_correct on raw scores
    # ------------------------------------------------------------------

    def _calculate_likelihood_stage(self, sustainData, S):
        """Compute stage likelihoods.

        When p_correct_mode is 'fixed' (or raw observations are unavailable) this
        delegates to the pySuStaIn base, which uses the pre-built prob_nl/prob_score
        matrices.  For 'global' and 'per_biomarker' modes the likelihood is computed
        directly from self._X_obs_raw using the current self._p_correct_arr so that
        EM-calibrated values take effect on the next iteration.
        """
        if self.p_correct_mode == "fixed" or self._X_obs_raw is None:
            return super()._calculate_likelihood_stage(sustainData, S)

        M_data = self._X_obs_raw.shape[0]
        if sustainData.getNumSamples() != M_data:
            # Dimension mismatch — fall back for CV folds / split sub-datasets.
            return super()._calculate_likelihood_stage(sustainData, S)

        N = int(self.stage_score.shape[1])
        N_scores = int(self.score_vals.shape[1])
        X = self._X_obs_raw  # (M, B)
        expected = self._compute_expected_scores(S)  # (B, N+1)
        log_prior = np.log(1.0 / (N + 1))

        p_perm_k = np.zeros((M_data, N + 1))

        if self.p_correct_mode == "global":
            p_c = float(self.p_correct)
            lp_correct = np.log(max(p_c, 1e-300))
            lp_wrong = np.log(max((1.0 - p_c) / N_scores, 1e-300))
            for k in range(N + 1):
                match = (X == expected[:, k][None, :])     # (M, B) bool
                nan_mask = np.isnan(X)
                log_p = np.where(nan_mask, 0.0, np.where(match, lp_correct, lp_wrong))
                p_perm_k[:, k] = np.exp(log_prior + np.sum(log_p, axis=1))

        else:  # per_biomarker
            p_c_arr = self._p_correct_arr                  # (B,)
            lp_correct = np.log(np.maximum(p_c_arr, 1e-300))      # (B,)
            lp_wrong = np.log(np.maximum((1.0 - p_c_arr) / N_scores, 1e-300))  # (B,)
            for k in range(N + 1):
                match = (X == expected[:, k][None, :])     # (M, B)
                nan_mask = np.isnan(X)
                log_p = np.where(nan_mask, 0.0, np.where(match, lp_correct[None, :], lp_wrong[None, :]))
                p_perm_k[:, k] = np.exp(log_prior + np.sum(log_p, axis=1))

        return p_perm_k

    def _calculate_likelihood_stage_batch(self, sustainData, sequences):
        """Compute _calculate_likelihood_stage for multiple sequences simultaneously.

        sequences: (n_pos, N) array-like; each row is a full event ordering.
        Returns: (M, N+1, n_pos) array.

        Replaces the per-position serial loop from the original pySuStaIn
        _optimise_parameters. Instead of calling _calculate_likelihood_stage
        once per candidate position (Python loop, ~2-3 ms each), all positions
        are batched into a single (M, n_pos, B, N+1) broadcast — see
        docs/em_position_batch_optimization.md for rationale and benchmarks.

        Fast path covers global and per_biomarker p_correct modes.
        Falls back to serial for fixed mode (delegates to read-only vendor code).
        """
        sequences = np.asarray(sequences, dtype=np.int32)
        n_pos = len(sequences)
        if n_pos == 1:
            return self._calculate_likelihood_stage(sustainData, sequences[0])[:, :, np.newaxis]

        N = int(self.stage_score.shape[1])
        log_prior = np.log(1.0 / (N + 1))

        # ------------------------------------------------------------------
        # Fast path A: fixed mode — replicates vendor _calculate_likelihood_stage
        # exactly, using the pre-built prob_nl / prob_score tables.
        #
        # For each sequence and stage k, the vendor tracks last_event_for_b:
        # the highest-threshold event already reached for each biomarker b.
        # Likelihood = prod_{b normal} prob_nl[m,b]
        #              × prod_{b abnormal} prob_score[m, last_event_for_b]
        #
        # We build last_event[pos, k, b] with an N-iteration loop (cheap),
        # then gather from the log-probability tables in one batched op.
        # ------------------------------------------------------------------
        if self.p_correct_mode == "fixed":
            B = sustainData.prob_nl.shape[1]
            bio_idx = self.stage_biomarker_index[0]  # (N,) biomarker for each event

            # last_event[pos, k, b]: event index of last-reached event for
            # biomarker b at stage k in sequence pos. -1 means still normal.
            last_event = np.full((n_pos, N + 1, B), -1, dtype=np.int32)
            for k in range(1, N + 1):
                last_event[:, k, :] = last_event[:, k - 1, :]
                just_reached = sequences[:, k - 1]           # (n_pos,)
                just_bio = bio_idx[just_reached]             # (n_pos,)
                last_event[np.arange(n_pos), k, just_bio] = just_reached

            is_normal = last_event == -1                     # (n_pos, N+1, B)
            safe_event = np.where(is_normal, 0, last_event)  # replace -1 → 0

            log_pnl = np.log(np.maximum(sustainData.prob_nl,  1e-300))  # (M, B)
            log_ps  = np.log(np.maximum(sustainData.prob_score, 1e-300))  # (M, N)

            _torch, _device = get_torch_device()
            if _torch is not None:
                _n_samples = sustainData.prob_nl.shape[0]
                if not hasattr(self, '_gpu_cache') or self._gpu_cache.get('device') != _device or self._gpu_cache.get('n_samples') != _n_samples:
                    self._gpu_cache = {
                        'device':    _device,
                        'n_samples': _n_samples,
                        't_log_pnl': _torch.as_tensor(log_pnl, dtype=_torch.float64, device=_device),
                        't_log_ps':  _torch.as_tensor(log_ps,  dtype=_torch.float64, device=_device),
                    }
                t_log_pnl = self._gpu_cache['t_log_pnl']
                t_log_ps  = self._gpu_cache['t_log_ps']
                t_safe    = _torch.as_tensor(safe_event, dtype=_torch.long, device=_device)
                t_is_norm = _torch.as_tensor(is_normal,                     device=_device)
                # gather: (M, n_pos, N+1, B)
                t_gathered = t_log_ps[:, t_safe]
                t_log_prob = _torch.where(t_is_norm[None], t_log_pnl[:, None, None, :], t_gathered)
                p = _torch.exp(log_prior + t_log_prob.sum(dim=3))  # (M, n_pos, N+1)
                return p.permute(0, 2, 1).cpu().numpy()

            # CPU numpy path (same math, no GPU)
            log_ps_gathered = log_ps[:, safe_event]          # (M, n_pos, N+1, B)
            log_prob = np.where(
                is_normal[np.newaxis],
                log_pnl[:, np.newaxis, np.newaxis, :],
                log_ps_gathered,
            )
            p = np.exp(log_prior + np.sum(log_prob, axis=3))  # (M, n_pos, N+1)
            return p.transpose(0, 2, 1)

        # ------------------------------------------------------------------
        # Fast path B: global / per_biomarker — use raw observed scores and
        # p_correct directly (no prob tables needed).
        # ------------------------------------------------------------------
        if self._X_obs_raw is None or sustainData.getNumSamples() != self._X_obs_raw.shape[0]:
            M = sustainData.getNumSamples()
            result = np.zeros((M, N + 1, n_pos))
            for i, seq in enumerate(sequences):
                result[:, :, i] = self._calculate_likelihood_stage(sustainData, seq)
            return result

        N_scores = int(self.score_vals.shape[1])
        X = self._X_obs_raw  # (M, B)
        nan_mask = np.isnan(X)  # (M, B)

        # expected_all: (n_pos, B, N+1)
        expected_all = np.stack([self._compute_expected_scores(seq) for seq in sequences], axis=0)

        _torch, _device = get_torch_device()
        if _torch is not None:
            t_X = _torch.as_tensor(X, dtype=_torch.float64, device=_device)
            t_expected = _torch.as_tensor(expected_all, dtype=_torch.float64, device=_device)
            t_nan = _torch.as_tensor(nan_mask, device=_device)
            match = (t_X[:, None, :, None] == t_expected[None])  # (M, n_pos, B, N+1)
            zeros = _torch.zeros(1, dtype=_torch.float64, device=_device)
            if self.p_correct_mode == "global":
                p_c = float(self.p_correct)
                lp_c = float(np.log(max(p_c, 1e-300)))
                lp_w = float(np.log(max((1.0 - p_c) / N_scores, 1e-300)))
                log_p = _torch.where(
                    t_nan[:, None, :, None], zeros,
                    _torch.where(match,
                        _torch.tensor(lp_c, dtype=_torch.float64, device=_device),
                        _torch.tensor(lp_w, dtype=_torch.float64, device=_device)),
                )
            else:  # per_biomarker
                t_lp_c = _torch.as_tensor(np.log(np.maximum(self._p_correct_arr, 1e-300)), dtype=_torch.float64, device=_device)
                t_lp_w = _torch.as_tensor(np.log(np.maximum((1.0 - self._p_correct_arr) / N_scores, 1e-300)), dtype=_torch.float64, device=_device)
                log_p = _torch.where(
                    t_nan[:, None, :, None], zeros,
                    _torch.where(match, t_lp_c[None, None, :, None], t_lp_w[None, None, :, None]),
                )
            p = _torch.exp(log_prior + _torch.sum(log_p, dim=2))  # (M, n_pos, N+1)
            return p.permute(0, 2, 1).cpu().numpy()

        # match: (M, n_pos, B, N+1)
        match = (X[:, None, :, None] == expected_all[None, :, :, :])

        if self.p_correct_mode == "global":
            p_c = float(self.p_correct)
            lp_correct = np.log(max(p_c, 1e-300))
            lp_wrong = np.log(max((1.0 - p_c) / N_scores, 1e-300))
            log_p = np.where(nan_mask[:, None, :, None], 0.0, np.where(match, lp_correct, lp_wrong))
        else:  # per_biomarker
            p_c_arr = self._p_correct_arr  # (B,)
            lp_correct = np.log(np.maximum(p_c_arr, 1e-300))  # (B,)
            lp_wrong = np.log(np.maximum((1.0 - p_c_arr) / N_scores, 1e-300))  # (B,)
            log_p = np.where(
                nan_mask[:, None, :, None], 0.0,
                np.where(match, lp_correct[None, None, :, None], lp_wrong[None, None, :, None])
            )

        # sum over B (axis=2): (M, n_pos, N+1) → transpose to (M, N+1, n_pos)
        p = np.exp(log_prior + np.sum(log_p, axis=2))
        return p.transpose(0, 2, 1)

    # ------------------------------------------------------------------
    # M-step: update p_correct from EM responsibilities
    # ------------------------------------------------------------------

    def _update_p_correct(
        self,
        X_obs_raw: np.ndarray | None,
        S_opt: np.ndarray,
        p_perm_k_norm: np.ndarray,
    ) -> None:
        """M-step update of p_correct (or per-biomarker p_correct_arr) from responsibility-weighted match counts.

        p_perm_k_norm[i, k, s] = p(stage=k, subtype=s | data_i), shape (M, N+1, N_S).
        NaN positions in X_obs_raw contribute 0 to both numerator and denominator.

        Updates self.p_correct and self._p_correct_arr in place.
        Silently returns when calibration is unavailable (fixed mode, no raw data, or M mismatch).
        """
        if self.p_correct_mode == "fixed":
            return
        if X_obs_raw is None:
            return

        X = np.asarray(X_obs_raw, dtype=float)
        M_obs = int(X.shape[0])
        M_resp = int(p_perm_k_norm.shape[0])
        if M_obs != M_resp:
            return  # dimension mismatch (e.g., CV fold subset)

        N_S = int(p_perm_k_norm.shape[2])
        N_scores = int(self.score_vals.shape[1])
        floor_val = 1.0 / N_scores + 1e-4
        B = int(X.shape[1])
        valid = (~np.isnan(X)).astype(float)   # (M, B)

        if self.p_correct_mode == "global":
            num = 0.0
            den = 0.0
            for s in range(N_S):
                expected = self._compute_expected_scores(S_opt[s])    # (B, N+1)
                for k in range(expected.shape[1]):
                    r_ik = p_perm_k_norm[:, k, s]                     # (M,)
                    match = (X == expected[:, k][None, :]).astype(float) * valid  # (M, B)
                    num += float(np.sum(r_ik * np.sum(match, axis=1)))
                    den += float(np.sum(r_ik * np.sum(valid, axis=1)))
            if den < 1e-12:
                return
            p_new = float(np.clip(num / den, floor_val, 1.0 - 1e-6))
            self.p_correct = p_new
            self._p_correct_arr[:] = p_new

        elif self.p_correct_mode == "per_biomarker":
            num_b = np.zeros(B, dtype=float)
            den_b = np.zeros(B, dtype=float)
            for s in range(N_S):
                expected = self._compute_expected_scores(S_opt[s])    # (B, N+1)
                for k in range(expected.shape[1]):
                    r_ik = p_perm_k_norm[:, k, s]                     # (M,)
                    match = (X == expected[:, k][None, :]).astype(float) * valid  # (M, B)
                    num_b += np.sum(r_ik[:, None] * match, axis=0)
                    den_b += np.sum(r_ik[:, None] * valid, axis=0)
            den_b = np.maximum(den_b, 1e-12)
            p_b = np.clip(num_b / den_b, floor_val, 1.0 - 1e-6)
            self._p_correct_arr = p_b
            self.p_correct = float(np.mean(p_b))

    # ------------------------------------------------------------------
    # _find_ml overrides — propagate calibrated p_correct from worker to main
    # ------------------------------------------------------------------

    def _find_ml_iteration(self, sustainData, seed_seq):
        """Override to return p_correct_arr as 4th element for propagation back to main process."""
        rng = np.random.default_rng(seed_seq)
        seq_init = self._initialise_sequence(sustainData, rng)
        f_init = [1]
        this_ml_sequence, this_ml_f, this_ml_likelihood, _, _, _ = self._perform_em(
            sustainData, seq_init, f_init, rng
        )
        return this_ml_sequence, this_ml_f, this_ml_likelihood, list(self._p_correct_arr)

    def _find_ml(self, sustainData):
        """Override to propagate p_correct from the best EM startpoint back to the main model.

        The parent _find_ml runs startpoints via pool.map() (separate worker processes).
        Each worker calls _optimise_parameters → _update_p_correct, updating
        self._p_correct_arr in the worker's copy only.  That update is lost when the
        worker exits.  This override captures the array returned by _find_ml_iteration
        and applies the winning startpoint's values to the main process model before
        MCMC begins.
        """
        if self.p_correct_mode == "fixed":
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

        # Apply p_correct from the winning startpoint to the main process model.
        winning_arr = np.asarray(pool_output_list[ix][3], dtype=float)
        self._p_correct_arr = winning_arr
        self.p_correct = float(np.mean(winning_arr))

        return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat

    def _find_ml_split_iteration(self, sustainData, seed_seq):
        """Override to return p_correct_arr as 4th element so _find_ml_split can propagate it."""
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
        return this_ml_sequence, this_ml_f, this_ml_likelihood, list(self._p_correct_arr)

    def _find_ml_split(self, sustainData):
        """Override to propagate p_correct from the best split startpoint back to the main model."""
        if self.p_correct_mode == "fixed":
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

        # Apply p_correct from the winning startpoint to the main process model.
        winning_arr = np.asarray(pool_output_list[ix[0]][3], dtype=float)
        self._p_correct_arr = winning_arr
        self.p_correct = float(np.mean(winning_arr))

        return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat

    # ------------------------------------------------------------------
    # MCMC optimisation settings
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Inner EM optimisation (base class — stacked/longitudinal subclasses
    # override and add the _update_p_correct call at the end)
    # ------------------------------------------------------------------

    def _optimise_parameters(self, sustainData, S_init, f_init, rng):
        # Optimise the parameters of the SuStaIn model

        M                                   = sustainData.getNumSamples()
        N_S                                 = S_init.shape[0]
        N                                   = self.stage_score.shape[1]

        S_opt                               = S_init.copy()
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
        order_seq                           = rng.permutation(N_S)

        for s in order_seq:
            other_prob_stage            = np.sum(p_perm_k * f_val_mat, 2) - p_perm_k[:, :, s] * f_val_mat[:, :, s]
            order_bio                       = rng.permutation(N)
            for i in order_bio:
                current_sequence            = S_opt[s]
                current_location            = np.array([0] * len(current_sequence))
                current_location[current_sequence.astype(int)] = np.arange(len(current_sequence))

                selected_event              = i
                move_event_from             = current_location[selected_event]

                this_stage_score           = self.stage_score[0, selected_event]
                selected_biomarker          = self.stage_biomarker_index[0, selected_event]
                possible_scores_biomarker  = self.stage_score[self.stage_biomarker_index == selected_biomarker]

                min_filter                  = possible_scores_biomarker < this_stage_score
                max_filter                  = possible_scores_biomarker > this_stage_score
                events                      = np.array(range(N))
                if np.any(min_filter):
                    min_score_bound        = max(possible_scores_biomarker[min_filter])
                    min_score_bound_event  = events[((self.stage_score[0] == min_score_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                    move_event_to_lower_bound = int(np.asarray(current_location[min_score_bound_event]).item()) + 1
                else:
                    move_event_to_lower_bound = 0
                if np.any(max_filter):
                    max_score_bound        = min(possible_scores_biomarker[max_filter])
                    max_score_bound_event  = events[((self.stage_score[0] == max_score_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                    move_event_to_upper_bound = int(np.asarray(current_location[max_score_bound_event]).item())
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
                    move_event_to           = possible_positions[index]
                    current_sequence        = np.delete(current_sequence, move_event_from, 0)
                    new_sequence            = np.concatenate([current_sequence[np.arange(move_event_to)], [selected_event], current_sequence[np.arange(move_event_to, N - 1)]])
                    possible_sequences[index, :] = new_sequence

                    possible_p_perm_k[:, :, index] = self._calculate_likelihood_stage(sustainData, new_sequence)

                    p_perm_k[:, :, s]       = possible_p_perm_k[:, :, index]
                    total_prob_stage        = other_prob_stage + p_perm_k[:, :, s] * f_val_mat[:, :, s]
                    total_prob_subj         = np.sum(total_prob_stage, 1)
                    possible_likelihood[index] = sum(np.log(total_prob_subj + 1e-250))

                possible_likelihood         = possible_likelihood.reshape(possible_likelihood.shape[0])
                max_likelihood              = max(possible_likelihood)
                this_S                      = possible_sequences[possible_likelihood == max_likelihood, :]
                this_S                      = this_S[0, :]
                S_opt[s]                    = this_S
                this_p_perm_k               = possible_p_perm_k[:, :, possible_likelihood == max_likelihood]
                p_perm_k[:, :, s]           = this_p_perm_k[:, :, 0]

            S_opt[s]                        = this_S

        p_perm_k_weighted                   = p_perm_k * f_val_mat
        p_perm_k_norm                       = p_perm_k_weighted / np.tile(np.sum(np.sum(p_perm_k_weighted, 1), 1).reshape(M, 1, 1), (1, N + 1, N_S))
        f_opt                               = (np.squeeze(sum(sum(p_perm_k_norm))) / sum(sum(sum(p_perm_k_norm)))).reshape(N_S, 1, 1)

        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))

        f_opt                               = f_opt.reshape(N_S)
        total_prob_stage                    = np.sum(p_perm_k * f_val_mat, 2)
        total_prob_subj                     = np.sum(total_prob_stage, 1)

        likelihood_opt                      = sum(np.log(total_prob_subj + 1e-250))

        return S_opt, f_opt, likelihood_opt

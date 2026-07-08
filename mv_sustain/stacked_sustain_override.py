"""
Stacked SuStaIn variants for the independent-visit training path.

In stacked (classic) SuStaIn, each visit is treated as independent during training — there is no joint patient-level pooling. At inference, the post-hoc longitudinal constraint (product of per-visit posteriors) can be applied to combine visits for a single patient, but this is separate from training.

These classes:
  - Inherit the corrected likelihood stages from the local zscore/ordinal overrides (sigma_noise wiring, normalization constant fix, missing-data policy).
  - Use the cache-recomputation optimisation from _LongitudinalLikelihoodMixin by treating each visit as its own trivial single-visit patient. With a trivial patient_index = [0, 1, ..., M-1], the longitudinal stats reduce to the classic per-visit EM — same mathematics, faster computation.
  - Fix the possible_positions edge case (np.array([move_event_to_lower_bound]) instead of the pySuStaIn bug np.array([0])) for all code paths.
  - Have no MV-specific branching: they are unconditionally stacked.

Class hierarchy:

    zscore_override.ZscoreSustain        (sigma_noise, cov, _calculate_likelihood_stage)
      └── StackedZscoreSustain           (trivial patient_index, cache opt, pos fix)

    ordinal_override.OrdinalSustain      (_calculate_likelihood_stage via pySuStaIn)
      └── StackedOrdinalSustain          (trivial patient_index, cache opt, pos fix)

Usage in sustain_utils.SustainRunner.initialize_model():
    use_longitudinal_likelihood=False → StackedZscoreSustain / StackedOrdinalSustain
    use_longitudinal_likelihood=True  → LongitudinalZscoreSustain / LongitudinalOrdinalSustain
"""

from __future__ import annotations

import numpy as np

from .zscore_override import ZscoreSustain
from .ordinal_override import OrdinalSustain
from .longitudinal_override import _LongitudinalLikelihoodMixin  # type: ignore


class StackedZscoreSustain(_LongitudinalLikelihoodMixin, ZscoreSustain):
    """
    Z-score SuStaIn for independent-visit (stacked) training.

    Every visit is treated as its own single-visit patient (trivial patient_index).
    This reduces the longitudinal joint likelihood to the product of independent
    visit likelihoods — mathematically identical to classic stacked SuStaIn EM,
    with the cache-recomputation optimisation applied.

    _calculate_likelihood is inherited from ZscoreSustain → pySuStaIn AbstractSustain,
    which computes the correct prior-weighted stacked marginal required by CVIC.
    """

    def __init__(self, *args, use_cache_recomputation: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache_recomputation = bool(use_cache_recomputation)

    def _optimise_parameters(self, sustainData, S_init, f_init, rng):
        M = sustainData.getNumSamples()
        N_S = S_init.shape[0]
        N = self.stage_zscore.shape[1]

        # Each visit is its own patient: reduces to classic stacked EM.
        patient_index = np.arange(M, dtype=np.int32)
        n_patients = M

        S_opt = S_init.copy()
        f_opt = np.asarray(f_init, dtype=float).reshape(N_S)
        p_perm_k = np.zeros((M, N + 1, N_S))

        for s in range(N_S):
            p_perm_k[:, :, s] = self._calculate_likelihood_stage(sustainData, S_opt[s])

        f_opt = self._longitudinal_update_f(p_perm_k, f_opt, patient_index, n_patients)
        order_seq = rng.permutation(N_S)
        use_cache = self.use_cache_recomputation

        for s in order_seq:
            if use_cache:
                logsumexp_other, log_f_s = self._prepare_longitudinal_loglike_cache(
                    p_perm_k, f_opt, patient_index, n_patients, s
                )
            order_bio = rng.permutation(N)
            current_sequence = S_opt[s]
            current_location = np.zeros(N, dtype=np.int64)
            current_location[current_sequence.astype(int)] = np.arange(N)
            for i in order_bio:
                selected_event = i
                move_event_from = current_location[selected_event]

                this_stage_zscore = self.stage_zscore[0, selected_event]
                selected_biomarker = self.stage_biomarker_index[0, selected_event]
                possible_zscores_biomarker = self.stage_zscore[self.stage_biomarker_index == selected_biomarker]

                min_filter = possible_zscores_biomarker < this_stage_zscore
                max_filter = possible_zscores_biomarker > this_stage_zscore
                events = np.array(range(N))
                if np.any(min_filter):
                    min_zscore_bound = max(possible_zscores_biomarker[min_filter])
                    min_zscore_bound_event = events[
                        ((self.stage_zscore[0] == min_zscore_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2
                    ]
                    move_event_to_lower_bound = current_location[min_zscore_bound_event] + 1
                else:
                    move_event_to_lower_bound = 0
                if np.any(max_filter):
                    max_zscore_bound = min(possible_zscores_biomarker[max_filter])
                    max_zscore_bound_event = events[
                        ((self.stage_zscore[0] == max_zscore_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2
                    ]
                    move_event_to_upper_bound = current_location[max_zscore_bound_event]
                else:
                    move_event_to_upper_bound = N

                if move_event_to_lower_bound == move_event_to_upper_bound:
                    possible_positions = np.array([move_event_to_lower_bound])
                else:
                    possible_positions = np.arange(move_event_to_lower_bound, move_event_to_upper_bound)

                possible_likelihood = np.zeros((len(possible_positions), 1))

                # Build all candidate sequences (base_seq is constant across positions)
                base_seq = np.delete(S_opt[s], move_event_from, 0)
                n_pos = len(possible_positions)
                possible_sequences = np.empty((n_pos, N), dtype=np.int32)
                for index, move_event_to in enumerate(possible_positions):
                    possible_sequences[index, :move_event_to] = base_seq[:move_event_to]
                    possible_sequences[index, move_event_to]  = selected_event
                    possible_sequences[index, move_event_to+1:] = base_seq[move_event_to:N-1]

                # Compute all position likelihoods in one batched call
                possible_p_perm_k = self._calculate_likelihood_stage_batch(sustainData, possible_sequences)

                # Evaluate loglike for each candidate (no likelihood recomputation)
                if use_cache:
                    possible_likelihood = self._longitudinal_loglike_cached_batch(
                        possible_p_perm_k, patient_index, n_patients, s, logsumexp_other, log_f_s
                    )
                else:
                    for index in range(len(possible_positions)):
                        p_perm_k[:, :, s] = possible_p_perm_k[:, :, index]
                        possible_likelihood[index] = self._longitudinal_loglike(
                            p_perm_k, f_opt, patient_index, n_patients
                        )

                possible_likelihood = possible_likelihood.reshape(possible_likelihood.shape[0])
                max_likelihood = max(possible_likelihood)
                this_S = possible_sequences[possible_likelihood == max_likelihood, :]
                this_S = this_S[0, :]
                S_opt[s] = this_S
                this_p_perm_k = possible_p_perm_k[:, :, possible_likelihood == max_likelihood]
                p_perm_k[:, :, s] = this_p_perm_k[:, :, 0]
                current_location = np.zeros(N, dtype=np.int64)
                current_location[this_S.astype(int)] = np.arange(N)

            S_opt[s] = this_S

        f_opt = self._longitudinal_update_f(p_perm_k, f_opt, patient_index, n_patients)

        if getattr(self, 'sigma_mode', 'fixed') != 'fixed':
            _, _, resp_visit = self._longitudinal_stats(p_perm_k, f_opt, patient_index, n_patients)
            stage_sum = np.sum(p_perm_k, axis=1, keepdims=True)
            p_stage_cond = p_perm_k / (stage_sum + 1e-250)
            p_norm = p_stage_cond * resp_visit[:, None, :]
            self._update_sigma(np.asarray(sustainData.data), S_opt, p_norm)

        likelihood_opt = self._longitudinal_loglike(p_perm_k, f_opt, patient_index, n_patients)
        return S_opt, f_opt, likelihood_opt


class StackedOrdinalSustain(_LongitudinalLikelihoodMixin, OrdinalSustain):
    """
    Ordinal SuStaIn for independent-visit (stacked) training.

    Every visit is treated as its own single-visit patient (trivial patient_index).
    This reduces the longitudinal joint likelihood to the product of independent
    visit likelihoods — mathematically identical to classic stacked SuStaIn EM,
    with the cache-recomputation optimisation applied.

    _calculate_likelihood is inherited from OrdinalSustain → pySuStaIn AbstractSustain,
    which computes the correct prior-weighted stacked marginal required by CVIC.
    """

    def __init__(self, *args, use_cache_recomputation: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache_recomputation = bool(use_cache_recomputation)

    def _optimise_parameters(self, sustainData, S_init, f_init, rng):
        M = sustainData.getNumSamples()
        N_S = S_init.shape[0]
        N = self.stage_score.shape[1]

        # Each visit is its own patient: reduces to classic stacked EM.
        patient_index = np.arange(M, dtype=np.int32)
        n_patients = M

        S_opt = S_init.copy()
        f_opt = np.asarray(f_init, dtype=float).reshape(N_S)
        p_perm_k = np.zeros((M, N + 1, N_S))

        for s in range(N_S):
            p_perm_k[:, :, s] = self._calculate_likelihood_stage(sustainData, S_opt[s])

        f_opt = self._longitudinal_update_f(p_perm_k, f_opt, patient_index, n_patients)
        order_seq = rng.permutation(N_S)
        use_cache = self.use_cache_recomputation

        for s in order_seq:
            if use_cache:
                logsumexp_other, log_f_s = self._prepare_longitudinal_loglike_cache(
                    p_perm_k, f_opt, patient_index, n_patients, s
                )
            order_bio = rng.permutation(N)
            current_sequence = S_opt[s]
            current_location = np.zeros(N, dtype=np.int64)
            current_location[current_sequence.astype(int)] = np.arange(N)
            for i in order_bio:
                selected_event = i
                move_event_from = current_location[selected_event]

                this_stage_score = self.stage_score[0, selected_event]
                selected_biomarker = self.stage_biomarker_index[0, selected_event]
                possible_scores_biomarker = self.stage_score[self.stage_biomarker_index == selected_biomarker]

                min_filter = possible_scores_biomarker < this_stage_score
                max_filter = possible_scores_biomarker > this_stage_score
                events = np.array(range(N))
                if np.any(min_filter):
                    min_score_bound = max(possible_scores_biomarker[min_filter])
                    min_score_bound_event = events[
                        ((self.stage_score[0] == min_score_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2
                    ]
                    move_event_to_lower_bound = int(np.asarray(current_location[min_score_bound_event]).item()) + 1
                else:
                    move_event_to_lower_bound = 0
                if np.any(max_filter):
                    max_score_bound = min(possible_scores_biomarker[max_filter])
                    max_score_bound_event = events[
                        ((self.stage_score[0] == max_score_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2
                    ]
                    move_event_to_upper_bound = int(np.asarray(current_location[max_score_bound_event]).item())
                else:
                    move_event_to_upper_bound = N

                if move_event_to_lower_bound == move_event_to_upper_bound:
                    possible_positions = np.array([move_event_to_lower_bound])
                else:
                    possible_positions = np.arange(move_event_to_lower_bound, move_event_to_upper_bound)

                possible_likelihood = np.zeros((len(possible_positions), 1))

                # Build all candidate sequences (base_seq is constant across positions)
                base_seq = np.delete(S_opt[s], move_event_from, 0)
                n_pos = len(possible_positions)
                possible_sequences = np.empty((n_pos, N), dtype=np.int32)
                for index, move_event_to in enumerate(possible_positions):
                    possible_sequences[index, :move_event_to] = base_seq[:move_event_to]
                    possible_sequences[index, move_event_to]  = selected_event
                    possible_sequences[index, move_event_to+1:] = base_seq[move_event_to:N-1]

                # Compute all position likelihoods in one batched call
                possible_p_perm_k = self._calculate_likelihood_stage_batch(sustainData, possible_sequences)

                # Evaluate loglike for each candidate (no likelihood recomputation)
                if use_cache:
                    possible_likelihood = self._longitudinal_loglike_cached_batch(
                        possible_p_perm_k, patient_index, n_patients, s, logsumexp_other, log_f_s
                    )
                else:
                    for index in range(len(possible_positions)):
                        p_perm_k[:, :, s] = possible_p_perm_k[:, :, index]
                        possible_likelihood[index] = self._longitudinal_loglike(
                            p_perm_k, f_opt, patient_index, n_patients
                        )

                possible_likelihood = possible_likelihood.reshape(possible_likelihood.shape[0])
                max_likelihood = max(possible_likelihood)
                this_S = possible_sequences[possible_likelihood == max_likelihood, :]
                this_S = this_S[0, :]
                S_opt[s] = this_S
                this_p_perm_k = possible_p_perm_k[:, :, possible_likelihood == max_likelihood]
                p_perm_k[:, :, s] = this_p_perm_k[:, :, 0]
                current_location = np.zeros(N, dtype=np.int64)
                current_location[this_S.astype(int)] = np.arange(N)

            S_opt[s] = this_S

        f_opt = self._longitudinal_update_f(p_perm_k, f_opt, patient_index, n_patients)

        if getattr(self, 'p_correct_mode', 'fixed') != 'fixed':
            _, _, resp_visit = self._longitudinal_stats(p_perm_k, f_opt, patient_index, n_patients)
            stage_sum = np.sum(p_perm_k, axis=1, keepdims=True)
            p_stage_cond = p_perm_k / (stage_sum + 1e-250)
            p_norm = p_stage_cond * resp_visit[:, None, :]
            self._update_p_correct(self._X_obs_raw, S_opt, p_norm)

        likelihood_opt = self._longitudinal_loglike(p_perm_k, f_opt, patient_index, n_patients)
        return S_opt, f_opt, likelihood_opt

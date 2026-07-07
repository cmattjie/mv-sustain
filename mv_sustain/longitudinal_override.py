"""
Longitudinal SuStaIn variants with patient-level subtype likelihoods.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from tqdm import tqdm

from .zscore_override import ZscoreSustain  # type: ignore
from .ordinal_override import OrdinalSustain  # type: ignore
from .mixture_override import MixtureSustain  # type: ignore
from pySuStaIn.ZscoreSustain import ZScoreSustainData  # type: ignore
from pySuStaIn.OrdinalSustain import OrdinalSustainData  # type: ignore
from pySuStaIn.MixtureSustain import MixtureSustainData  # type: ignore


class LongitudinalZScoreSustainData(ZScoreSustainData):
    """Z-score sustain data with patient identifiers for longitudinal grouping."""

    def __init__(self, data: np.ndarray, numStages: int, patient_ids: Sequence[object]):
        super().__init__(data, numStages)
        self.patient_ids = np.asarray(patient_ids)
        if self.patient_ids.shape[0] != self.data.shape[0]:
            raise ValueError("patient_ids length must match number of samples.")

    def reindex(self, index):
        return LongitudinalZScoreSustainData(self.data[index,], self.getNumStages(), self.patient_ids[index])


class LongitudinalOrdinalSustainData(OrdinalSustainData):
    """Ordinal sustain data with patient identifiers for longitudinal grouping."""

    def __init__(self, prob_nl: np.ndarray, prob_score: np.ndarray, numStages: int, patient_ids: Sequence[object]):
        super().__init__(prob_nl, prob_score, numStages)
        self.patient_ids = np.asarray(patient_ids)
        if self.patient_ids.shape[0] != self.prob_nl.shape[0]:
            raise ValueError("patient_ids length must match number of samples.")

    def reindex(self, index):
        return LongitudinalOrdinalSustainData(
            self.prob_nl[index,],
            self.prob_score[index,],
            self.getNumStages(),
            self.patient_ids[index],
        )


class LongitudinalMixtureSustainData(MixtureSustainData):
    """Mixture sustain data with patient identifiers for longitudinal grouping."""

    def __init__(self, L_yes: np.ndarray, L_no: np.ndarray, numStages: int, patient_ids: Sequence[object]):
        super().__init__(L_yes, L_no, numStages)
        self.patient_ids = np.asarray(patient_ids)
        if self.patient_ids.shape[0] != self.L_yes.shape[0]:
            raise ValueError("patient_ids length must match number of samples.")

    def reindex(self, index):
        return LongitudinalMixtureSustainData(
            self.L_yes[index,],
            self.L_no[index,],
            self.getNumStages(),
            self.patient_ids[index],
        )


class _LongitudinalLikelihoodMixin:
    """Shared longitudinal likelihood helpers for SuStaIn variants."""

    def _get_longitudinal_index(self, sustainData: object) -> tuple[np.ndarray | None, int | None]:
        patient_ids = getattr(sustainData, "patient_ids", None)
        if patient_ids is None:
            return None, None
        unique_ids, inverse = np.unique(np.asarray(patient_ids), return_inverse=True)
        return inverse, int(unique_ids.size)

    def _longitudinal_stats(
        self,
        p_perm_k: np.ndarray,
        f: np.ndarray,
        patient_index: np.ndarray,
        n_patients: int,
        *,
        eps: float = 1e-250,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        # p_perm_k: (n_visits, n_stages, n_subtypes)
        p_visit_subtype = np.sum(p_perm_k, axis=1)
        log_p_visit_subtype = np.log(p_visit_subtype + eps)

        log_p_patient_subtype = np.zeros((n_patients, p_visit_subtype.shape[1]), dtype=float)
        np.add.at(log_p_patient_subtype, patient_index, log_p_visit_subtype)

        f = np.asarray(f, dtype=float).reshape(-1)
        log_f = np.log(f + eps)
        log_mix = log_p_patient_subtype + log_f[None, :]
        max_log = np.max(log_mix, axis=1, keepdims=True)
        log_p_patient = max_log[:, 0] + np.log(np.sum(np.exp(log_mix - max_log), axis=1) + eps)
        loglike = float(np.sum(log_p_patient))

        resp_patient = np.exp(log_mix - log_p_patient[:, None])
        resp_visit = resp_patient[patient_index]
        return loglike, resp_patient, resp_visit

    def _longitudinal_update_f(
        self,
        p_perm_k: np.ndarray,
        f: np.ndarray,
        patient_index: np.ndarray,
        n_patients: int,
    ) -> np.ndarray:
        _, resp_patient, _ = self._longitudinal_stats(p_perm_k, f, patient_index, n_patients)
        return np.mean(resp_patient, axis=0)

    def _longitudinal_loglike(
        self,
        p_perm_k: np.ndarray,
        f: np.ndarray,
        patient_index: np.ndarray,
        n_patients: int,
    ) -> float:
        loglike, _, _ = self._longitudinal_stats(p_perm_k, f, patient_index, n_patients)
        return loglike

    def _prepare_longitudinal_loglike_cache(
        self,
        p_perm_k: np.ndarray,
        f: np.ndarray,
        patient_index: np.ndarray,
        n_patients: int,
        subtype_index: int,
        *,
        eps: float = 1e-250,
    ) -> tuple[np.ndarray, float]:
        p_visit_subtype = np.sum(p_perm_k, axis=1)
        log_p_visit_subtype = np.log(p_visit_subtype + eps)

        log_p_patient_subtype = np.zeros((n_patients, p_visit_subtype.shape[1]), dtype=float)
        np.add.at(log_p_patient_subtype, patient_index, log_p_visit_subtype)

        f = np.asarray(f, dtype=float).reshape(-1)
        log_f = np.log(f + eps)
        log_mix = log_p_patient_subtype + log_f[None, :]

        n_subtypes = log_mix.shape[1]
        if n_subtypes == 1:
            logsumexp_other = np.full((n_patients,), -np.inf)
        else:
            mask = np.ones(n_subtypes, dtype=bool)
            mask[subtype_index] = False
            log_mix_other = log_mix[:, mask]
            max_other = np.max(log_mix_other, axis=1)
            sum_exp_other = np.sum(np.exp(log_mix_other - max_other[:, None]), axis=1)
            logsumexp_other = np.where(
                sum_exp_other > 0,
                max_other + np.log(sum_exp_other),
                -np.inf,
            )

        return logsumexp_other, float(log_f[subtype_index])

    def _longitudinal_loglike_cached(
        self,
        p_perm_k: np.ndarray,
        patient_index: np.ndarray,
        n_patients: int,
        subtype_index: int,
        logsumexp_other: np.ndarray,
        log_f_s: float,
        *,
        eps: float = 1e-250,
    ) -> float:
        p_visit_subtype_s = np.sum(p_perm_k[:, :, subtype_index], axis=1)
        log_p_visit_subtype_s = np.log(p_visit_subtype_s + eps)

        log_p_patient_subtype_s = np.zeros(n_patients, dtype=float)
        np.add.at(log_p_patient_subtype_s, patient_index, log_p_visit_subtype_s)

        log_mix_s = log_p_patient_subtype_s + log_f_s
        max_mix = np.maximum(logsumexp_other, log_mix_s)
        log_p_patient = max_mix + np.log(
            np.exp(logsumexp_other - max_mix) + np.exp(log_mix_s - max_mix) + eps
        )
        return float(np.sum(log_p_patient))

    def _longitudinal_loglike_cached_batch(
        self,
        possible_p_perm_k: np.ndarray,
        patient_index: np.ndarray,
        n_patients: int,
        subtype_index: int,
        logsumexp_other: np.ndarray,
        log_f_s: float,
        *,
        eps: float = 1e-250,
    ) -> np.ndarray:
        # possible_p_perm_k: (M, N+1, n_pos) for one subtype's candidate sequences
        n_pos = possible_p_perm_k.shape[2]

        # Sum over stages (axis=1): (M, n_pos)
        p_visit_s = np.sum(possible_p_perm_k, axis=1)
        log_p_visit_s = np.log(p_visit_s + eps)

        # Scatter-add visits → patients: (n_patients, n_pos)
        log_p_patient_s = np.zeros((n_patients, n_pos), dtype=float)
        np.add.at(log_p_patient_s, patient_index, log_p_visit_s)

        log_mix_s   = log_p_patient_s + log_f_s                          # (n_patients, n_pos)
        max_mix     = np.maximum(logsumexp_other[:, None], log_mix_s)    # (n_patients, n_pos)
        log_p_patient = max_mix + np.log(
            np.exp(logsumexp_other[:, None] - max_mix)
            + np.exp(log_mix_s - max_mix)
            + eps
        )
        return np.sum(log_p_patient, axis=0)                              # (n_pos,)


class LongitudinalZscoreSustain(_LongitudinalLikelihoodMixin, ZscoreSustain):
    """Z-score SuStaIn with patient-level subtype likelihood."""

    def __init__(self, *args, patient_ids: Sequence[object] | None = None, use_cache_recomputation: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache_recomputation = use_cache_recomputation
        if patient_ids is not None:
            numStages = self._ZscoreSustain__sustainData.getNumStages()
            data = self._ZscoreSustain__sustainData.data
            sustain_data = LongitudinalZScoreSustainData(data, numStages, patient_ids)
            self._ZscoreSustain__sustainData = sustain_data
            self._AbstractSustain__sustainData = sustain_data

    def _calculate_likelihood(self, sustainData, S, f):
        patient_index, n_patients = self._get_longitudinal_index(sustainData)
        if patient_index is None:
            return super()._calculate_likelihood(sustainData, S, f)

        M = sustainData.getNumSamples()
        N_S = S.shape[0]
        N = sustainData.getNumStages()

        p_perm_k = np.zeros((M, N + 1, N_S))
        for s in range(N_S):
            p_perm_k[:, :, s] = self._calculate_likelihood_stage(sustainData, S[s])

        loglike, _, resp_visit = self._longitudinal_stats(p_perm_k, f, patient_index, n_patients)
        total_prob_cluster = resp_visit
        total_prob_stage = np.sum(p_perm_k * resp_visit[:, None, :], axis=2)
        # Use prior f (not resp_visit) so total_prob_subj is the marginal likelihood
        # required by CVIC. resp_visit is the posterior, which biases CVIC toward MV-SuStaIn.
        f_arr = np.asarray(f, dtype=float).reshape(-1)
        total_prob_subj = np.sum(p_perm_k * f_arr[None, None, :], axis=(1, 2))
        return loglike, total_prob_subj, total_prob_stage, total_prob_cluster, p_perm_k

    def _optimise_parameters(self, sustainData, S_init, f_init, rng):
        patient_index, n_patients = self._get_longitudinal_index(sustainData)
        # Stacked path routes through StackedZscoreSustain; patient_index is always
        # non-None here (LongitudinalZscoreSustain always has patient_ids set).
        if patient_index is None:
            raise RuntimeError(
                "LongitudinalZscoreSustain._optimise_parameters called without patient_ids. "
                "Use StackedZscoreSustain for the stacked (patient_ids=None) path."
            )

        M = sustainData.getNumSamples()
        N_S = S_init.shape[0]
        N = self.stage_zscore.shape[1]

        S_opt = S_init.copy()
        f_opt = np.asarray(f_init, dtype=float).reshape(N_S)
        p_perm_k = np.zeros((M, N + 1, N_S))

        for s in range(N_S):
            p_perm_k[:, :, s] = self._calculate_likelihood_stage(sustainData, S_opt[s])

        f_opt = self._longitudinal_update_f(p_perm_k, f_opt, patient_index, n_patients)
        order_seq = rng.permutation(N_S)
        use_cache = getattr(self, 'use_cache_recomputation', True)

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

    def subtype_and_stage_individuals_newData(
        self,
        data_new,
        samples_sequence,
        samples_f,
        N_samples,
        patient_ids: Sequence[object] | None = None,
    ):
        if patient_ids is None:
            return super().subtype_and_stage_individuals_newData(
                data_new,
                samples_sequence,
                samples_f,
                N_samples,
            )

        data_new = np.asarray(data_new, dtype=float)
        if data_new.ndim != 2:
            raise ValueError("data_new must be 2D.")
        patient_ids_arr = np.asarray(patient_ids)
        if patient_ids_arr.shape[0] != data_new.shape[0]:
            raise ValueError("patient_ids length must match number of new samples.")

        numStages_new = self._ZscoreSustain__sustainData.getNumStages()
        sustainData_newData = LongitudinalZScoreSustainData(data_new, numStages_new, patient_ids_arr)

        return self.subtype_and_stage_individuals(
            sustainData_newData,
            samples_sequence,
            samples_f,
            N_samples,
        )


class LongitudinalOrdinalSustain(_LongitudinalLikelihoodMixin, OrdinalSustain):
    """Ordinal SuStaIn with patient-level subtype likelihood."""

    def __init__(self, *args, patient_ids: Sequence[object] | None = None, use_cache_recomputation: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache_recomputation = use_cache_recomputation
        if patient_ids is not None:
            numStages = self._OrdinalSustain__sustainData.getNumStages()
            prob_nl = self._OrdinalSustain__sustainData.prob_nl
            prob_score = self._OrdinalSustain__sustainData.prob_score
            sustain_data = LongitudinalOrdinalSustainData(prob_nl, prob_score, numStages, patient_ids)
            self._OrdinalSustain__sustainData = sustain_data
            self._AbstractSustain__sustainData = sustain_data

    def _calculate_likelihood(self, sustainData, S, f):
        patient_index, n_patients = self._get_longitudinal_index(sustainData)
        if patient_index is None:
            return super()._calculate_likelihood(sustainData, S, f)

        M = sustainData.getNumSamples()
        N_S = S.shape[0]
        N = sustainData.getNumStages()

        p_perm_k = np.zeros((M, N + 1, N_S))
        for s in range(N_S):
            p_perm_k[:, :, s] = self._calculate_likelihood_stage(sustainData, S[s])

        loglike, _, resp_visit = self._longitudinal_stats(p_perm_k, f, patient_index, n_patients)
        total_prob_cluster = resp_visit
        total_prob_stage = np.sum(p_perm_k * resp_visit[:, None, :], axis=2)
        # Use prior f (not resp_visit) so total_prob_subj is the marginal likelihood
        # required by CVIC. resp_visit is the posterior, which biases CVIC toward MV-SuStaIn.
        f_arr = np.asarray(f, dtype=float).reshape(-1)
        total_prob_subj = np.sum(p_perm_k * f_arr[None, None, :], axis=(1, 2))
        return loglike, total_prob_subj, total_prob_stage, total_prob_cluster, p_perm_k

    def _optimise_parameters(self, sustainData, S_init, f_init, rng):
        patient_index, n_patients = self._get_longitudinal_index(sustainData)
        # Stacked path routes through StackedOrdinalSustain; patient_index is always
        # non-None here (LongitudinalOrdinalSustain always has patient_ids set).
        if patient_index is None:
            raise RuntimeError(
                "LongitudinalOrdinalSustain._optimise_parameters called without patient_ids. "
                "Use StackedOrdinalSustain for the stacked (patient_ids=None) path."
            )

        M = sustainData.getNumSamples()
        N_S = S_init.shape[0]
        N = self.stage_score.shape[1]

        S_opt = S_init.copy()
        f_opt = np.asarray(f_init, dtype=float).reshape(N_S)
        p_perm_k = np.zeros((M, N + 1, N_S))

        for s in range(N_S):
            p_perm_k[:, :, s] = self._calculate_likelihood_stage(sustainData, S_opt[s])

        f_opt = self._longitudinal_update_f(p_perm_k, f_opt, patient_index, n_patients)
        order_seq = rng.permutation(N_S)
        use_cache = getattr(self, 'use_cache_recomputation', True)

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

    def subtype_and_stage_individuals_newData(
        self,
        prob_nl_new,
        prob_score_new,
        samples_sequence,
        samples_f,
        N_samples,
        patient_ids: Sequence[object] | None = None,
    ):
        if patient_ids is None:
            return super().subtype_and_stage_individuals_newData(
                prob_nl_new,
                prob_score_new,
                samples_sequence,
                samples_f,
                N_samples,
            )

        prob_nl_new = np.asarray(prob_nl_new, dtype=float)
        prob_score_new = np.asarray(prob_score_new, dtype=float)
        if prob_nl_new.ndim != 2:
            raise ValueError("prob_nl_new must be 2D.")
        if prob_score_new.ndim != 3:
            raise ValueError("prob_score_new must be 3D.")

        patient_ids_arr = np.asarray(patient_ids)
        if patient_ids_arr.shape[0] != prob_nl_new.shape[0]:
            raise ValueError("patient_ids length must match number of new samples.")

        numBio_new = prob_nl_new.shape[1]
        if numBio_new != self._OrdinalSustain__sustainData.getNumBiomarkers():
            raise ValueError("Number of biomarkers in new data must match training data.")

        numStages = self._OrdinalSustain__sustainData.getNumStages()
        prob_score_new = prob_score_new.transpose(0, 2, 1)
        prob_score_new = prob_score_new.reshape(
            prob_score_new.shape[0],
            prob_score_new.shape[1] * prob_score_new.shape[2],
        )
        prob_score_new = prob_score_new[:, self.IX_select[0, :]]
        prob_score_new = prob_score_new.reshape(prob_nl_new.shape[0], self.stage_score.shape[1])

        sustainData_newData = LongitudinalOrdinalSustainData(
            prob_nl_new,
            prob_score_new,
            numStages,
            patient_ids_arr,
        )

        return self.subtype_and_stage_individuals(
            sustainData_newData,
            samples_sequence,
            samples_f,
            N_samples,
        )


class LongitudinalMixtureSustain(_LongitudinalLikelihoodMixin, MixtureSustain):
    """Mixture SuStaIn with patient-level subtype likelihood."""

    def __init__(self, *args, patient_ids: Sequence[object] | None = None, use_cache_recomputation: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache_recomputation = use_cache_recomputation
        if patient_ids is not None:
            numStages = self._MixtureSustain__sustainData.getNumStages()
            L_yes = self._MixtureSustain__sustainData.L_yes
            L_no = self._MixtureSustain__sustainData.L_no
            sustain_data = LongitudinalMixtureSustainData(L_yes, L_no, numStages, patient_ids)
            self._MixtureSustain__sustainData = sustain_data
            self._AbstractSustain__sustainData = sustain_data

    def _calculate_likelihood(self, sustainData, S, f):
        patient_index, n_patients = self._get_longitudinal_index(sustainData)
        if patient_index is None:
            return super()._calculate_likelihood(sustainData, S, f)

        M = sustainData.getNumSamples()
        N_S = S.shape[0]
        N = sustainData.getNumStages()

        p_perm_k = np.zeros((M, N + 1, N_S))
        for s in range(N_S):
            p_perm_k[:, :, s] = self._calculate_likelihood_stage(sustainData, S[s])

        loglike, _, resp_visit = self._longitudinal_stats(p_perm_k, f, patient_index, n_patients)
        total_prob_cluster = resp_visit
        total_prob_stage = np.sum(p_perm_k * resp_visit[:, None, :], axis=2)
        # Use prior f (not resp_visit) so total_prob_subj is the marginal likelihood
        # required by CVIC. resp_visit is the posterior, which biases CVIC toward MV-SuStaIn.
        f_arr = np.asarray(f, dtype=float).reshape(-1)
        total_prob_subj = np.sum(p_perm_k * f_arr[None, None, :], axis=(1, 2))
        return loglike, total_prob_subj, total_prob_stage, total_prob_cluster, p_perm_k

    def _optimise_parameters(self, sustainData, S_init, f_init, rng):
        patient_index, n_patients = self._get_longitudinal_index(sustainData)
        if patient_index is None and not getattr(self, 'use_cache_recomputation', True):
            return super()._optimise_parameters(sustainData, S_init, f_init, rng)
        if patient_index is None:
            # stacked + cache: trivial patient_index (each visit is its own patient)
            M_tmp = sustainData.getNumSamples()
            patient_index = np.arange(M_tmp, dtype=np.int32)
            n_patients = M_tmp

        M = sustainData.getNumSamples()
        N_S = S_init.shape[0]
        N = sustainData.getNumStages()

        S_opt = S_init.copy()
        f_opt = np.asarray(f_init, dtype=float).reshape(N_S)
        p_perm_k = np.zeros((M, N + 1, N_S))

        for s in range(N_S):
            p_perm_k[:, :, s] = self._calculate_likelihood_stage(sustainData, S_opt[s])

        f_opt = self._longitudinal_update_f(p_perm_k, f_opt, patient_index, n_patients)
        order_seq = rng.permutation(N_S)
        use_cache = getattr(self, 'use_cache_recomputation', True)

        for s in order_seq:
            if use_cache:
                logsumexp_other, log_f_s = self._prepare_longitudinal_loglike_cache(
                    p_perm_k, f_opt, patient_index, n_patients, s
                )
            order_bio = rng.permutation(N)
            if self.use_dp:
                for i in order_bio:
                    current_sequence = S_opt[s]
                    assert len(current_sequence) == N
                    current_location = np.zeros(N, dtype=int)
                    current_location[current_sequence.astype(int)] = np.arange(N)

                    selected_event = i
                    move_event_from = current_location[selected_event]

                    possible_likelihood = np.zeros((N, 1))
                    possible_p_perm_k = np.zeros((M, N + 1, N))

                    current_sequence = np.delete(current_sequence, move_event_from, 0)
                    new_sequence = np.append(current_sequence, selected_event)

                    temp_p_perm_k, cp_yes, cp_no, cp_no_org = self._calculate_likelihood_subset(
                        sustainData.L_yes, sustainData.L_no, new_sequence
                    )
                    p_perm_k[:, :, s] = temp_p_perm_k
                    possible_p_perm_k[:, :, N - 1] = temp_p_perm_k

                    if use_cache:
                        possible_likelihood[N - 1] = self._longitudinal_loglike_cached(
                            p_perm_k,
                            patient_index,
                            n_patients,
                            s,
                            logsumexp_other,
                            log_f_s,
                        )
                    else:
                        possible_likelihood[N - 1] = self._longitudinal_loglike(
                            p_perm_k, f_opt, patient_index, n_patients
                        )

                    for position in range(N - 2, -1, -1):
                        temp_p_perm_k, cp_yes, cp_no, _ = self._calculate_likelihood_subset(
                            sustainData.L_yes,
                            sustainData.L_no,
                            new_sequence,
                            selected_event,
                            position,
                            cp_yes,
                            cp_no,
                            cp_no_org,
                        )
                        p_perm_k[:, :, s] = temp_p_perm_k
                        possible_p_perm_k[:, :, position] = temp_p_perm_k
                        if use_cache:
                            possible_likelihood[position] = self._longitudinal_loglike_cached(
                                p_perm_k,
                                patient_index,
                                n_patients,
                                s,
                                logsumexp_other,
                                log_f_s,
                            )
                        else:
                            possible_likelihood[position] = self._longitudinal_loglike(
                                p_perm_k, f_opt, patient_index, n_patients
                            )

                    max_i = np.argmax(possible_likelihood)
                    S = np.insert(current_sequence, max_i, selected_event)
                    max_p_perm_k = possible_p_perm_k[:, :, max_i]

                    S_opt[s] = S
                    p_perm_k[:, :, s] = max_p_perm_k
            else:
                for i in order_bio:
                    current_sequence = S_opt[s]
                    assert len(current_sequence) == N
                    current_location = np.array([0] * N)
                    current_location[current_sequence.astype(int)] = np.arange(N)

                    selected_event = i
                    move_event_from = current_location[selected_event]

                    possible_positions = np.arange(N)
                    possible_sequences = np.zeros((len(possible_positions), N))
                    possible_likelihood = np.zeros((len(possible_positions), 1))
                    possible_p_perm_k = np.zeros((M, N + 1, len(possible_positions)))
                    for index in range(len(possible_positions)):
                        current_sequence = S_opt[s]
                        move_event_to = possible_positions[index]
                        current_sequence = np.delete(current_sequence, move_event_from, 0)
                        new_sequence = np.concatenate(
                            [current_sequence[np.arange(move_event_to)], [selected_event], current_sequence[np.arange(move_event_to, N - 1)]]
                        )
                        possible_sequences[index, :] = new_sequence

                        possible_p_perm_k[:, :, index] = self._calculate_likelihood_stage(sustainData, new_sequence)

                        p_perm_k[:, :, s] = possible_p_perm_k[:, :, index]
                        if use_cache:
                            possible_likelihood[index] = self._longitudinal_loglike_cached(
                                p_perm_k,
                                patient_index,
                                n_patients,
                                s,
                                logsumexp_other,
                                log_f_s,
                            )
                        else:
                            possible_likelihood[index] = self._longitudinal_loglike(
                                p_perm_k, f_opt, patient_index, n_patients
                            )

                    possible_likelihood = possible_likelihood.reshape(possible_likelihood.shape[0])
                    max_likelihood = np.max(possible_likelihood)
                    this_S = possible_sequences[possible_likelihood == max_likelihood, :]
                    this_S = this_S[0, :]
                    S_opt[s] = this_S
                    this_p_perm_k = possible_p_perm_k[:, :, possible_likelihood == max_likelihood]
                    p_perm_k[:, :, s] = this_p_perm_k[:, :, 0]

                S_opt[s] = this_S

        f_opt = self._longitudinal_update_f(p_perm_k, f_opt, patient_index, n_patients)
        likelihood_opt = self._longitudinal_loglike(p_perm_k, f_opt, patient_index, n_patients)
        return S_opt, f_opt, likelihood_opt

    def _perform_mcmc(self, sustainData, seq_init, f_init, n_iterations, seq_sigma, f_sigma):
        patient_index, n_patients = self._get_longitudinal_index(sustainData)
        if patient_index is None:
            return super()._perform_mcmc(sustainData, seq_init, f_init, n_iterations, seq_sigma, f_sigma)

        M = sustainData.getNumSamples()
        N = sustainData.getNumStages()
        N_S = seq_init.shape[0]

        if isinstance(f_sigma, float):
            f_sigma = np.array([f_sigma])

        samples_sequence = np.zeros((N_S, N, n_iterations))
        samples_f = np.zeros((N_S, n_iterations))
        samples_likelihood = np.zeros((n_iterations, 1))
        samples_sequence[:, :, 0] = seq_init
        samples_f[:, 0] = f_init

        tqdm_update_iters = int(n_iterations / 1000) if n_iterations > 100000 else None

        for i in tqdm(range(n_iterations), "MCMC Iteration", n_iterations, miniters=tqdm_update_iters):
            if i > 0:
                seq_order = self.global_rng.permutation(N_S)
                move_event_from = np.ceil(N * self.global_rng.random(len(seq_order))).astype(int) - 1
                current_sequence = samples_sequence[seq_order, :, i - 1]

                selected_event = current_sequence[np.arange(current_sequence.shape[0]), move_event_from]

                distance = np.arange(N) + np.zeros((len(seq_order), 1)) - move_event_from[:, np.newaxis]
                weight = self.calc_coeff(seq_sigma) * self.calc_exp(distance, 0.0, seq_sigma)
                weight = np.divide(weight, weight.sum(1)[:, None])
                index = [self.global_rng.choice(np.arange(len(row)), 1, replace=True, p=row)[0] for row in weight]
                move_event_to = np.arange(N)[index]

                new_seq = current_sequence.copy()
                new_seq[np.arange(len(seq_order)), move_event_from] = new_seq[np.arange(len(seq_order)), move_event_to]
                new_seq[np.arange(len(seq_order)), move_event_to] = selected_event
                samples_sequence[seq_order, :, i] = new_seq

                new_f = samples_f[:, i - 1] + f_sigma * self.global_rng.standard_normal()
                new_f = (np.fabs(new_f) / np.sum(np.fabs(new_f)))
                samples_f[:, i] = new_f

            S = samples_sequence[:, :, i]
            p_perm_k = np.zeros((M, N + 1, N_S))
            for s in range(N_S):
                p_perm_k[:, :, s] = self._calculate_likelihood_stage(sustainData, S[s, :])

            likelihood_sample = self._longitudinal_loglike(p_perm_k, samples_f[:, i], patient_index, n_patients)
            samples_likelihood[i] = likelihood_sample

            if i > 0:
                ratio = np.exp(samples_likelihood[i] - samples_likelihood[i - 1])
                if ratio < self.global_rng.random():
                    samples_likelihood[i] = samples_likelihood[i - 1]
                    samples_sequence[:, :, i] = samples_sequence[:, :, i - 1]
                    samples_f[:, i] = samples_f[:, i - 1]

        perm_index = np.where(samples_likelihood == np.max(samples_likelihood))
        perm_index = perm_index[0][0]
        ml_likelihood = np.max(samples_likelihood)
        ml_sequence = samples_sequence[:, :, perm_index]
        ml_f = samples_f[:, perm_index]
        return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood

    def subtype_and_stage_individuals_newData(
        self,
        L_yes_new,
        L_no_new,
        samples_sequence,
        samples_f,
        N_samples,
        patient_ids: Sequence[object] | None = None,
    ):
        if patient_ids is None:
            return super().subtype_and_stage_individuals_newData(
                L_yes_new,
                L_no_new,
                samples_sequence,
                samples_f,
                N_samples,
            )

        L_yes_new = np.asarray(L_yes_new, dtype=float)
        L_no_new = np.asarray(L_no_new, dtype=float)
        if L_yes_new.ndim != 2 or L_no_new.ndim != 2:
            raise ValueError("L_yes_new and L_no_new must be 2D.")
        if L_yes_new.shape != L_no_new.shape:
            raise ValueError("L_yes_new and L_no_new must have the same shape.")

        patient_ids_arr = np.asarray(patient_ids)
        if patient_ids_arr.shape[0] != L_yes_new.shape[0]:
            raise ValueError("patient_ids length must match number of new samples.")

        numStages_new = L_yes_new.shape[1]
        if numStages_new != self._MixtureSustain__sustainData.getNumStages():
            raise ValueError("Number of stages in new data should match training data.")

        sustainData_newData = LongitudinalMixtureSustainData(
            L_yes_new,
            L_no_new,
            numStages_new,
            patient_ids_arr,
        )

        return self.subtype_and_stage_individuals(
            sustainData_newData,
            samples_sequence,
            samples_f,
            N_samples,
        )

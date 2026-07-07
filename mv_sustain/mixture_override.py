"""
Optimized wrappers around pySuStaIn MixtureSustain.
"""

from __future__ import annotations

import numpy as np

from pySuStaIn.MixtureSustain import MixtureSustain as _MixtureSustain  # type: ignore


class MixtureSustain(_MixtureSustain):
    """Mixture SuStaIn with reduced redundant recomputation."""

    def _optimise_mcmc_settings(self, sustainData, seq_init, f_init):
        # Optimise the perturbation size for the MCMC algorithm.
        n_iterations_MCMC_optimisation = max(1, int(self.N_iterations_MCMC))

        n_passes_optimisation = 3
        seq_sigma_currentpass = 1
        f_sigma_currentpass = 0.01  # magic number

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

    def _optimise_parameters(self, sustainData, S_init, f_init, rng):
        # Optimise the parameters of the SuStaIn model

        M                                   = sustainData.getNumSamples()
        N_S                                 = S_init.shape[0]
        N                                   = sustainData.getNumStages()

        S_opt                               = S_init.copy()  # have to copy or changes will be passed to S_init
        f_opt                               = np.array(f_init).reshape(N_S, 1, 1)
        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))
        p_perm_k                            = np.zeros((M, N + 1, N_S))

        for s in range(N_S):
            p_perm_k[:, :, s]               = self._calculate_likelihood_stage(sustainData, S_opt[s])

        p_perm_k_weighted                   = p_perm_k * f_val_mat
        # the second summation axis is different to Matlab version
        #p_perm_k_norm                       = p_perm_k_weighted / np.tile(np.sum(np.sum(p_perm_k_weighted, 1), 1).reshape(M, 1, 1), (1, N + 1, N_S))
        # adding 1e-250 fixes divide by zero problem that happens rarely
        p_perm_k_norm                       = p_perm_k_weighted / np.sum(p_perm_k_weighted + 1e-250, axis=(1, 2), keepdims=True)

        f_opt                               = (np.squeeze(np.sum(p_perm_k_norm, axis = (1, 0))) / np.sum(p_perm_k_norm)).reshape(N_S, 1, 1)
        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))
        order_seq                           = rng.permutation(N_S)    #np.random.permutation(N_S)  # this will produce different random numbers to Matlab

        for s in order_seq:
            other_prob_stage                = np.sum(p_perm_k * f_val_mat, 2) - p_perm_k[:, :, s] * f_val_mat[:, :, s]
            order_bio                       = rng.permutation(N) #np.random.permutation(N)  # this will produce different random numbers to Matlab
            # optimised version
            if self.use_dp:
                for i in order_bio:
                    current_sequence = S_opt[s]
                    assert(len(current_sequence) == N)
                    current_location = np.zeros(N, dtype = int)
                    current_location[current_sequence.astype(int)] = np.arange(N)

                    selected_event = i
                    move_event_from = current_location[selected_event]

                    possible_likelihood = np.zeros((N, 1))
                    possible_p_perm_k = np.zeros((M, N + 1, N))

                    current_sequence = np.delete(current_sequence, move_event_from, 0)
                    new_sequence = np.append(current_sequence, selected_event)

                    temp_p_perm_k, cp_yes, cp_no, cp_no_org = self._calculate_likelihood_subset(sustainData.L_yes, sustainData.L_no, new_sequence)
                    p_perm_k[:, :, s] = temp_p_perm_k
                    possible_p_perm_k[:, :, N - 1] = temp_p_perm_k

                    total_prob_stage = other_prob_stage + p_perm_k[:, :, s] * f_val_mat[:, :, s]
                    total_prob_subj = np.sum(total_prob_stage, axis=1)
                    possible_likelihood[N - 1] = np.sum(np.log(total_prob_subj + 1e-250))

                    for position in range(N - 2, -1, -1):
                        temp_p_perm_k, cp_yes, cp_no, _ = self._calculate_likelihood_subset(sustainData.L_yes, sustainData.L_no, 
                                                                                            new_sequence, selected_event,
                                                                                            position, cp_yes, cp_no, 
                                                                                            cp_no_org)
                        
                        # calculate log_likelihood
                        p_perm_k[:, :, s] = temp_p_perm_k
                        possible_p_perm_k[:, :, position] = temp_p_perm_k
                        total_prob_stage = other_prob_stage + p_perm_k[:, :, s] * f_val_mat[:, :, s]
                        total_prob_subj = np.sum(total_prob_stage, 1)
                        possible_likelihood[position] = np.sum(np.log(total_prob_subj + 1e-250))

                    max_i = np.argmax(possible_likelihood)
                    S = np.insert(current_sequence, max_i, selected_event)
                    max_likelihood = possible_likelihood[max_i]
                    max_p_perm_k = possible_p_perm_k[:, :, max_i]

                    S_opt[s] = S
                    p_perm_k[:, :, s] = max_p_perm_k                   

            else:
                # original version
                for i in order_bio:
                    current_sequence            = S_opt[s]
                    assert(len(current_sequence)==N)
                    current_location            = np.array([0] * N)
                    current_location[current_sequence.astype(int)] = np.arange(N)

                    selected_event              = i

                    move_event_from             = current_location[selected_event]

                    possible_positions          = np.arange(N)
                    possible_sequences          = np.zeros((len(possible_positions), N))
                    possible_likelihood         = np.zeros((len(possible_positions), 1))
                    possible_p_perm_k           = np.zeros((M, N + 1, len(possible_positions)))
                    for index in range(len(possible_positions)):
                        current_sequence        = S_opt[s]

                        #choose a position in the sequence to move an event to
                        move_event_to           = possible_positions[index]

                        #move this event in its new position
                        current_sequence        = np.delete(current_sequence, move_event_from, 0)  # this is different to the Matlab version, which call current_sequence(move_event_from) = []
                        new_sequence            = np.concatenate([current_sequence[np.arange(move_event_to)], [selected_event], current_sequence[np.arange(move_event_to, N - 1)]])
                        possible_sequences[index, :] = new_sequence

                        possible_p_perm_k[:, :, index] = self._calculate_likelihood_stage(sustainData, new_sequence)

                        p_perm_k[:, :, s]       = possible_p_perm_k[:, :, index]
                        total_prob_stage        = other_prob_stage + p_perm_k[:, :, s] * f_val_mat[:, :, s]
                        total_prob_subj         = np.sum(total_prob_stage, 1)
                        possible_likelihood[index] = np.sum(np.log(total_prob_subj + 1e-250))

                    possible_likelihood         = possible_likelihood.reshape(possible_likelihood.shape[0])
                    max_likelihood              = np.max(possible_likelihood)
                    this_S                      = possible_sequences[possible_likelihood == max_likelihood, :]
                    this_S                      = this_S[0, :]
                    S_opt[s]                    = this_S
                    this_p_perm_k               = possible_p_perm_k[:, :, possible_likelihood == max_likelihood]
                    p_perm_k[:, :, s]           = this_p_perm_k[:, :, 0]

                S_opt[s]                        = this_S

        p_perm_k_weighted                   = p_perm_k * f_val_mat
        p_perm_k_norm                       = p_perm_k_weighted / np.tile(np.sum(np.sum(p_perm_k_weighted, 1), 1).reshape(M, 1, 1), (1, N + 1, N_S))  # the second summation axis is different to Matlab version
        f_opt                               = (np.squeeze(np.sum(p_perm_k_norm, axis = (1, 0))) / np.sum(p_perm_k_norm)).reshape(N_S, 1, 1)

        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))

        f_opt                               = f_opt.reshape(N_S)
        total_prob_stage                    = np.sum(p_perm_k * f_val_mat, 2)
        total_prob_subj                     = np.sum(total_prob_stage, 1)

        likelihood_opt                      = np.sum(np.log(total_prob_subj + 1e-250))

        return S_opt, f_opt, likelihood_opt

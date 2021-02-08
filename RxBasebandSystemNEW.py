import numpy as np


class RxBasebandSystem:
    def __init__(self, multi_ant_sys, caz):
        self.SNR = multi_ant_sys.SNR_lin
        self.num_ant = multi_ant_sys.num_ant_txrx
        self.correlation_obs = 0
        self.NFFT = multi_ant_sys.NFFT
        self.NCP = multi_ant_sys.len_CP

        self.synch_used_bins = multi_ant_sys.used_bins_synch

        self.start_sample = multi_ant_sys.len_CP - 4
        self.stride_value = int(np.round(multi_ant_sys.len_CP/2))
        self.rx_buffer_time = multi_ant_sys.buffer_data_rx_time
        self.rx_buffer_length = int(np.size(multi_ant_sys.buffer_data_rx_time))
        self.loop_z = int(np.ceil(np.size(multi_ant_sys.buffer_data_rx_time) / self.stride_value))
        self.d_long = np.zeros([self.loop_z, 1], dtype=complex)
        self.norm_q = np.conj(caz.ZChu0) * caz.ZChu0
        self.ptr_adj = 0
        self.p = 0
        self.sym_count = 0
        self.tap_delay = 5
        self.MM = [1, 62]
        self.synch_data_shape = multi_ant_sys.symbol_pattern

        self.rx_buffer_time_window = np.zeros([1, self.NFFT], dtype=complex)

        self.synch_reference = caz.ZChu0

        self.genie_chan_time = multi_ant_sys.genie_chan_time
        self.genie_chan_freq = multi_ant_sys.channel_freq

        self.time_synch_reference = np.zeros((multi_ant_sys.num_ant_txrx, 250, 3), dtype=complex)
        self.est_chan_freq_p = np.zeros((self.num_ant, self.rx_buffer_length, self.NFFT))
        self.est_chan_freq_n = np.zeros((self.num_ant, self.rx_buffer_length, self.NFFT - 2))

        self.est_chan_impulse = np.zeros((self.num_ant, self.rx_buffer_length, self.NFFT))
        self.est_synch_freq = np.zeros((self.num_ant, self.rx_buffer_length, self.NFFT - 2))

    def param_est_synch(self):
        ptr_adj = 0
        ptr_frame = 0

        # This next line have no impact on code completion. ptr_synch_0 is used as a pointer buffer to compare genie channel and the final channel estimation results.
        # ptr_synch_0 = 0

        b = 0

        x = np.zeros([1, self.tap_delay], dtype=float)
        xp = np.zeros((2, 2), dtype=float)

        for m in range(1):

            # This next line have no impact on code completion. chan_q is used as a reference buffer to compare genie channel and the final channel estimation results.
            # chan_q = np.reshape(self.genie_chan_time[m, 0, :], (np.size(self.genie_chan_time[m, 0, :]), 1))

            while self.p <= self.loop_z:
                self.p += 1

                if self.correlation_obs == 0:
                    ptr_frame = (self.p - 1) * self.stride_value + self.start_sample + self.ptr_adj
                elif self.correlation_obs < 5:
                    ptr_frame = ptr_frame + np.sum(self.synch_data_shape) * (self.NCP + self.NFFT)
                else:
                    ptr_frame = np.round(xp[-1, :] * b - self.NCP/4)

                if (self.MM[0]-1) * self.rx_buffer_length + self.NFFT + ptr_frame - 1 < np.size(self.rx_buffer_time):
                    for LL in range(self.MM[0]):
                        self.rx_buffer_time_window[LL * self.NFFT+1: (LL+1)*self.NFFT] = self.rx_buffer_time[m, (LL * self.rx_buffer_length + ptr_frame): (LL + 1) * self.rx_buffer_length + ptr_frame + self.NFFT]
                    # If Data Use Freq Data
                    tmp_1_vector = np.zeros([self.MM[0], self.NFFT], dtype=float)

                    for LL in range(self.MM[0]):
                        tmp_1_vector[LL, 0:self.NFFT] = np.fft.fft(self.rx_buffer_time[LL * self.NFFT + 1: (LL + 1)*self.NFFT], self.NFFT)

                    # If Synch Use Synch Data
                    synch_data_00 = tmp_1_vector[:, self.synch_used_bins]
                    synch_data_0 = np.transpose(synch_data_00).reshape((1, synch_data_00.size))
                    power_est = np.sum(np.dot(synch_data_0, np.conj(synch_data_0)))/len(synch_data_0)
                    synch_data = synch_data_0/np.sqrt(power_est)

                    # Genie: Actual Frequency Channel
                    # These next three lines have no impact on code completion. They are used as references to the final channel estimation results.
                    # chan_freq_0 = np.reshape(self.genie_chan_freq[self.synch_used_bins], (1, self.synch_used_bins.size))
                    # chan_freq = np.tile(chan_freq_0, (1, self.MM[0]))
                    # est_synch_data = np.dot(np.reshape(self.synch_reference, (1, self.synch_reference.size)), np.reshape(chan_freq, (1, chan_freq.size)))

                    # Delay Estimation
                    p_matrix_0 = np.exp(1j * 2 * (np.pi / self.NFFT) * np.dot(np.transpose(self.synch_used_bins - 1), range(0, self.NCP)))
                    p_matrix = np.tile(p_matrix_0, (self.MM[0], 1))
                    delay_matrix = np.conj(self.synch_reference) * np.diag(self.synch_data_shape) * p_matrix
                    dmax, dmaxind0 = np.max(np.abs(delay_matrix))
                    # Difference in MATLAB and Python Code
                    dmax_ind = dmaxind0 - 0  # FIXME: -1 or -0?

                    self.d_long[self.p] = dmax
                    if dmax > 0.5 * len(self.synch_data_shape) or self.correlation_obs > 0:
                        if dmax_ind > np.round(0.75*self.NCP):
                            if self.correlation_obs == 0:
                                ptr_adj = ptr_adj + np.round(self.NCP * 0.5)
                                ptr_frame = (self.p-1) * self.stride_value + self.start_sample + ptr_adj
                            elif self.correlation_obs < 5:
                                ptr_frame = ptr_frame + np.round(self.NCP * 0.5)
                            for LL in range(self.MM[0]):
                                self.rx_buffer_time_window[LL * self.NFFT + 1: (LL+1)*self.NFFT] = self.rx_buffer_time[m, LL*self.rx_buffer_length + ptr_frame: (LL+1) * self.rx_buffer_length + ptr_frame + self.NFFT - 1]
                            tmp_1_vector = np.zeros([self.MM[0], self.NFFT], dtype=float)
                            for LL in range(self.MM[0]):
                                tmp_1_vector[LL, 0:self.NFFT] = np.fft.fft(self.rx_buffer_time_window[LL * self.NFFT + 1: (LL + 1) * self.NFFT], self.NFFT)
                            synch_data_00 = tmp_1_vector[:, self.synch_used_bins]
                            synch_data_0 = np.transpose(synch_data_00).reshape((1, synch_data_00.size))
                            power_est = np.sum(np.dot(synch_data_0, np.conj(synch_data_0))) / len(synch_data_0)
                            synch_data = synch_data_0 / np.sqrt(power_est)

                            # Genie: Actual Frequency Channel
                            # These next three lines have no impact on code completion. They are used as references to the final channel estimation results.
                            # chan_freq_0 = np.reshape(self.genie_chan_freq[self.synch_used_bins], (1, self.synch_used_bins.size))
                            # chan_freq = np.tile(chan_freq_0, (1, self.MM[0]))
                            # est_synch_data = np.dot(np.reshape(self.synch_reference, (1, self.synch_reference.size)), np.reshape(chan_freq, (1, chan_freq.size)))

                            p_matrix_0 = np.exp(1j * 2 * (np.pi / self.NFFT) * np.dot(np.transpose(self.synch_used_bins - 1), range(0, self.NCP)))
                            p_matrix = np.tile(p_matrix_0, (self.MM[0], 1))
                            delay_matrix = np.conj(self.synch_reference) * np.diag(self.synch_data_shape) * p_matrix
                            dmax, dmaxind0 = np.max(np.abs(delay_matrix))

                            # Difference in MATLAB and Python Code
                            # dmax_ind = dmaxind0 - 1  # FIXME: -1 or +0?

                            self.d_long[self.p] = dmax

                        time_synch_index = self.time_synch_reference[m, np.max(self.correlation_obs), 0]
                        if (ptr_frame - time_synch_index) > (2 * self.NCP + self.NFFT + 1) or self.correlation_obs == 0:
                            self.correlation_obs += 1

                            # This next line have no impact on code completion. Commenting them out is just for sake of comprehensive conversion of the source code.
                            # atmp, max_d = np.abs(chan_q).max(axis=1), np.abs(chan_q).argmax(axis=1)  # FIXME: Fix Axis

                            self.time_synch_reference[m, self.correlation_obs, 0] = ptr_frame
                            self.time_synch_reference[m, self.correlation_obs, 1] = dmax_ind
                            self.time_synch_reference[m, self.correlation_obs, 2] = dmax

                            # This next line have no impact on code completion. Commenting them out is just for sake of comprehensive conversion of the source code.
                            # max_dest = dmax

                            ptr_synch_0 = np.sum(self.time_synch_reference[m, self.correlation_obs, 0:2])
                            x[np.remainder(self.sym_count, self.tap_delay) + 1] = self.sym_count * np.sum(synch_data)
                            self.sym_count += 1

                            x2 = x[0:np.min(self.correlation_obs, self.tap_delay)]
                            xplus = [x2, self.sym_count * np.sum(self.synch_data_shape)]
                            xp = [np.ones(np.size(xplus)), np.transpose(xplus)]

                            if self.correlation_obs > 3:
                                y = np.transpose(ptr_synch_0[0:np.min(self.tap_delay, self.correlation_obs)])
                                x = [np.ones(np.size(x2)), np.transpose(x2)]

                                b = x / y

                            data_recovered = np.diag(synch_data) * p_matrix[:, dmax_ind + 1]  # FIXME: +1 or -1?

                            chan_est_1 = np.zeros((self.NFFT, 1))

                            tmp_v1 = data_recovered * np.transpose(self.synch_reference) / (1 + 1/self.SNR)

                            ll = np.size(tmp_v1) / self.MM[0]
                            chan_est_00 = np.reshape(tmp_v1, (ll, self.MM[0]))
                            chan_est_0 = np.transpose(chan_est_00)
                            chan_est = np.sum(chan_est_0, 1)/self.MM[0]
                            chan_est_1[self.synch_used_bins] = chan_est
                            self.est_chan_freq_p[m, self.correlation_obs, 0:len(chan_est_1)] = chan_est_1
                            self.est_chan_freq_n[m, self.correlation_obs, 0:len(chan_est_1)] = chan_est

                            chan_est_time = np.fft.ifft(chan_est_1, self.NFFT)
                            self.est_chan_impulse[m, self.correlation_obs, 0:len(chan_est_time)] = chan_est_time
                            chan_est_ext = np.transpose(np.tile(chan_est, (1, self.MM[0])))
                            self.est_synch_freq[m, self.correlation_obs, 0: len(self.synch_used_bins) * self.MM[0]] = data_recovered * np.conj(chan_est_ext) / ((np.conj(chan_est_ext) * chan_est_ext) + (1/self.SNR))

import numpy as np
import matplotlib.pyplot as plt
#Mixups between RX buff time and RX buff time window
class RxBasebandSystem:
    def __init__(self, multi_ant_sys, Caz, param_est, case):
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
        self.norm_q = np.conj(Caz.ZChu0) * Caz.ZChu0
        self.ptr_adj = 0
        self.p = 0
        self.sym_count = 0
        self.tap_delay = 5
        self.MM = [1, 62]
        self.synch_data_shape = multi_ant_sys.symbol_pattern
        x = np.zeros([1, self.tap_delay], dtype=float)
        self.rx_buffer_time_window = np.zeros([1, self.NFFT], dtype=complex)

        self.synch_reference = Caz.ZChu0

        self.genie_chan_time = multi_ant_sys.genie_chan_time
        self.genie_chan_freq = multi_ant_sys.channel_freq

    def param_est_synch(self, sys_model):
        for m in range(1):
            while self.p <= self.loop_z:
                self.p += 1
                if self.correlation_obs == 0:
                    ptr_frame = (self.p - 1) * self.stride_value + self.start_sample + self.ptr_adj
                elif self.correlation_obs < 5:
                    ptr_frame = ptr_frame + np.sum(self.synch_data_shape) * (self.NCP + self.NFFT)
                else:
                    ptr_frame = np.round(XP[-1, :] * b - self.NCP/4)

                if (self.MM[0]-1) * self.rx_buffer_length + self.NFFT + ptr_frame - 1 < np.size(self.rx_buffer_time):
                    for LL in range(self.MM[0]):
                        self.rx_buffer_time_window[(LL)*self.NFFT+1: (LL+1)*self.NFFT] = self.rx_buffer_time[m, ((LL) * self.rx_buffer_length + ptr_frame): (LL) * self.rx_buffer_length + ptr_frame + self.NFFT]
                    # If Data Use Freq Data
                    tmp_1_vector = np.zeros([self.MM[0], self.NFFT], dtype=float)
                    for LL in range(self.MM[0]):
                        tmp_1_vector[LL, 0:self.NFFT] = np.fft.fft(self.rx_buffer_time[(LL-1)*self.NFFT+1: LL*self.NFFT], self.NFFT)

                    # If Synch Use Synch Data
                    synch_data_00 = tmp_1_vector[:, self.synch_used_bins]
                    synch_data_0 = np.transpose(synch_data_00).reshape((1, synch_data_00.size))
                    power_est = np.sum(np.dot(synch_data_0, np.conj(synch_data_0)))/len(synch_data_0)
                    synch_data = synch_data_0/np.sqrt(power_est)

                    # Genie: Actual Frequency Channel
                    chan_freq_0 = np.reshape(self.genie_chan_freq[self.synch_used_bins], (1, self.synch_used_bins.size))
                    chan_freq = np.tile(chan_freq_0, (1, self.MM[0]))
                    est_synch_data = np.dot(np.reshape(self.synch_reference, (1, self.synch_reference.size)), np.reshape(chan_freq, (1, chan_freq.size)))

                    #Delay Estimation
                    p_matrix_0 = np.exp(1j*2*(np.pi/self.NFFT) * np.dot(np.transpose(self.synch_used_bins - 1), range(0, self.NCP)))
                    p_matrix = np.tile(p_matrix_0, (self.MM[0], 1))
                    delay_matrix = np.conj(self.synch_reference) * np.diag(self.synch_data_shape) * p_matrix
                    dmax, dmaxind0 = np.max(np.abs(delay_matrix))
                    # Difference in MATLAB and Python Code
                    # dmaxind = dmaxind0 - 1

                    dlong[p] = dmax  # TODO Fix!
                    if dmax > 0.5 * len(self.synch_data_shape) or self.correlation_obs > 0:
                        if dmaxind > np.round(0.75*self.NCP):
                            if self.correlation_obs == 0:
                                ptr_adj = ptr_adj + np.round(self.NCP * 0.5)
                                ptr_frame = (p-1) * self.stride_value + self.start_sample + ptr_adj
                            elif self.correlation_obs < 5:
                                ptr_frame  = ptr_frame + np.round(self.NCP * 0.5)
                            for LL in range(self.MM[0]):
                                self.rx_buffer_time_window[LL * self.NFFT + 1: (LL+1)*self.NFFT] = self.rx_buffer_time[m, LL*self.rx_buffer_length + ptr_frame: (LL+1) self.rx_buffer_length + ptr_frame + self.NFFT - 1]
                            tmp_1_vector = np.zeros([self.MM[0], self.NFFT], dtype=float)
                            for LL in range(self.MM[0]):
                                tmp_1_vector[LL, 0:self.NFFT] = np.fft.fft(self.rx_buffer_time_window[LL * self.NFFT + 1: (LL + 1) * self.NFFT], self.NFFT)
                            synch_data_00 = tmp_1_vector[:, self.synch_used_bins]
                            synch_data_0 = np.transpose(synch_data_00).reshape((1, synch_data_00.size))
                            power_est = np.sum(np.dot(synch_data_0, np.conj(synch_data_0))) / len(synch_data_0)
                            synch_data = synch_data_0 / np.sqrt(power_est)
                            # Genie: Actual Frequency Channel
                            chan_freq_0 = np.reshape(self.genie_chan_freq[self.synch_used_bins],
                                                     (1, self.synch_used_bins.size))
                            chan_freq = np.tile(chan_freq_0, (1, self.MM[0]))
                            est_synch_data = np.dot(
                                np.reshape(self.synch_reference, (1, self.synch_reference.size)),
                                np.reshape(chan_freq, (1, chan_freq.size)))
                            p_matrix_0 = np.exp(
                                1j * 2 * (np.pi / self.NFFT) * np.dot(np.transpose(self.synch_used_bins - 1),
                                                                      range(0, self.NCP)))
                            p_matrix = np.tile(p_matrix_0, (self.MM[0], 1))
                            delay_matrix = np.conj(self.synch_reference) * np.diag(self.synch_data_shape) * p_matrix
                            dmax, dmaxind0 = np.max(np.abs(delay_matrix))
                            # Difference in MATLAB and Python Code
                            # dmaxind = dmaxind0 - 1
                            dlong[p] = dmax  # TODO Fix!
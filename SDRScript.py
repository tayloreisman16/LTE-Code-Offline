import pickle

import matplotlib.pyplot as plt
import numpy as np
from FrequencyOffsetTuner import FrequencyOffsetTuner
from MultiAntennaSystem import MultiAntennaSystem
from OFDM import OFDM
from RxBasebandSystemNEW import RxBasebandSystem
from SynchSignal import SynchSignal
# import plotly.plotly as py
from SystemModel import SystemModel

# from scipy import signal
# from RxBitRecovery import RxBitRecovery


SNR = 50  # dB

num_cases = 1

offline_directory = '/home/tayloreisman/Desktop/signal_952019_8bins.mat'
matlab_object_name = 'signal'

SDR_profiles = {0: {'system_scenario': '4G5GSISO-TU',
                    'diagnostic': 1,
                    'wireless_channel': 'AWGN',
                    'channel_band': 0.97 * 960e3,
                    'bin_spacing': 15e3,
                    'channel_profile': 'LTE-TU',
                    'CP_type': 'Normal',
                    'num_ant_txrx': 1,
                    'param_est': 'Estimated',
                    'MIMO_method': 'SpMult',
                    'SNR': SNR,
                    'ebno_db': [24],
                    'num_symbols': [1],
                    'stream_size': 1},
                1: {'system_scenario': 'WIFIMIMOSM-A',
                    'diagnostic': 0,
                    'wireless_channel': 'Fading',
                    'channel_band': 0.9 * 20e6,
                    'bin_spacing': 312.5e3,
                    'channel_profile': 'Indoor A',
                    'CP_type': 'Extended',
                    'num_ant_txrx': 2,
                    'param_est': 'Ideal',
                    'MIMO_method': 'SpMult',
                    'SNR': SNR,
                    'ebno_db': [6, 7, 8, 9, 10, 14, 16, 20, 24],
                    'num_symbols': [10, 10, 10, 10, 10, 10, 10, 10, 10],
                    'stream_size': 2}}
min = -10e3
max = 10e3
step = 100
frequencies = np.arange(min, max, step)

for case in range(num_cases):
    sys_model = SystemModel(SDR_profiles[case])

    if SDR_profiles[case]['diagnostic'] == 0:
        loop_runs = len(SDR_profiles[case]['ebno_db'])
    else:
        loop_runs = 1

    bit_errors = np.zeros((loop_runs, SDR_profiles[case]['stream_size']), dtype=float)
    bit_error_rate = np.zeros((loop_runs, SDR_profiles[case]['stream_size']), dtype=float)

    for loop_iter in range(loop_runs):
        sig_datatype = sys_model.sig_datatype
        chan_type = sys_model.wireless_channel
        phy_chan = sys_model.phy_chan
        NFFT = sys_model.NFFT

        num_bins0 = sys_model.num_bins0  # Max umber of occupied bins for data
        num_bins1 = 4 * np.floor(num_bins0 / 4)  # Make number of bins a multiple of 4 for MIMO

        # positive and negative bin indices
        # all_bins = np.array(list(range(-int(num_bins1 / 2), 0)) + list(range(1, int(num_bins1 / 2) + 1)))
        # all_bins = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
        all_bins = np.array([1])
        # print("Bins Used Index: ", all_bins)
        # positive and negative bin indices
        ref_bins0 = np.random.randint(1, int(num_bins1 / 2) + 1, size=int(np.floor(num_bins1 * sys_model.ref_sigs / 2)))
        ref_bins = np.unique(ref_bins0)
        # positive and negative bin indices
        ref_only_bins = np.sort(np.concatenate((-ref_bins, ref_bins)))  # Bins occupied by pilot (reference) signals
        # print(ref_only_bins)
        # positive and negative bin indices - converted to & replaced by positive only in MultiAntennaSystem class
        data_only_bins = np.setdiff1d(all_bins, ref_only_bins)  # Actual bins occupied by data
        print("Actual Bins Occupied by Data:", data_only_bins)

        num_used_bins = len(data_only_bins)
        print('Number of Used Bins: ', num_used_bins)
        modulation_type = sys_model.modulation_type
        bits_per_bin = sys_model.bits_per_bin

        if SDR_profiles[case]['diagnostic'] == 0:
            SNR_dB = sys_model.ebno_db[loop_iter]
        else:
            SNR_dB = SNR

        synch_data = sys_model.synch_data
        num_synchdata_patterns = int(np.ceil(sys_model.num_symbols[loop_iter] / sum(synch_data)))
        num_symbols = sum(synch_data) * num_synchdata_patterns

        # 0 - synch symbol, 1 - data symbol
        symbol_pattern0 = np.concatenate((np.zeros(synch_data[0]), np.ones(synch_data[1])))
        symbol_pattern = np.tile(symbol_pattern0, num_synchdata_patterns)
        print("Symbol_pattern0: ", symbol_pattern0, num_synchdata_patterns)

        # sum of symbol_pattern gives the total number of data symbols

        binary_info = np.random.randint(0, 2, (
            sys_model.stream_size, int(sum(symbol_pattern) * num_used_bins * bits_per_bin)))
        # binary_info = np.zeros((sys_model.stream_size, int(sum(symbol_pattern) * num_used_bins * bits_per_bin)))
        # binary_info[0, :] = np.array([0, 1]*int(sum(symbol_pattern) * num_used_bins))
        fs = sys_model.fs  # Sampling frequency
        Ts = 1 / fs  # Sampling period

        delta_f = sys_model.bin_spacing
        len_CP = 0
        if sys_model.CP_type == 'Normal':
            len_CP = round(NFFT / 4)  # cyclic prefix (IN Samples !!)
        elif sys_model.CP_type == 'Extended':
            len_CP = round(NFFT / 4 + NFFT / 8)  # cyclic prefix (IN Samples !!)
        else:
            print('Wrong CP Type')
            exit(0)

        num_ant_txrx = sys_model.num_ant_txrx
        param_est = sys_model.param_est
        MIMO_method = sys_model.MIMO_method
        num_synch_bins = sys_model.num_synch_bins
        # print(num_synch_bins)
        channel_profile = sys_model.channel_profile

        # OFDM class object
        OFDM_data = OFDM(len_CP, num_used_bins, modulation_type, NFFT, delta_f)

        diagnostic = sys_model.diagnostic
        wireless_channel = sys_model.wireless_channel
        stream_size = sys_model.stream_size

        # print(all_bins)
        # syschan
        # MultiAntennaSystem class object
        multi_ant_sys = MultiAntennaSystem(
            OFDM_data, num_ant_txrx, MIMO_method, all_bins, num_symbols, symbol_pattern,
            fs, channel_profile, diagnostic, wireless_channel, stream_size, data_only_bins, ref_only_bins)

        # SynchSignal class object
        Caz = SynchSignal(len_CP, num_synch_bins, num_ant_txrx, NFFT, synch_data)

        multi_ant_sys.multi_ant_binary_map(Caz, binary_info, synch_data)

        # At this point multi_ant_sys.buffer_data_tx contains all the transmit synch and QPSK data
        # placed in the corect bins
        # plt.plot(multi_ant_sys.buffer_data_tx.real, multi_ant_sys.buffer_data_tx.imag, '.')
        # plt.show()
        multi_ant_sys.multi_ant_symb_gen(num_symbols)
        # print(len(multi_ant_sys.buffer_data_tx_time))
        # print(multi_ant_sys.buffer_data_tx_time)
        # IMPORT FROM GNURADIO HERE
        # !!!!!!!!!!!!!!!!!!!!!!!!!

        # with open("shuffled_8bin_seed_4_10_30_19_2timeFs.pckl", 'wb') as handle:
        #     pickle.dump(multi_ant_sys.buffer_data_tx_time, handle, protocol=2)
        # **** multi_ant_sys.buffer_data_tx_time is the variable to pckl for GNURadio transmitter **** #

        # Receive signal after convolution with channel

        multi_ant_sys.rx_signal_gen()

        # Receive signal with noise added
        multi_ant_sys.additive_noise(sys_model.SNR_type, SNR_dB, wireless_channel, sys_model.sig_datatype)
        plt.plot(multi_ant_sys.buffer_data_rx_time[0, :].real)
        plt.plot(multi_ant_sys.buffer_data_rx_time[0, :].imag)
        plt.show()
        # data_gnu = scipy.io.loadmat(offline_directory)
        # multi_ant_sys.buffer_data_rx_time = data_gnu[matlab_object_name]
        # print(multi_ant_sys.buffer_data_rx_time)
        frequency_offset_estimator = 0
        if frequency_offset_estimator == 1:
            buffer_data_rx_time_fo = multi_ant_sys.buffer_data_rx_time
            freq_mu_buffer = np.zeros([1])
            freq_sigma_buffer = np.zeros([1])
            clear_condition = 0
            timeseries_storage = multi_ant_sys.buffer_data_rx_time

            for frequency in frequencies:
                print("Checking Frequency Offset at frequency ", frequency, " Hz!")
                multi_ant_sys.buffer_dat_rx_time = timeseries_storage
                for index in range(multi_ant_sys.buffer_data_rx_time.shape[1]):
                    multi_ant_sys.buffer_data_rx_time[0, index] = timeseries_storage[0, index] * np.exp(
                        1j * 2 * np.pi * frequency * Ts * index)

                rx_sys = RxBasebandSystem(multi_ant_sys, Caz, param_est, case)

                rx_sys.param_est_synch(sys_model)
            # print('Number of synch symbols found', rx_sys.corr_obs)
            # rx_sys.corr_frequencyobs is one less than the total number of synchs present in the buffer
                rx_sys.rx_data_demod()
                rx_newshape = rx_sys.est_data_freq.shape[0] * rx_sys.est_data_freq.shape[1] * rx_sys.est_data_freq.shape[2]
                rx_phasors = np.reshape(rx_sys.est_data_freq, (1, rx_newshape))
                fosys = FrequencyOffsetTuner()
                mu, sigma = fosys.constellation_deviation(rx_phasors)
                freq_mu_buffer = np.append(freq_mu_buffer, mu)
                freq_sigma_buffer = np.append(freq_sigma_buffer, sigma)
                if clear_condition == 0:
                    freq_mu_buffer = np.delete(freq_mu_buffer, 0)
                    freq_sigma_buffer = np.delete(freq_sigma_buffer, 0)
                    clear_condition = 1
            best_frequency_std_index = np.argmin(freq_sigma_buffer[freq_sigma_buffer != 0])
            best_frequency_avg_index = np.argmin(freq_mu_buffer[freq_sigma_buffer != 0])
            print("Optimal Frequency at, with best std, at ", frequencies[best_frequency_std_index], " Hz!")
            print("Best STD: ", freq_sigma_buffer[best_frequency_std_index])

            print("Optimal Frequency at, with best avg, at ", frequencies[best_frequency_avg_index], " Hz!")
            print("Best AVG: ", freq_mu_buffer[best_frequency_avg_index])

            for index in range(multi_ant_sys.buffer_data_rx_time.shape[1]):
                multi_ant_sys.buffer_data_rx_time[0, index] = timeseries_storage[0, index] * np.exp(
                    1j * 2 * np.pi * frequencies[best_frequency_std_index] * Ts * index)
                # print("Manual Check of Frequency Shift: ",
                # np.exp(1j * 2 * np.pi * frequencies[best_frequency_index] * Ts * index))
        # f, t, Sxx = signal.spectrogram(multi_ant_sys.buffer_data_rx_time[0, :], fs, return_onesided=False)
        # print(t, f, Sxx)
        # plt.pcolormesh(t, f, Sxx)
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.show()
        # f, t, Sxx = signal.spectrogram(multi_ant_sys.buffer_data_rx_time[0, :].imag, fs)
        # print(t, f, Sxx)
        # plt.pcolormesh(t, f, Sxx)
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.show()
        # for index in range(multi_ant_sys.buffer_data_rx_time.shape[1]):
        #     multi_ant_sys.buffer_data_rx_time[0, index] = multi_ant_sys.buffer_data_rx_time[0, index] * np.exp(
        #         1j * 2 * np.pi * 0.0 * Ts * index)
        # N = 2 ** np.ceil(np.log2(multi_ant_sys.buffer_data_rx_time.shape[1]))
        # yaxf = 20*np.log10(np.abs(np.fft.fft(multi_ant_sys.buffer_data_rx_time[0])))
        # t = np.arange(0, 4095)
        # n = int(multi_ant_sys.buffer_data_rx_time.shape[1])
        # k = np.arange(n)
        # T = n/fs
        # frq = k/T
        # frq = frq[range(int(n/2))]
        # Y = np.fft.fft(multi_ant_sys.buffer_data_rx_time[0, :])/n
        # Y = Y[range(int(n/2))]
        # fig, ax = plt.subplots(2, 1)
        # ax[0].plot(t, multi_ant_sys.buffer_data_rx_time[0, :])
        # ax[0].set_xlabel('Time')
        # ax[0].set_ylabel('Amplitude')
        # ax[1].plot(frq, abs(Y), 'r')  # plotting the spectrum
        # ax[1].set_xlabel('Freq (Hz)')
        # ax[1].set_ylabel('|Y(freq)|')
        # plt.show()

        rx_sys = RxBasebandSystem(multi_ant_sys, Caz)

        rx_sys.param_est_synch()

        rx_sys.rx_data_demod()
        rx_newshape = rx_sys.est_data_freq.shape[0] * rx_sys.est_data_freq.shape[1] * rx_sys.est_data_freq.shape[2]
        rx_phasors = np.reshape(rx_sys.est_data_freq, (1, rx_newshape))

        if sys_model.diagnostic == 1:
            # IQ plot
            rx_newshape = rx_sys.est_data_freq.shape[0] * rx_sys.est_data_freq.shape[1] * rx_sys.est_data_freq.shape[2]
            rx_phasors = np.reshape(rx_sys.est_data_freq, (1, rx_newshape))
            print(rx_phasors[0, :])
            rx_phasors_reduced = rx_phasors[0, :]
            rx_phasors_reduced = rx_phasors_reduced[np.abs(rx_phasors_reduced) <= 5]
            plt.plot(rx_phasors_reduced[:].real, rx_phasors_reduced[:].imag, '.')
            plt.show()

    #     bit_rec = RxBitRecovery(
    #     rx_sys.est_data_freq, rx_sys.used_bins_data, rx_sys.corr_obs, rx_sys.symbol_pattern, binary_info,
    #                             num_symbols)
    #
    #     bit_rec.soft_bit_recovery()
    #
    #     bit_errors[loop_iter, 0:stream_size] = bit_rec.raw_BER
    #     bit_error_rate[loop_iter, 0:stream_size] = bit_rec.rawerror_rate
    #
    # if SDR_profiles[case]['diagnostic'] == 0:
    #     for stream in range(SDR_profiles[case]['stream_size']):
    #         plt.semilogy(sys_model.ebno_db, bit_error_rate[:, stream])
    #         plt.xlabel('Eb/No (dB)')
    #         plt.ylabel('BER')
    #         plt.grid()
    #         plt.show()

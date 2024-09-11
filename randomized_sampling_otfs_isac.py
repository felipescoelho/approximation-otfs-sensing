"""randomized_sampling_otfs_isac.py

luizfelipe.coelho@smt.ufrj.br
may 20, 2024
"""


import argparse
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from matplotlib import cm
from scipy.constants import speed_of_light as c
from tqdm import tqdm
from src.channel import gen_targets, tx2rx, awgn
from src.otfs import idft_mat, dft_mat
from src.utils import (mf_mat, min_max_norm, zadoff_chu, periodogram,
                       add_offset, gen_approx_mat, print_info, permute_dft,
                       find_targets, classify_targets, estimate_pd_pf,
                       interp1d_curve)


plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'mathptmx',
    'font.size': 8
})


def arg_parser():
    parser = argparse.ArgumentParser()
    help_M = 'Number of elements in fast-time. It defines the number of bins' \
        + ' in the range axis of the radar system and the number of bins in ' \
        + 'the delay axis of the OTFS (number of sub-carriers)'
    parser.add_argument('--M', type=int, default=256, help=help_M)
    help_N = 'Number of elements in slow-time. It defines the number of bins' \
        + ' in the Doppler axis of the radar system and the OTFS (number of ' \
        + 'time slots)'
    parser.add_argument('--N', type=int, default=64, help=help_N)
    parser.add_argument('--P', type=int, default=1, help='Number of targets')
    parser.add_argument('--L', type=int, default=16,
                        help='Number of selected data')
    parser.add_argument('--fc', type=float, default=24,
                        help='Carrier frequency in GHz.')
    parser.add_argument('--Df', type=float, default=3,
                        help='Subcarrier spacing in kHz.')
    parser.add_argument('--snr_lim', type=str, default='-15,21',
                        help='SNR limits for simulation.')
    parser.add_argument('--seq_len', type=int, default=128,
                        help='Sequence length')
    parser.add_argument('--ensemble', type=int, default=1000)
    parser.add_argument('-m', '--mode', type=str, default='sim',
                        choices=['sim', 'psd', 'test'])
    parser.add_argument('--default_seed', action=argparse.BooleanOptionalAction,
                        help='Sets a default seed value (42).')
    parser.add_argument('--figures', action=argparse.BooleanOptionalAction,
                        help='Will skip simulation and generate figures.')
    args = parser.parse_args()

    return args


def worker(setup: tuple):
    """A funtion to handle parallel processing."""

    idx, L, snr, threshold, ensemble, M, N, P, Df, fc, seed = setup

    rng = np.random.default_rng(seed=seed)

    seq_len0 = M
    seq_len1 = int(M/2)
    # Transmitter:
    x_zc0 = zadoff_chu(seq_len0)
    x_zc1 = zadoff_chu(seq_len1)
    X_zc0 = np.zeros((M, N), dtype=np.complex128)
    X_zc1 = np.zeros((M, N), dtype=np.complex128)
    X_zc0[:seq_len0, 0] = x_zc0
    X_zc1[:seq_len1, 0] = x_zc1
    X_dt0 = X_zc0 @ idft_mat(N)
    X_dt1 = X_zc1 @ idft_mat(N)
    x_t0 = X_dt0.T.flatten()
    x_t1 = X_dt1.T.flatten()
    X_corr0 = mf_mat(X_zc0[:, 0])
    X_corr1 = mf_mat(X_zc1[:, 0])
    # Memory allocation
    pd0 = np.zeros((ensemble, 6))
    pf0 = np.zeros((ensemble, 6))
    pd1 = np.zeros((ensemble, 6))
    pf1 = np.zeros((ensemble, 6))
    for it in range(ensemble):
        # Channel:
        spamseed = rng.integers(99999999)
        targets, targets_pos = gen_targets(seed=spamseed, M=M, N=N, P=P,
                                           Df=Df*1e-3, fc=fc*1e-9)
        r_t0 = tx2rx(x_t0, targets, M, N, T, fc, snr, spamseed)
        r_t1 = tx2rx(x_t1, targets, M, N, T, fc, snr, spamseed)
        # Receiver:
        Y_dt0 = r_t0.reshape((N, M)).T
        Y_dt1 = r_t1.reshape((N, M)).T
        Y_dd0 = Y_dt0 @ dft_mat(N)
        Y_dd1 = Y_dt1 @ dft_mat(N)
        # Data selection:
        Q0_rnd = gen_approx_mat(Y_dd0, L, 'rnd', rng.integers(99999))
        Q0_rnd_l1 = gen_approx_mat(Y_dd0, L, 'rnd-l1', rng.integers(99999))
        Q0_rnd_l2 = gen_approx_mat(Y_dd0, L, 'rnd-l2', rng.integers(99999))
        Q0_det_l1 = gen_approx_mat(Y_dd0, L, 'det-l1')
        Q0_det_l2 = gen_approx_mat(Y_dd0, L, 'det-l2')
        Q1_rnd = gen_approx_mat(Y_dd1, L, 'rnd', rng.integers(99999))
        Q1_rnd_l1 = gen_approx_mat(Y_dd1, L, 'rnd-l1', rng.integers(99999))
        Q1_rnd_l2 = gen_approx_mat(Y_dd1, L, 'rnd-l2', rng.integers(99999))
        Q1_det_l1 = gen_approx_mat(Y_dd1, L, 'det-l1')
        Q1_det_l2 = gen_approx_mat(Y_dd1, L, 'det-l2')
        # Correlation and normalization:
        Y0 = np.abs(X_corr0 @ Y_dd0) @ permute_dft(N) / seq_len0
        Y0_rnd = np.abs(X_corr0 @ Q0_rnd.T @ Q0_rnd @ Y_dd0) \
            @ permute_dft(N) / seq_len0
        Y0_rnd_l1 = np.abs(X_corr0 @ Q0_rnd_l1.T @ Q0_rnd_l1 @ Y_dd0) \
            @ permute_dft(N) / seq_len0
        Y0_rnd_l2 = np.abs(X_corr0 @ Q0_rnd_l2.T @ Q0_rnd_l2 @ Y_dd0) \
            @ permute_dft(N) / seq_len0
        Y0_det_l1 = np.abs(X_corr0 @ Q0_det_l1.T @ Q0_det_l1 @ Y_dd0) \
            @ permute_dft(N) / seq_len0
        Y0_det_l2 = np.abs(X_corr0 @ Q0_det_l2.T @ Q0_det_l2 @ Y_dd0) \
            @ permute_dft(N) / seq_len0
        Y1 = np.abs(X_corr1 @ Y_dd1) @ permute_dft(N) / seq_len1
        Y1_rnd = np.abs(X_corr1 @ Q1_rnd.T @ Q1_rnd @ Y_dd1) \
            @ permute_dft(N) / seq_len1
        Y1_rnd_l1 = np.abs(X_corr1 @ Q1_rnd_l1.T @ Q1_rnd_l1 @ Y_dd1) \
            @ permute_dft(N) / seq_len1
        Y1_rnd_l2 = np.abs(X_corr1 @ Q1_rnd_l2.T @ Q1_rnd_l2 @ Y_dd1) \
            @ permute_dft(N) / seq_len1
        Y1_det_l1 = np.abs(X_corr1 @ Q1_det_l1.T @ Q1_det_l1 @ Y_dd1) \
            @ permute_dft(N) / seq_len1
        Y1_det_l2 = np.abs(X_corr1 @ Q1_det_l2.T @ Q1_det_l2 @ Y_dd1) \
            @ permute_dft(N) / seq_len1
        # Classify Targets
        res0 = classify_targets(find_targets(Y0, threshold), targets_pos, M, N,
                                P)
        res0_rnd = classify_targets(find_targets(Y0_rnd, threshold),
                                    targets_pos, M, N, P)
        res0_rnd_l1 = classify_targets(find_targets(Y0_rnd_l1, threshold),
                                       targets_pos, M, N, P)
        res0_rnd_l2 = classify_targets(find_targets(Y0_rnd_l2, threshold),
                                       targets_pos, M, N, P)
        res0_det_l1 = classify_targets(find_targets(Y0_det_l1, threshold),
                                       targets_pos, M, N, P)
        res0_det_l2 = classify_targets(find_targets(Y0_det_l2, threshold),
                                       targets_pos, M, N, P)
        res1 = classify_targets(find_targets(Y1, threshold), targets_pos, M, N,
                                P)
        res1_rnd = classify_targets(find_targets(Y1_rnd, threshold),
                                    targets_pos, M, N, P)
        res1_rnd_l1 = classify_targets(find_targets(Y1_rnd_l1, threshold),
                                       targets_pos, M, N, P)
        res1_rnd_l2 = classify_targets(find_targets(Y1_rnd_l2, threshold),
                                       targets_pos, M, N, P)
        res1_det_l1 = classify_targets(find_targets(Y1_det_l1, threshold),
                                       targets_pos, M, N, P)
        res1_det_l2 = classify_targets(find_targets(Y1_det_l2, threshold),
                                       targets_pos, M, N, P)
        # Estimate Probabilities
        pd0[it, 0], pf0[it, 0] = estimate_pd_pf(res0)
        pd0[it, 1], pf0[it, 1] = estimate_pd_pf(res0_rnd)
        pd0[it, 2], pf0[it, 2] = estimate_pd_pf(res0_rnd_l1)
        pd0[it, 3], pf0[it, 3] = estimate_pd_pf(res0_rnd_l2)
        pd0[it, 4], pf0[it, 4] = estimate_pd_pf(res0_det_l1)
        pd0[it, 5], pf0[it, 5] = estimate_pd_pf(res0_det_l2)
        pd1[it, 0], pf1[it, 0] = estimate_pd_pf(res1)
        pd1[it, 1], pf1[it, 1] = estimate_pd_pf(res1_rnd)
        pd1[it, 2], pf1[it, 2] = estimate_pd_pf(res1_rnd_l1)
        pd1[it, 3], pf1[it, 3] = estimate_pd_pf(res1_rnd_l2)
        pd1[it, 4], pf1[it, 4] = estimate_pd_pf(res1_det_l1)
        pd1[it, 5], pf1[it, 5] = estimate_pd_pf(res1_det_l2)
    # Average Resutls
    pd0_avg = np.mean(pd0, axis=0)
    pf0_avg = np.mean(pf0, axis=0)
    pd1_avg = np.mean(pd1, axis=0)
    pf1_avg = np.mean(pf1, axis=0)

    return (idx, pd0_avg, pf0_avg, pd1_avg, pf1_avg)


if __name__ == '__main__':
    
    args = arg_parser()

    Df = args.Df * 1e3  # Subcarrier spacing
    fc = args.fc * 1e9  # Carrier frequency
    M = args.M  # delay axis
    N = args.N  # Doppler axis
    P = args.P  # Number of targets
    L = args.L  # Number of selected samples
    T = 1/Df  # Block duration
    Ts = T/M  # Sampling period
    if args.default_seed:
        rng = np.random.default_rng(seed=42)
    else:
        rng = np.random.default_rng()

    match args.mode:
        case 'test':

            L = 16
            X0 = np.zeros((M, N), dtype=np.complex128)
            X0[:, 2] = zadoff_chu(M)
            X0[:, 25] = np.roll(zadoff_chu(M), 50)
            X0_corr = mf_mat(X0[:, 2])
            Y0 = np.abs(X0_corr @ X0)
            Q0_rand = gen_approx_mat(X0, L, 'rnd')
            Y0_rand = np.abs(X0_corr @ Q0_rand.T @ Q0_rand @ X0)
            Q0_rand_l1 = gen_approx_mat(X0, L, 'rnd-l1')
            Y0_rand_l1 = np.abs(X0_corr @ Q0_rand_l1.T @ Q0_rand_l1 @ X0)
            Q0_rand_l2 = gen_approx_mat(X0, L, 'rnd-l2')
            Y0_rand_l2 = np.abs(X0_corr @ Q0_rand_l2.T @ Q0_rand_l2 @ X0)
            Q0_det_l1 = gen_approx_mat(X0, L, 'det-l1')
            Y0_det_l1 = np.abs(X0_corr @ Q0_det_l1.T @ Q0_det_l1 @ X0)
            Q0_det_l2 = gen_approx_mat(X0, L, 'det-l2')
            Y0_det_l2 = np.abs(X0_corr @ Q0_det_l2.T @ Q0_det_l2 @ X0)

            print('Full dynamic')
            print(f'full: {np.max(Y0)}')
            print(f'rand: {np.max(Y0_rand)}')
            print(f'rand-l1: {np.max(Y0_rand_l1)}')
            print(f'rand-l2: {np.max(Y0_rand_l2)}')
            print(f'det-l1: {np.max(Y0_det_l1)}')
            print(f'det-l2: {np.max(Y0_det_l2)}')

            seq_len = 32
            X1 = np.zeros((M, N), dtype=np.complex128)
            X1[:seq_len, 2] = zadoff_chu(seq_len)
            X1[50:50+seq_len, 25] = zadoff_chu(seq_len)
            X1_corr = mf_mat(X1[:, 2])
            Y1 = np.abs(X1_corr @ X1)
            Q1_rand = gen_approx_mat(X1, L, 'rnd')
            Y1_rand = np.abs(X1_corr @ Q1_rand.T @ Q1_rand @ X1)
            Q1_rand_l1 = gen_approx_mat(X1, L, 'rnd-l1')
            Y1_rand_l1 = np.abs(X1_corr @ Q1_rand_l1.T @ Q1_rand_l1 @ X1)
            Q1_rand_l2 = gen_approx_mat(X1, L, 'rnd-l2')
            Y1_rand_l2 = np.abs(X1_corr @ Q1_rand_l2.T @ Q1_rand_l2 @ X1)
            Q1_det_l1 = gen_approx_mat(X1, L, 'det-l1')
            Y1_det_l1 = np.abs(X1_corr @ Q1_det_l1.T @ Q1_det_l1 @ X1)
            Q1_det_l2 = gen_approx_mat(X1, L, 'det-l2')
            Y1_det_l2 = np.abs(X1_corr @ Q1_det_l2.T @ Q1_det_l2 @ X1)

            print(f'{seq_len}/{M} dynamic : {M/seq_len}')
            print(f'full: {np.max(Y1)}')
            print(f'rand: {np.max(Y1_rand)}')
            print(f'rand-l1: {np.max(Y1_rand_l1)}')
            print(f'rand-l2: {np.max(Y1_rand_l2)}')
            print(f'det-l1: {np.max(Y1_det_l1)}')
            print(f'det-l2: {np.max(Y1_det_l2)}')

        case 'sim':
            
            # Make folders:
            fig_folder = 'figures/otfs-rand'
            os.makedirs(fig_folder, exist_ok=True)
            res_folder = 'results/otds-rand'
            os.makedirs(res_folder, exist_ok=True)
            res_path = os.path.join(res_folder, f'results_{P}_targets.npz')
            
            # Generate the axes
            min_Doppler = -Df/2  # Hz
            max_Doppler = Df/2 - Df/N
            min_velocity = c*min_Doppler/fc  # m/s
            max_velocity = c*max_Doppler/fc
            max_delay = T  # s
            max_range = c*max_delay/2  # m
            mesh_Doppler, mesh_delay = np.meshgrid(
                3.6*np.linspace(min_velocity, max_velocity, N),
                np.linspace(0, max_range, M)
            )

            seq_len0 = M
            seq_len1 = int(M/2)
            sausage = [int(val) for val in args.snr_lim.split(',')]
            snrs = np.arange(*sausage, 5)
            thresholds = np.linspace(0, 5, 20)
            L_axis = np.arange(1, M, 15)
            pd0 = np.zeros((6, len(thresholds), len(snrs), len(L_axis)))
            pf0 = np.zeros((6, len(thresholds), len(snrs), len(L_axis)))
            pd1 = np.zeros((6, len(thresholds), len(snrs), len(L_axis)))
            pf1 = np.zeros((6, len(thresholds), len(snrs), len(L_axis)))
            # Transmitter:
            x_zc0 = zadoff_chu(seq_len0)
            x_zc1 = zadoff_chu(seq_len1)
            X_zc0 = np.zeros((M, N), dtype=np.complex128)
            X_zc1 = np.zeros((M, N), dtype=np.complex128)
            X_zc0[:seq_len0, 0] = x_zc0
            X_zc1[:seq_len1, 0] = x_zc1
            X_dt0 = X_zc0 @ idft_mat(N)
            X_dt1 = X_zc1 @ idft_mat(N)
            x_t0 = X_dt0.T.flatten()
            x_t1 = X_dt1.T.flatten()
            if not args.figures:
                idx_list = list(itertools.product(range(len(L_axis)),
                                                  range(len(snrs)),
                                                  range(len(thresholds))))
                setup_list = [(idx, L_axis[idx[0]], snrs[idx[1]],
                               thresholds[idx[2]], args.ensemble, M, N, P, Df,
                               fc, rng.integers(99999999)) for idx in idx_list]
                with Pool(cpu_count()) as p:
                    for data in tqdm(p.imap_unordered(worker, setup_list), total=len(setup_list)):
                        idx, pd0_avg, pf0_avg, pd1_avg, pf1_avg = data
                        pd0[:, idx[2], idx[1], idx[0]] = pd0_avg
                        pf0[:, idx[2], idx[1], idx[0]] = pf0_avg
                        pd1[:, idx[2], idx[1], idx[0]] = pd1_avg
                        pf1[:, idx[2], idx[1], idx[0]] = pf1_avg

                np.savez(res_path, pd0, pf0, pd1, pf1)

            else:

                res_file = np.load(res_path)
                pd0 = res_file['arr_0']
                pf0 = res_file['arr_1']
                pd1 = res_file['arr_2']
                pf1 = res_file['arr_3']

                # Sort for increasing false alarm rate, drop duplicates, and
                # interpolate data.
                pf_axis = np.linspace(0, 1, 1000)
                pd0_interp = np.zeros((6, len(pf_axis), len(snrs), len(L_axis)))
                pd1_interp = np.zeros((6, len(pf_axis), len(snrs), len(L_axis)))
                for idx0 in range(len(L_axis)):
                    for idx1 in range(len(snrs)):
                        pd0_interp[0, :, idx1, idx0] = interp1d_curve(
                            pf0[0, :, idx1, idx0], pd0[0, :, idx1, idx0],
                            pf_axis
                        )
                        pd0_interp[1, :, idx1, idx0] = interp1d_curve(
                            pf0[1, :, idx1, idx0], pd0[1, :, idx1, idx0],
                            pf_axis
                        )
                        pd0_interp[2, :, idx1, idx0] = interp1d_curve(
                            pf0[2, :, idx1, idx0], pd0[2, :, idx1, idx0],
                            pf_axis
                        )
                        pd0_interp[3, :, idx1, idx0] = interp1d_curve(
                            pf0[3, :, idx1, idx0], pd0[3, :, idx1, idx0],
                            pf_axis
                        )
                        pd0_interp[4, :, idx1, idx0] = interp1d_curve(
                            pf0[4, :, idx1, idx0], pd0[4, :, idx1, idx0],
                            pf_axis
                        )
                        pd0_interp[5, :, idx1, idx0] = interp1d_curve(
                            pf0[5, :, idx1, idx0], pd0[5, :, idx1, idx0],
                            pf_axis
                        )
                        # Not full dynamic
                        pd1_interp[0, :, idx1, idx0] = interp1d_curve(
                            pf1[0, :, idx1, idx0], pd1[0, :, idx1, idx0],
                            pf_axis
                        )
                        pd1_interp[1, :, idx1, idx0] = interp1d_curve(
                            pf1[1, :, idx1, idx0], pd1[1, :, idx1, idx0],
                            pf_axis
                        )
                        pd1_interp[2, :, idx1, idx0] = interp1d_curve(
                            pf1[2, :, idx1, idx0], pd1[2, :, idx1, idx0],
                            pf_axis
                        )
                        pd1_interp[3, :, idx1, idx0] = interp1d_curve(
                            pf1[3, :, idx1, idx0], pd1[3, :, idx1, idx0],
                            pf_axis
                        )
                        pd1_interp[4, :, idx1, idx0] = interp1d_curve(
                            pf1[4, :, idx1, idx0], pd1[4, :, idx1, idx0],
                            pf_axis
                        )
                        pd1_interp[5, :, idx1, idx0] = interp1d_curve(
                            pf1[5, :, idx1, idx0], pd1[5, :, idx1, idx0],
                            pf_axis
                        )

                idx_pf = 5
                idx_L = 1
                idx_snr = 3
                golden_ratio = (1 + 5**.5)/2
                width = 3.5
                height = width/golden_ratio
                print(f'Pf = {pf_axis[idx_pf]}')
                print(f'L = {L_axis[idx_L]}')
                print(f'SNR = {snrs[idx_snr]}')
                pd0_full_avg = np.mean(pd0_interp, axis=3)
                pd1_full_avg = np.mean(pd1_interp, axis=3)
                fig1 = plt.figure(figsize=(width, height))
                ax = fig1.add_subplot(111)
                ax.plot(snrs, pd0_full_avg[0, idx_pf, :], label='Complete')
                ax.plot(snrs, pd0_interp[1, idx_pf, :, idx_L], label='rand')
                ax.plot(snrs, pd0_interp[2, idx_pf, :, idx_L], label='rand-$\ell_1$')
                ax.plot(snrs, pd0_interp[3, idx_pf, :, idx_L], label='rand-$\ell_2$')
                ax.plot(snrs, pd0_interp[4, idx_pf, :, idx_L], label='det-$\ell_1$')
                ax.plot(snrs, pd0_interp[5, idx_pf, :, idx_L], label='det-$\ell_2$')
                ax.set_xlabel('SNR, dB')
                ax.set_ylabel('$P_d$')
                ax.legend(ncol=2)
                fig1.tight_layout()

                fig2 = plt.figure(figsize=(width, height))
                ax = fig2.add_subplot(111)
                ax.plot(snrs, pd1_full_avg[0, idx_pf, :], label='Complete')
                ax.plot(snrs, pd1_interp[1, idx_pf, :, idx_L], label='rand')
                ax.plot(snrs, pd1_interp[2, idx_pf, :, idx_L], label='rand-$\ell_1$')
                ax.plot(snrs, pd1_interp[3, idx_pf, :, idx_L], label='rand-$\ell_2$')
                ax.plot(snrs, pd1_interp[4, idx_pf, :, idx_L], label='det-$\ell_1$')
                ax.plot(snrs, pd1_interp[5, idx_pf, :, idx_L], label='det-$\ell_2$')
                ax.set_xlabel('SNR, dB')
                ax.set_ylabel('$P_d$')
                ax.legend(ncol=2)
                fig2.tight_layout()

                fig3 = plt.figure(figsize=(width, height))
                ax = fig3.add_subplot(111)
                ax.plot(pf_axis, pd0_full_avg[0, :, idx_snr], label='Complete')
                ax.plot(pf_axis, pd0_interp[1, :, idx_snr, idx_L],
                        label='rand')
                ax.plot(pf_axis, pd0_interp[2, :, idx_snr, idx_L],
                        label='rand-$\ell_1$')
                ax.plot(pf_axis, pd0_interp[3, :, idx_snr, idx_L],
                        label='rand-$\ell_2$')
                ax.plot(pf_axis, pd0_interp[4, :, idx_snr, idx_L],
                        label='det-$\ell_1$')
                ax.plot(pf_axis, pd0_interp[5, :, idx_snr, idx_L],
                        label='det-$\ell_2$')
                ax.set_xlabel('$P_f$')
                ax.set_ylabel('$P_d$')
                ax.legend()
                fig3.tight_layout()

                fig4 = plt.figure(figsize=(width, height))
                ax = fig4.add_subplot(111)
                ax.plot(pf_axis, pd1_full_avg[0, :, idx_snr], label='Complete')
                ax.plot(pf_axis, pd1_interp[1, :, idx_snr, idx_L],
                        label='rand')
                ax.plot(pf_axis, pd1_interp[2, :, idx_snr, idx_L],
                        label='rand-$\ell_1$')
                ax.plot(pf_axis, pd1_interp[3, :, idx_snr, idx_L],
                        label='rand-$\ell_2$')
                ax.plot(pf_axis, pd1_interp[4, :, idx_snr, idx_L],
                        label='det-$\ell_1$')
                ax.plot(pf_axis, pd1_interp[5, :, idx_snr, idx_L],
                        label='det-$\ell_2$')
                ax.set_xlabel('$P_f$')
                ax.set_ylabel('$P_d$')
                ax.legend()
                fig4.tight_layout()

                fig5 = plt.figure(figsize=(width, height))
                ax = fig5.add_subplot(111)
                ax.hlines(pd0_full_avg[0, idx_pf, idx_snr], L_axis[0], L_axis[-1],
                          label='Complete')
                ax.plot(L_axis, pd0_interp[1, idx_pf, idx_snr, :], label='rand',
                        c='tab:orange')
                ax.plot(L_axis, pd0_interp[2, idx_pf, idx_snr, :],
                        label='rand-$\ell_1$', c='tab:green')
                ax.plot(L_axis, pd0_interp[3, idx_pf, idx_snr, :],
                        label='rand-$\ell_2$', c='tab:red')
                ax.plot(L_axis, pd0_interp[4, idx_pf, idx_snr, :],
                        label='det-$\ell_1$', c='tab:purple')
                ax.plot(L_axis, pd0_interp[5, idx_pf, idx_snr, :],
                        label='det-$\ell_2$', c='tab:brown')
                ax.set_xlabel('$L$')
                ax.set_ylabel('$P_d$')
                ax.legend()
                fig5.tight_layout()

                fig6 = plt.figure(figsize=(width, height))
                ax = fig6.add_subplot(111)
                ax.hlines(pd1_full_avg[0, idx_pf, idx_snr], L_axis[0], L_axis[-1],
                          label='Complete')
                ax.plot(L_axis, pd1_interp[1, idx_pf, idx_snr, :], label='rand',
                        c='tab:orange')
                ax.plot(L_axis, pd1_interp[2, idx_pf, idx_snr, :],
                        label='rand-$\ell_1$', c='tab:green')
                ax.plot(L_axis, pd1_interp[3, idx_pf, idx_snr, :],
                        label='rand-$\ell_2$', c='tab:red')
                ax.plot(L_axis, pd1_interp[4, idx_pf, idx_snr, :],
                        label='det-$\ell_1$', c='tab:purple')
                ax.plot(L_axis, pd1_interp[5, idx_pf, idx_snr, :],
                        label='det-$\ell_2$', c='tab:brown')
                ax.set_xlabel('$L$')
                ax.set_ylabel('$P_d$')
                ax.legend()
                fig6.tight_layout()

                fig1_path = os.path.join(
                    fig_folder, f'pd_snr_P_{P}_L_{L_axis[idx_L]}_pf_{pf_axis[idx_pf]}_full_dynamic.eps'
                )
                fig1.savefig(fig1_path, format='eps', bbox_inches='tight')
                fig2_path = os.path.join(
                    fig_folder, f'pd_snr_P_{P}_L_{L_axis[idx_L]}_pf_{pf_axis[idx_pf]}_half_dynamic.eps'
                )
                fig2.savefig(fig2_path, format='eps', bbox_inches='tight')
                fig3_path = os.path.join(
                    fig_folder, f'pf_pd_P_{P}_L_{L_axis[idx_L]}_full_dynamic.eps'
                )
                fig3.savefig(fig3_path, format='eps', bbox_inches='tight')
                fig4_path = os.path.join(
                    fig_folder, f'pf_pd_P_{P}_L_{L_axis[idx_L]}_half_dynamic.eps'
                )
                fig4.savefig(fig4_path, format='eps', bbox_inches='tight')
                fig5_path = os.path.join(
                    fig_folder, f'L_pd_P_{P}_snr_{snrs[-1]}_pf_{pf_axis[idx_pf]}_full_dynamic.eps'
                )
                fig5.savefig(fig5_path, format='eps', bbox_inches='tight')
                fig6_path = os.path.join(
                    fig_folder, f'L_pd_P_{P}_snr_{snrs[-1]}_pf_{pf_axis[idx_pf]}_half_dynamic.eps'
                )
                fig6.savefig(fig6_path, format='eps', bbox_inches='tight')


                plt.show()

            
            # fig1 = plt.figure()
            # ax0 = fig1.add_subplot(121)
            # c0 = ax0.pcolormesh(mesh_Doppler, mesh_delay, Y, cmap=cm.coolwarm)
            # ax0.set_xlabel('Target speed, km/h')
            # ax0.set_ylabel('Target range, m')
            # ax0.set_title('Full matched filter')
            # ax1 = fig1.add_subplot(122)
            # c1 = ax1.pcolormesh(mesh_Doppler, mesh_delay, Y_hat,
            #                     cmap=cm.coolwarm)
            # ax1.set_xlabel('Target speed, km/h')
            # ax1.set_ylabel('Target range, m')
            # ax1.set_title('Low-rank matched filter')
            # fig1.colorbar(c0, ax=ax0)
            # fig1.colorbar(c1, ax=ax1)
            # fig1.tight_layout()

            # fig2 = plt.figure()
            # ax = fig2.add_subplot(111)
            # ax.plot(pf, pd, label='Full MF')
            # ax.plot(pf_hat, pd_hat, label='Low-Rank MF')
            # ax.set_xlabel('Probability of false alarm, $P_f$')
            # ax.set_ylabel('Probability of detection, $P_d$')
            # ax.legend()
            # fig2.tight_layout()

            # fig1_path = os.path.join(fig_folder, 'example.eps')
            # fig2_path = os.path.join(fig_folder, 'roc.eps')
            # fig1.savefig(fig1_path, bbox_inches='tight')
            # fig2.savefig(fig2_path, bbox_inches='tight')

            
        case 'psd':
            pass

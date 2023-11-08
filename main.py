import os
import cv2
import pywt
import time
import numpy as np
import tensorly as tl
from tensorly.tenalg import multi_mode_dot
from tensorly.tt_tensor import tt_to_tensor
from tensorly.decomposition import tucker, parafac, tensor_train, tensor_ring
from utils import bit2byte
from lloyds_quantization import *
from entropy_coding import *
from BayesianOptimize import BayeOpt
from setting_ranks import tucker_ranks


def tensor_decomposition(all_data, config, decom='Tucker', bayes=False):
    np.random.seed(config.seed)
    Estimated = []
    core_list = []
    factors_list = []
    core_size = 0
    factor_size = 0
    t0 = time.clock()
    ranks = tucker_ranks(all_data[0][0])
    t1 = time.clock() - t0
    print(t1)
    print(ranks)
    bayeopt = BayeOpt(list(all_data[0][0].shape), 100)

    for i in range(len(all_data)):  # one tensor unit
        batch_data = all_data[i][0]
        batch_mask = all_data[i][1]

        # --------------------------Tucker decomposition--------------------------
        if decom == 'TK':
            if bayes:   # finding Tucker ranks using Bayesian optimization
                t0 = time.clock()
                ranks = bayeopt.estimate_ranks(all_data[i])
                t1 = time.clock() - t0
                print(t1)
                print(ranks)
            core, factors = tucker(tensor=batch_data, rank=ranks, mask=batch_mask)
            tensor = multi_mode_dot(core, factors)
            Estimated += [tensor.squeeze()]

        # ----------------------------CP decomposition----------------------------
        elif decom == 'CP':
            cp_tensor = parafac(tensor=batch_data, rank=config.features[0], mask=batch_mask)
            Estimated += [tl.cp_to_tensor(cp_tensor)]

        # ----------------------------TT decomposition----------------------------
        # Since TT and TR have restrictions on rank,
        # transposition is needed to bring the large data dimensions to the front
        elif decom == 'TT':
            config.tr_ranks[0] = config.tr_ranks[-1] = 1
            if config.dataset == 'PMU': batch_data = np.transpose(batch_data, [3, 2, 1, 0])     # PMU [3, 2, 1, 0]
            else: batch_data = np.transpose(batch_data, [2, 0, 1])    # MBD PL HA [2, 0, 1]
            factors = tensor_train(input_tensor=batch_data, rank=config.tr_ranks.tolist())
            full_tensor = tt_to_tensor(factors)
            if config.dataset == 'PMU': full_tensor = np.transpose(full_tensor, [3, 2, 1, 0])   # PMU [3, 2, 1, 0]
            else: full_tensor = np.transpose(full_tensor, [1, 2, 0])    # MBD PL HA [1, 2, 0]
            Estimated += [full_tensor]

        # ----------------------------TR decomposition----------------------------
        elif decom == 'TR':
            if config.dataset == 'PMU': batch_data = np.transpose(batch_data, [3, 2, 1, 0])
            else: batch_data = np.transpose(batch_data, [2, 0, 1])
            factors = tensor_ring(input_tensor=batch_data, rank=config.tr_ranks.tolist())
            full_tensor = factors[0]
            for factor in factors[1:-1]:
                full_tensor = tl.tensordot(full_tensor, factor, [-1, 0])
            full_tensor = tl.tensordot(full_tensor, factors[-1], ([0, -1], [-1, 0]))
            if config.dataset == 'PMU': full_tensor = np.transpose(full_tensor, [3, 2, 1, 0])
            else: full_tensor = np.transpose(full_tensor, [1, 2, 0])
            Estimated += [full_tensor]
        else:
            raise ValueError('Please select a valid decomposition method: TK, CP, TT, TR')

        # -----------------------Calculate the size of core and factor-----------------------
        if decom == 'TK':
            core_list += [core]
            np.savetxt('./results/{}/core {}.txt'.format(config.dataset, i), core.reshape(-1), fmt='%.4f')
            core_size += os.stat('./results/{}/core {}.txt'.format(config.dataset, i)).st_size
        factors_list += [factors]
        for n in range(len(factors)):
            np.savetxt('./results/{}/factor {}_{}.txt'.format(config.dataset, i, n), factors[n].reshape(-1), fmt='%.4f')
            factor_size += os.stat('./results/{}/factor {}_{}.txt'.format(config.dataset, i, n)).st_size
    Estimated = np.concatenate(Estimated)
    return Estimated, core_list, factors_list, core_size, factor_size


def quantization_coding(dataset, core_list, factors_list, num_core_bits=5, num_factor_bits=10, digit=4):
    Estimated = []
    num_core_bits = int(np.round(np.log10(np.prod(list(core_list[0].shape))))) + 2
    total_core_bits = 0
    codebook_size = 0
    for i in range(len(core_list)):
        # ---------------------Core Quantization and Coding---------------------
        huffman_code, codebook, core_t = core_compression(core_list[i], num_core_bits, digit)
        total_core_bits += len(huffman_code)
        np.savetxt('./results/{}/codebook {}.txt'.format(dataset, i), codebook, fmt='%.{}f'.format(digit))
        codebook_size += os.stat('./results/{}/codebook {}.txt'.format(dataset, i)).st_size
        huffman_str = bit2byte(huffman_code)
        with open('./results/{}/huffman {}.bin'.format(dataset, i), 'wb') as f:
            f.write(bytes(huffman_str))

        # --------------------Factor Quantization and Coding--------------------
        factors_q, factors_t = factor_compression(factors_list[i], num_factor_bits)
        for n in range(3):
            np.savetxt('./results/{}/factor_q {}_{}.txt'.format(dataset, i, n), factors_q[n], fmt='%d')

        # --------------------------Reconstruction Tensor--------------------------
        tensor = multi_mode_dot(core_t, factors_t)
        Estimated += [tensor.squeeze()]

    total_factor_bits = np.sum(np.asarray([matrix.size for matrix in factors_list[0]])) * (len(factors_list)) * (num_factor_bits+1)
    Estimated = np.concatenate(Estimated)
    return Estimated, codebook_size, total_core_bits, total_factor_bits


def core_compression(core, num_bits, digit):
    ranks = core.shape
    core = core.reshape(-1)

    # ---------------------Core Quantization---------------------
    partition, codebook, _, _ = lloyds(abs(core), 2 ** num_bits)
    codebook = np.round(codebook, digit)
    index, quant, distor = quantiz(abs(core), partition, codebook)
    index += (core < 0) * (1 << num_bits)
    quant = (np.sign(core) * quant).reshape(ranks)

    # -----------------------Core Encoding-----------------------
    rle = run_length_encoding(index)
    huffman_tree = Huffman(rle)
    huffman_code = Huffman.encode(huffman_tree, rle)

    # -----------------------Core Decoding-----------------------
    rle_decode = Huffman.decode(huffman_tree, huffman_code)
    assert rle == rle_decode
    index_decode = run_length_decoding(rle_decode)
    assert np.all(index == index_decode)

    # -----------------Core Inverse Quantization-----------------
    sign = (index_decode >> num_bits) & 1
    index_decode &= ~ (1 << num_bits)
    core_t = ((1 - sign * 2) * codebook[index_decode]).reshape(ranks)

    assert np.all(quant == core_t)
    return huffman_code, codebook, core_t


def factor_compression(factors, num_bits):
    factors_q = []   # encode
    factors_t = []   # decode
    for n in range(len(factors)):
        quant = np.round(np.abs(factors[n]) * ((1 << num_bits)-1))
        quant += (factors[n] < 0) * (1 << num_bits)
        quant = quant.astype(int)
        factors_q += [quant]

        dequant = quant.copy()
        sign = (dequant >> num_bits) & 1
        dequant &= ~ (1 << num_bits)
        dequant = (1 - sign * 2) * dequant / ((1 << num_bits)-1)

        factors_t += [dequant]
    return factors_q, factors_t


def uniform_quantization(core, num_bits):
    max_val = np.max(core)
    min_val = np.min(core)
    step = (max_val - min_val) / (num_bits - 1)
    codebook = np.arange(0, num_bits, 1) * step + min_val
    partition = (codebook[:-1] + codebook[1:])/2
    return partition, codebook


def transformation(all_data, model, theta):
    Estimated = []
    dct_size = 0
    for i in range(len(all_data)):
        batch_data = all_data[i][0]
        ndim = batch_data.shape
        batch_data = batch_data.reshape(ndim[0]*ndim[1], -1)
        if model == 'DCT':
            coefficient = cv2.dct(batch_data.astype(np.float32))
            quantized, num_coeffs = threshold_quantization(coefficient, theta)
            reconstructed = cv2.idct(quantized.astype(np.float32))
        elif model == 'DWT':
            coefficient = pywt.dwt2(batch_data, 'db4')
            LL, (LH, HL, HH) = coefficient
            LL, n1 = threshold_quantization(LL, theta)
            LH, n2 = threshold_quantization(LH, theta)
            HL, n3 = threshold_quantization(HL, theta)
            HH, n4 = threshold_quantization(HH, theta)
            num_coeffs = n1+n2+n3+n4
            coefficient = LL, (LH, HL, HH)
            reconstructed = pywt.idwt2(coefficient, 'db4')
        else:
            raise Exception("Illegal option!")
        reconstructed = reconstructed.reshape(ndim)
        Estimated += [reconstructed]
        dct_size += num_coeffs
    Estimated = np.concatenate(Estimated)
    size = 1
    for dim in list(Estimated.shape):
        size = size * dim
    compression_ratio = size / dct_size
    return Estimated, compression_ratio


def threshold_quantization(coefficient, threshold):
    sorted_coeffs = np.sort(np.abs(coefficient.flatten()))[::-1]
    total_energy = np.sum(sorted_coeffs ** 2)
    energy_threshold = threshold * total_energy
    cumulative_energy = 0.0
    num_coeffs = -1
    for coeff in sorted_coeffs:
        cumulative_energy += coeff ** 2
        num_coeffs += 1
        if cumulative_energy >= energy_threshold:
            break
    quantized = np.where(np.abs(coefficient) >= sorted_coeffs[num_coeffs], coefficient, 0)
    return quantized, num_coeffs

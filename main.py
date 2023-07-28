import os
from tensorly.tenalg import multi_mode_dot
from tensorly.decomposition import tucker
from config import Config
from get_data import Dataset
from utils import bit2byte
from lloyds_quantization import *
from entropy_coding import *


train_ratio = 0.9
dataset = 'MBD'       # PlanetLab Harvard72 MBD PMU
config = Config('./data/' + dataset + '.ini')
np.random.seed(config.seed)
ds_handler = Dataset(config, train_ratio)
results_path = './results/{}/'.format(dataset)
if not os.path.exists(results_path):
    os.makedirs(results_path)



def tucker_decomposition(ds_handler):
    Estimated = []
    core_list = []
    factors_list = []
    core_size = 0
    factor_size = 0
    all_batch_data_mask = ds_handler.get_all_batch_data()
    factor_dat = []
    for i in range(len(all_batch_data_mask)):
        batch_data = all_batch_data_mask[i][0]
        batch_mask = all_batch_data_mask[i][1]

        core, factors = tucker(tensor=batch_data, rank=config.features.tolist(), mask=batch_mask)
        core_list += [core]
        factors_list += [factors]
        np.savetxt('./results/{}/core {}.txt'.format(dataset, i), core.reshape(-1), fmt='%.4f')
        core_size += os.stat('./results/{}/core {}.txt'.format(dataset, i)).st_size
        for n in range(3):
            np.savetxt('./results/{}/factor {}_{}.txt'.format(dataset, i, n), factors[n], fmt='%.4f')
            factor_size += os.stat('./results/{}/factor {}_{}.txt'.format(dataset, i, n)).st_size

        tensor = multi_mode_dot(core, factors)
        Estimated += [tensor.squeeze()]
        factor_dat.append(np.concatenate([factors[0].reshape(-1), factors[1].reshape(-1), factors[2].reshape(-1)]))
    np.asarray(factor_dat).tofile('./results/{}/factor.dat'.format(dataset))
    Estimated = np.concatenate(Estimated)
    return Estimated, core_list, factors_list, core_size, factor_size


def quantization_coding(core_list, factors_list, num_core_bits=5, num_factor_bits=10, digit=4):
    Estimated = []
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

        # -----------------------------Decomposition----------------------------
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
    print(np.median(np.abs(quant - abs(core))))
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
    for n in range(3):
        quant = np.round(np.abs(factors[n]) * ((1 << num_bits)-1))
        quant += (factors[n] < 0) * (1 << num_bits)
        quant = quant.astype(int)
        factors_q += [quant]

        dequant = quant.copy()
        sign = (dequant >> num_bits) & 1
        dequant &= ~ (1 << num_bits)
        dequant = (1 - sign * 2) * dequant / ((1 << num_bits)-1)

        factors_t += [dequant]
        print(np.median(np.abs(dequant - np.abs(factors[n]))))
    return factors_q, factors_t







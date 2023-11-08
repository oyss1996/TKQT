import argparse
import numpy as np
from config import Config
from get_data import Dataset
from utils import accuracy, record
from main import tensor_decomposition, quantization_coding
from main import transformation     # DCT DWT

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='TK', help='TK, TT, TR, DCT&DWT')
parser.add_argument('--dataset', type=str, default='MBD', help='PSD, MBD, PlanetLab, Harvard&DWT')
parser.add_argument('--bayes', type=bool, default=False, help='Whether to use Bayesian optimization to search for rank')
args = parser.parse_args()

method = args.method
dataset = args.dataset
bayes = args.bayes
train_ratio = 0.9
config = Config(dataset)
ds_handler = Dataset(config, train_ratio)
all_data = ds_handler.get_all_batch_data()


if method == 'TK' or method == 'TT' or method == 'TR':
    print(method + ' ' + dataset)
    Estimated, core_list, factors_list, core_size, factor_size = tensor_decomposition(all_data, config, method, bayes)
    td_error = accuracy(Estimated, ds_handler)
    cr_td = config.data_size / (core_size + factor_size)    # TT TR: core_size=0
    error_list = [np.asarray(td_error + [cr_td])]
    print('{:6}\tobs nmae\tmiss nmae\tall nmae\tobs nrmse\tmiss nrmse\tall nrmse\tcompresion ratio'.format(dataset))
    print('{:6}\t{:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.2f}'
          .format(method, td_error[0], td_error[1], td_error[2], td_error[3], td_error[4], td_error[5], cr_td))

    if method == 'TK':
        Estimated, codebook_size, total_core_bits, total_factor_bits = quantization_coding(dataset, core_list, factors_list)
        qt_error = accuracy(Estimated, ds_handler)
        cr_qt, cr_tkqt = (core_size + factor_size)/(codebook_size + total_core_bits/8 + total_factor_bits/8), \
                         config.data_size/(codebook_size + total_core_bits/8 + total_factor_bits/8)
        error_list += [np.asarray(qt_error + [cr_qt, cr_tkqt])]

        print('QT    \t{:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.2f}'
              .format(qt_error[0], qt_error[1], qt_error[2], qt_error[3], qt_error[4], qt_error[5], cr_qt))
        print('Data\t|\tCore\tFactor\t|\tcodebook\tindex\tfactor\tcompresion ratio')
        print('{:d}|\t{:d}\t{:d}\t|\t{:d}\t{:d}\t{:d}\t{:5.2f}'.format(config.data_size, core_size, factor_size, codebook_size, total_core_bits, total_factor_bits, cr_tkqt))

    record(dataset, method+'_'+dataset, error_list)


elif method == 'DCT&DWT':
    print('Discrete Cosine / Wavelet Transform')
    print('{:10}\tobs nmae\tmiss nmae\tall nmae\tobs nrmse\tmiss nrmse\tall nrmse\tcompression ratio'.format(dataset))
    error_list = []
    for trans in ['DCT', 'DWT']:
        for theta in [0.9, 0.99, 0.999]:
            Estimated, compression_ratio = transformation(all_data, trans, theta)
            Estimated = np.clip(Estimated, 0, 1)
            dct_error = accuracy(Estimated, ds_handler) + [compression_ratio]
            error_list += [np.asarray(dct_error)]
            print('{}{:1.3f}      \t{:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}'
                  .format(trans, theta, dct_error[0], dct_error[1], dct_error[2], dct_error[3], dct_error[4], dct_error[5], dct_error[6]))
    record(dataset, 'DCT_DWT', error_list)

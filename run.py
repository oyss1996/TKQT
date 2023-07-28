from main import *
from utils import accuracy, record


try:
    print('Tucker decomposition')
    Estimated, core_list, factors_list, core_size, factor_size = tucker_decomposition(ds_handler)
    Estimated = np.clip(Estimated, 0, 1)
    tk_error = accuracy(Estimated, ds_handler)
    error_list = [np.asarray(tk_error)]

    print('Lloyd\'s-max quantization')
    Estimated, codebook_size, total_core_bits, total_factor_bits = quantization_coding(core_list, factors_list)
    Estimated = np.clip(Estimated, 0, 1)
    qt_error = accuracy(Estimated, ds_handler)
    cr_tk, cr_qt, cr_tkqt = config.data_size/(core_size + factor_size), \
                            (core_size + factor_size)/(codebook_size + total_core_bits/8 + total_factor_bits/8), \
                            config.data_size/(codebook_size + total_core_bits/8 + total_factor_bits/8)
    error_list += [np.asarray(list(qt_error) + [cr_tk, cr_qt, cr_tkqt])]

    print('{:10}\ttrain nmae\ttest nmae\tall nmae\ttrain nrmse\ttest nrmse\tall nrmse'.format(dataset))
    print('TK      \t{:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}'
          .format(tk_error[0], tk_error[1], tk_error[2], tk_error[3], tk_error[4], tk_error[5]))
    print('QT      \t{:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}'
          .format(qt_error[0], qt_error[1], qt_error[2], qt_error[3], qt_error[4], qt_error[5]))

    print('Data\t|\tCore\tFactor\t|\tcodebook\tindex\tfactor')
    print('{:d}|\t{:d}\t{:d}\t|\t{:d}\t{:d}\t{:d}'.format(config.data_size, core_size, factor_size, codebook_size, total_core_bits, total_factor_bits))
    print('Compression ratio:\t{:5.4f}\t{:5.4f}\t{:5.4f}'.format(cr_tk, cr_qt, cr_tkqt))
    record(results_path, error_list)



except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

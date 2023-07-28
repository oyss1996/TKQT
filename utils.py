import os
import csv
import numpy as np


def accuracy(estimated, ds_handler):
    estimated = estimated * ds_handler.max_value
    train_data = ds_handler.train_data * ds_handler.max_value
    train_mask = ds_handler.train_mask
    test_data = ds_handler.test_data * ds_handler.max_value
    test_mask = ds_handler.test_mask
    train_nmae, train_nrmse = metric(train_data, train_mask * estimated)
    test_nmae, test_nrmse = metric(test_data, test_mask * estimated)
    all_nmae, all_nrmse = metric(train_data+test_data, (train_mask+test_mask) * estimated)
    return train_nmae, test_nmae, all_nmae, train_nrmse, test_nrmse, all_nrmse


def metric(data, predict):
    error = np.abs(predict - data)
    NMAE = np.sum(error) / np.sum(np.abs(data))
    NRMSE = np.sqrt(np.sum(error ** 2) / np.sum(data ** 2))
    return NMAE, NRMSE


def record(result_path, error_list):
    record_file = result_path + 'error.csv'
    header = ['Obs_nmae', 'Miss_nmae', 'All_nmae', 'Obs_nrmse', 'Miss_nrmse', 'All_nrmse', 'cr_tk', 'cr_qt', 'cr_tkqt']
    with open(record_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for row in error_list:
            writer.writerow(row)
        csvfile.close()
    print("Successfully write to csv file!")


def bit2byte(bit_str, num_bits=None):
    if isinstance(num_bits, int):
        bit_str = ''.join(format(x, '0{}b'.format(num_bits)) for x in bit_str.flatten())  # 将数组中的元素转换为num_bits比特二进制表示
    padding = (8 - (len(bit_str)+3) % 8) % 8  # 计算需要填充的0的个数，使比特串长度是8的倍数，padding<8，占3位
    bit_str = format(padding, '03b') + bit_str   # 在比特串的开始位置添加表示填充长度的3比特二进制数
    bit_str += '0' * padding    # 串尾填充0，使比特串长度是8的倍数
    byte_str = bytearray(int(bit_str[i:i + 8], 2) for i in range(0, len(bit_str), 8))   # 将比特串转换为字节串
    # with open('./results/{}/huffman {}.bin'.format(dataset, i), 'wb') as f:
    #     f.write(bytes(byte_str))
    return bytes(byte_str)


def byte2bit(byte_str):
    bit_list = []
    for byte in byte_str:
        for i in range(8):
            bit = (byte >> i) & 1
            bit_list.append(bit)
    padding = int(''.join(str(num) for num in bit_list[0:3]), 2)
    bit_str = bit_list[3:-padding]
    return np.asarray(bit_str)
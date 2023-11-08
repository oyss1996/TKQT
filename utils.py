import os
import csv
import numpy as np


def accuracy(estimated, ds_handler):
    estimated = np.clip(estimated, 0, 1)
    train_data = ds_handler.train_data
    train_mask = ds_handler.train_mask
    test_data = ds_handler.test_data
    test_mask = ds_handler.test_mask
    train_nmae, train_nrmse = metric(train_data, train_mask * estimated)
    test_nmae, test_nrmse = metric(test_data, test_mask * estimated)
    all_nmae, all_nrmse = metric(train_data+test_data, (train_mask+test_mask) * estimated)
    return [train_nmae, test_nmae, all_nmae, train_nrmse, test_nrmse, all_nrmse]


def accuracy_unit(estimated, ds_handler):
    estimated = np.clip(estimated, 0, 1)
    train_data = ds_handler.train_data
    train_mask = ds_handler.train_mask
    test_data = ds_handler.test_data
    test_mask = ds_handler.test_mask

    error = []
    for i in range(0, ds_handler.num_dim[0] // ds_handler.window):
        est = estimated[i * ds_handler.window: (i + 1) * ds_handler.window]
        trd = train_data[i * ds_handler.window: (i + 1) * ds_handler.window]
        trm = train_mask[i * ds_handler.window: (i + 1) * ds_handler.window]
        ted = test_data[i * ds_handler.window: (i + 1) * ds_handler.window]
        tem = test_mask[i * ds_handler.window: (i + 1) * ds_handler.window]
        tr_nmae, tr_nrmse = metric(trd, trm * est)
        te_nmae, te_nrmse = metric(ted, tem * est)
        al_nmae, al_nrmse = metric(trd+ted, (trm+tem) * est)
        error += [np.asarray([tr_nmae, te_nmae, al_nmae, tr_nrmse, te_nrmse, al_nrmse])]
    return error


def accuracy_unit1(Estimated, ds_handler):
    Estimated = np.clip(Estimated, 0, 1)
    estimated = []
    train_data = []
    train_mask = []
    test_data = []
    test_mask = []
    for i in range(0, ds_handler.num_dim[0] // ds_handler.window):
        estimated.append(Estimated[i * ds_handler.window: (i + 1) * ds_handler.window])
        train_data.append(ds_handler.train_data[i * ds_handler.window: (i + 1) * ds_handler.window])
        train_mask.append(ds_handler.train_mask[i * ds_handler.window: (i + 1) * ds_handler.window])
        test_data.append(ds_handler.test_data[i * ds_handler.window: (i + 1) * ds_handler.window])
        test_mask.append(ds_handler.test_mask[i * ds_handler.window: (i + 1) * ds_handler.window])
    estimated = np.stack((estimated))
    train_data = np.stack((train_data))
    train_mask = np.stack((train_mask))
    test_data = np.stack((test_data))
    test_mask = np.stack((test_mask))
    error = []
    for i in range(0, 4):
        est = estimated[:, i, :, :, :]
        trd = train_data[:, i, :, :, :]
        trm = train_mask[:, i, :, :, :]
        ted = test_data[:, i, :, :, :]
        tem = test_mask[:, i, :, :, :]
        tr_nmae, tr_nrmse = metric(trd, trm * est)
        te_nmae, te_nrmse = metric(ted, tem * est)
        al_nmae, al_nrmse = metric(trd+ted, (trm+tem) * est)
        error += [np.asarray([tr_nmae, te_nmae, al_nmae, tr_nrmse, te_nrmse, al_nrmse])]
    return error

def metric(data, predict):
    error = np.abs(predict - data)
    NMAE = np.sum(error) / np.sum(np.abs(data))
    NRMSE = np.sqrt(np.sum(error ** 2) / np.sum(data ** 2))
    return NMAE, NRMSE


def record(dataset, i, error_list):
    result_path = './results/{}/'.format(dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    record_file = result_path + 'error_' + i + '.csv'
    header = ['train_nmae', 'test_nmae', 'train_nrmse', 'test_nrmse']
    with open(record_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(header)
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


def tensor2tuple(tensor_):
    tuple_ = []
    if len(tensor_.shape) == 3:
        [I, J, K] = list(tensor_.shape)
        for i in range(I):
            for j in range(J):
                for k in range(K):
                        if tensor_[i, j, k] != 0:
                            tuple_ += [np.asarray([i, j, k, tensor_[i, j, k]])]

    elif len(tensor_.shape) == 4:
        [I, J, K, L] = list(tensor_.shape)
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    for l in range(L):
                        if tensor_[i, j, k, l] != 0:
                            tuple_ += [np.asarray([i, j, k, l, tensor_[i, j, k, l]])]
    return np.asarray(tuple_)


def tuple2tensor(shape, tuple_):

    tensor_ = np.zeros(shape)
    indices = tuple_[:, 0:-1].astype(int)
    values = tuple_[:, -1]

    if indices.shape(1) == 3:
        for n in range(len(values)):
            [i, j, k] = indices[n].tolist()
            tensor_[i, j, k] = values[n]
    if indices.shape(1) == 4:
        for n in range(len(values)):
            [i, j, k, l] = indices[n].tolist()
            tensor_[i, j, k, l] = values[n]
    return tensor_





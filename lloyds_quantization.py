import numpy as np


def quantiz(sig, partition, *varargin):
    # decode: Y = CODEBOOK[INDX]
    # Initial error checks ------------------------------------
    if len(sig) == 0 or len(sig.shape) != 1 or not np.all(np.isreal(sig)):
        raise ValueError("Invalid signal")

    if len(partition) == 0 or len(partition.shape) != 1 or not np.all(np.isreal(partition)) or \
            not np.all(np.sort(partition) == partition):
        raise ValueError("Invalid partition")

    # Compute INDX
    indx = np.zeros(sig.shape, dtype=np.int32)
    for i in range(len(partition)):
        indx += (sig > partition[i])

    if len(varargin) < 1:  # Don't output quantized values
        return indx

    # Compute QUANTV
    codebook = varargin[0]
    if len(codebook) == 0 or len(codebook.shape) != 1 or not np.all(np.isreal(codebook)) or \
            len(codebook) != len(partition) + 1:
        raise ValueError("Invalid codebook")
    quantv = codebook[indx]

    # Compute distortion
    distor = np.sum((sig - quantv) ** 2) / len(sig)
    return indx, quantv, distor


def lloyds(training_set, ini_codebook, tol=1e-7):
    # validation verification and format conversion
    if len(training_set) < 1:
        raise ValueError('EmptyTRAINING_SET')
    elif len(training_set.shape) > 1:
        raise ValueError('NonVectorTRAINING_SET')
    elif not np.all(np.isreal(ini_codebook)) or ini_codebook is None:
        raise ValueError('InvalidINI_CODEBOOK')
    elif isinstance(ini_codebook, int):
        # place the ini_codebook at the center
        if ini_codebook < 1:
            raise ValueError('NonPositiveINI_CODEBOOK')
        min_training = np.min(training_set)
        max_training = np.max(training_set)
        int_training = (max_training - min_training) / ini_codebook
        if int_training <= 0:
            raise ValueError('InvalidTRAINING_SET')
        codebook = np.linspace(min_training + int_training / 2,
                               max_training - int_training / 2, ini_codebook)
    else:
        min_training = np.min(training_set)
        max_training = np.max(training_set)
        codebook = np.sort(ini_codebook)
        ini_codebook = len(codebook)

    # initial partition
    partition = (codebook[1:ini_codebook] + codebook[0:ini_codebook - 1]) / 2

    # distortion computation, initialization
    index, quant, distor = quantiz(training_set, partition, codebook)
    last_distor = 0

    ter_cond2 = np.finfo(float).eps * max_training
    if distor > ter_cond2:
        rel_distor = abs(distor - last_distor) / distor
    else:
        rel_distor = distor

    while (rel_distor > tol) and (rel_distor > ter_cond2):
        # using the centroid condition, find the optimal codebook.
        for i in range(ini_codebook):
            waste1 = np.where(index == i)[0]
            if len(waste1) > 0:
                codebook[i] = np.mean(training_set[waste1])
            else:
                if i == 0:
                    tmp = training_set[training_set <= partition[0]]
                    if len(tmp) == 0:
                        codebook[0] = (partition[0] + min_training) / 2
                    else:
                        codebook[0] = np.mean(tmp)
                elif i == ini_codebook - 1:
                    tmp = training_set[training_set >= partition[-1]]
                    if len(tmp) == 0:
                        codebook[-1] = (max_training + partition[-1]) / 2
                    else:
                        codebook[-1] = np.mean(tmp)
                else:
                    tmp = training_set[(training_set >= partition[i - 1]) & (training_set <= partition[i])]
                    if len(tmp) == 0:
                        codebook[i] = (partition[i] + partition[i - 1]) / 2
                    else:
                        codebook[i] = np.mean(tmp)

        # compute sorted partition
        partition = np.sort((codebook[1:ini_codebook] + codebook[0:ini_codebook - 1]) / 2)

        # testing condition
        last_distor = distor
        index, quant, distor = quantiz(training_set, partition, codebook)
        if distor > ter_cond2:
            rel_distor = abs(distor - last_distor) / distor
        else:
            rel_distor = distor

    return partition, codebook, distor, rel_distor

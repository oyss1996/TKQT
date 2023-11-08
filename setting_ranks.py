import numpy as np
import tensorly as tl


def rules(delta, ratio, threshold):
    if len(delta) < 12:
        for i in range(1, len(delta)):
            if ratio[i-1]-ratio[i] < 0.01:
                return i + 1
        return len(delta)

    for i in range(len(delta)):
        if ((delta[i] < 0.01 and (np.all(delta[i:i + 10] <= 0.01)) or
             (delta[i] < threshold and np.all(np.round(np.abs(delta[i:i + 10] - delta[i]), 3) <= 0.01)))):
                return i + 1

    if ratio[-1] < 0.01:
        rank = np.argmin(delta[int(np.ceil(len(delta)/2)) - 3: int(np.ceil(len(delta)/2)) + 3]) + 1 + int(np.ceil(len(delta)/2)) - 3
    else:
        rank = np.argmin(delta[2*len(delta)//3:]) + 1 + 2*len(delta)//3
    return rank


def tucker_ranks(tensor):
    ranks = []
    for i in range(tensor.ndim):
        unfold = tl.base.unfold(tensor, i)
        _, diag, _ = np.linalg.svd(unfold, full_matrices=False)
        delta = diag[:-1] / diag[1:] - 1
        ratio = diag[1:] / diag[0]
        threshold = np.percentile(delta, 25)
        ranks.append(rules(delta, ratio, threshold))

        # # SV80
        # total_energy = 0.8 * np.sum(diag)
        # for r in range(1, len(diag)+1):
        #     if np.sum(diag[0:r]) >= total_energy:
        #         ranks.append(r)
        #         break
    return ranks

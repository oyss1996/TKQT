import numpy as np
import optuna
from tensorly.decomposition import tucker
from tensorly.tenalg import multi_mode_dot

import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


class BayeOpt(object):
    def __init__(self, shape, n_trials):
        self.shape = shape
        self.n_trials = n_trials
        self.data_size = np.prod(shape)

        self.data = None
        self.mask = None
        self.normdata = 1
        self.normdata2 = 1
        self.train_tensor = None
        self.train_mask = None

    def estimate_ranks(self, data):
        self.data = data[0]
        self.mask = data[1]
        self.normdata = np.sum(np.abs(self.data))
        self.normdata2 = np.sum(self.data ** 2)

        num_sample = int(np.ceil(np.max(self.mask) * 0.9))
        self.train_mask = ((self.mask > 0) & (self.mask <= num_sample)).astype(np.float32)
        self.train_tensor = self.train_mask * data[0]

        best_params, _ = self.optimizer_optuna(self.n_trials)
        ranks = []
        for _, value in best_params.items():
            ranks.append(value)
        return ranks

    def validation_loss(self, rank0, rank1, rank2, rank3):
        if rank3 is None:
            ranks = [rank0, rank1, rank2]
        else:
            ranks = [rank0, rank1, rank2, rank3]

        core, factors = tucker(tensor=self.train_tensor, rank=ranks, mask=self.train_mask, n_iter_max=0)
        estimate_tensor = multi_mode_dot(core, factors)
        error = np.multiply(self.mask, np.abs(estimate_tensor - self.data))
        NMAE = np.sum(error) / self.normdata
        NRMSE = np.sqrt(np.sum(error ** 2) / self.normdata2)
        CR_inverse = (np.prod(list(core.shape)) + np.sum(np.prod(list(factor.shape)) for factor in factors)) / self.data_size
        return NMAE+NRMSE+CR_inverse

    def optuna_objective(self, trial):
        # Define parameter space
        rank0 = trial.suggest_int("rank0", 1, self.shape[0])  # int type, (name, upper, lower, step=1)
        rank1 = trial.suggest_int("rank1", 1, self.shape[1])
        rank2 = trial.suggest_int("rank2", 1, self.shape[2])
        if len(self.shape) >= 4:    # 4D tensor
            rank3 = trial.suggest_int("rank3", 1, self.shape[3])
        else:
            rank3 = None

        score = self.validation_loss(rank0, rank1, rank2, rank3)

        return score

    def optimizer_optuna(self, n_trials):
        # Default algorithm "TPE", default optimization direction "minimize"
        study = optuna.create_study(pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=10))

        # Start optimization
        study.optimize(self.optuna_objective    # objective function
                       , n_trials=n_trials      # Maximum number of trials
                       )

        # Optimized results can be called directly from the optimized object study
        # Print the optimal parameters with the best loss values
        # print("\n", "\n", "best params: ", study.best_trial.params,
        #       "\n", "\n", "best score: ", study.best_trial.values,
        #       "\n")

        return study.best_trial.params, study.best_trial.values

import numpy as np

class os_random(object) :
    def __init__(self):
        pass

    def perform(self, exps) :

        exps = exps.argmax(-1);

        tot_oslrs = [np.array([0.33, 0.33, 0.33])]

        for r_idx, exp_row in enumerate(exps):
            rand_lr = np.random.random(size=(3,));
            tot_oslrs.append(rand_lr/rand_lr.sum())

        return np.stack(tot_oslrs)


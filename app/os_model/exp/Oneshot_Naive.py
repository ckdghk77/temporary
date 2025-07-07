import numpy as np

class os_naive(object) :
    def __init__(self):
        pass

    def perform(self, exps) :

        exps = exps.argmax(-1);

        novel_stim = np.where(exps[:, :5] == 2)[0][-1];
        novel_rew = np.where(exps[:, 5:] == 4)[0][-1];

        if novel_stim == novel_rew :
            lr = np.array([0.0, 0.0, 1.0])
        else :
            lr = np.array([0.33, 0.33, 0.33])

        tot_oslrs = [lr] * (exps.shape[0]+1)

        return np.stack(tot_oslrs)


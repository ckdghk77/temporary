
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
from app.os_model.exp.Oneshot_sangwan import os_sangwan
import numpy as np


class mlrose_OS(object):
    def __init__(self, fitting_type, init_state, max_attempts=10, max_iters=1000, random_state=1):
        self.fitting_type = fitting_type
        self.init_state = init_state
        self.state_len = len(init_state)
        self.max_attempts = max_attempts
        self.max_iters = max_iters
        self.random_state = random_state

    def fitness_func(self, state):
        os_model = os_sangwan(primacy=state[0], recency=state[1], lr=state[2])

        loss = 0
        for stim, out in zip(self.stims, self.outs) :
            m_out = os_model.perform(stim)
            m_out = m_out[-1]
            loss += abs(m_out[-1] - out[-1])
            #loss += (m_out[-1] - out[-1])**2
            #m_out_os_idx = m_out[-1:] - sum(m_out[:2]) / 2.0;
            #out_os_idx = out[-1:] - sum(out[:2])/2.0
            #loss += sum((m_out_os_idx - out_os_idx)**2)
        #print(loss)
        #if self.fitting_type == "annealing" :
            #loss += (np.random.random()-0.5)*0.2

        return loss

    def fit(self, stims, outs):
        fit_func = mlrose.CustomFitness(self.fitness_func)
        prob_func = mlrose.ContinuousOpt(length=self.state_len,
                                         min_val=0.1, max_val=2.0,
                                         fitness_fn=fit_func, maximize=False, step=0.1)
        self.stims, self.outs = stims, outs

        if self.fitting_type == "annealing" :
            best_state, best_fitness = mlrose.simulated_annealing(prob_func, schedule=mlrose.ExpDecay(),
                                                                  max_attempts=self.max_attempts,
                                                                  max_iters=self.max_iters,
                                                                  init_state=self.init_state,
                                                                  random_state=self.random_state)
        elif self.fitting_type == "genetic" :
            best_state, best_fitness = mlrose.genetic_alg(prob_func, pop_size=100,
                                                                    mutation_prob=0.2,
                                                                    max_attempts=10,
                                                                  max_iters=self.max_iters,
                                                                  random_state=self.random_state)
        elif self.fitting_type == "rhill_climb":
            best_state, best_fitness = mlrose.random_hill_climb(prob_func,
                                                                    max_attempts=self.max_attempts,
                                                                    restarts=10,
                                                                  max_iters=self.max_iters,
                                                                  random_state=self.random_state)



        print("Best_state: {}/{}".format(best_state, best_fitness))

        base_fitness = self.fitness_func([0.33, 0.33, 0.1])

        return best_state, best_fitness, base_fitness





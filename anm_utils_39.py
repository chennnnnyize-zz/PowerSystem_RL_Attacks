#REwrite ANM6Easy

import torch
import os
import pprint
from anm39 import ANM39

from collections import namedtuple
from itertools import count
import math
import random
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


eps_train=1.0
eps_train_final=0.05
eps_test=0.005
epoch=100
step_per_epoch=10000
collect_per_step=10
batch_size=32
training_num=16
test_num=10
logdir='log'
log_path = os.path.join(logdir, 'dqn')

class ANM39_Easy(ANM39):
    """The :code:`ANM6Easy-v0` task."""

    def __init__(self):
        observation = 'state'  # fully observable environment
        K = 1
        delta_t = 0.25         # 15 minutes between timesteps
        gamma = 0.995
        lamb = 100
        aux_bounds = np.array([[0, 24 / delta_t - 1]])
        costs_clipping = (1, 100)
        super().__init__(observation, K, delta_t, gamma, lamb, aux_bounds,
                         costs_clipping)

        # Consumption and maximum generation 24-hour time series.
        self.P_loads = _get_load_time_series()
        self.P_maxs = _get_gen_time_series()

    def init_state(self):
        #####################n_dev, n_gen, n_des = 11, 3, 2
        n_dev, n_gen, n_des = 52, 13, 2

        state = np.zeros(2 * n_dev + n_des + n_gen + self.K)

        t_0 = self.np_random.randint(0, int(24 / self.delta_t))
        state[-1] = t_0

        ######################### Load (P, Q) injections.
        #for dev_id, p_load in zip([1, 3, 5, 7, 9], self.P_loads):
        for dev_id, p_load in zip([1, 3, 5, 7, 9, 11,12,13,14,15,16,17,18,19,20,21,22,
                                   23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41], self.P_loads):
            state[dev_id] = p_load[t_0]
            state[n_dev + dev_id] = \
                p_load[t_0] * self.simulator.devices[dev_id].qp_ratio

        # Non-slack generator (P, Q) injections.
        ##########################for idx, (dev_id, p_max) in enumerate(zip([2, 4, 10], self.P_maxs)):
        for idx, (dev_id, p_max) in enumerate(zip([2, 4, 10, 42,43,44,45,46,47,48,49,50,51], self.P_maxs)):
            state[2 * n_dev + n_des + idx] = p_max[t_0]
            state[dev_id] = p_max[t_0]
            state[n_dev + dev_id] = \
                self.np_random.uniform(self.simulator.devices[dev_id].q_min,
                                       self.simulator.devices[dev_id].q_max)

        # Energy storage unit.
        for idx, dev_id in enumerate([6, 8]):
            state[2 * n_dev + idx] = \
                self.np_random.uniform(self.simulator.devices[dev_id].soc_min,
                                       self.simulator.devices[dev_id].soc_max)
        return state

    def next_vars(self, s_t):
        aux = int((s_t[-1] + 1) % (24 / self.delta_t))
        aux_2=aux-1
        vars = []
        for p_load in self.P_loads:
            vars.append(p_load[aux_2])
        for p_max in self.P_maxs:
            vars.append(p_max[aux_2])

        vars.append(aux)

        return np.array(vars)

    def reset(self, date_init=None):
        obs = super().reset()

        # Reset the time of the day based on the auxiliary variable.
        date = self.date
        new_date = self.date + self.state[-1] * self.timestep_length
        super().reset_date(new_date)

        return obs


def _get_load_time_series():
    """Return the fixed 24-hour time-series for the load injections."""

    # Device 1 (residential load).
    s1 = - np.ones(25)
    s12 = np.linspace(-1.5, -4.5, 7)
    s2 = - 5 * np.ones(13)
    s23 = np.linspace(-4.625, -2.375, 7)
    s3 = - 2 * np.ones(13)
    P1 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1],
                         s1[:4]))

    # Device 3 (industrial load).
    s1 = -4 * np.ones(25)
    s12 = np.linspace(-4.75, -9.25, 7)
    s2 = - 10 * np.ones(13)
    s23 = np.linspace(-11.25, -18.75, 7)
    s3 = - 20 * np.ones(13)
    P3 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1],
                         s1[:4]))

    # Device 5 (EV charging station load).
    s1 = np.zeros(25)
    s12 = np.linspace(-3.125, -21.875, 7)
    s2 = - 25 * np.ones(13)
    s23 = np.linspace(-21.875, -3.125, 7)
    s3 = np.zeros(13)
    P5 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1],
                         s1[:4]))

    # Device 7 (residential load).
    s1 = - np.ones(25)
    s12 = np.linspace(-1.6, -4.6, 7)
    s2 = - 5 * np.ones(13)
    s23 = np.linspace(-4.625, -2.375, 7)
    s3 = - 2 * np.ones(13)
    P7 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1],
                         s1[:4]))
    # Device 9 (residential load).
    s1 = - np.ones(25)
    s12 = np.linspace(-1.6, -4.6, 7)
    s2 = - 5 * np.ones(13)
    s23 = np.linspace(-4.625, -2.375, 7)
    s3 = - 2 * np.ones(13)
    P9 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1],
                         s1[:4]))

    P9=np.tile(P9,(32,1))*0.3
    #################P_loads = np.vstack((P1, P3, P5, P7, P9))
    P_loads = np.vstack((P1, P3, P5, P7, P9))
    assert P_loads.shape == (36, 96)

    return P_loads


def _get_gen_time_series():
    """Return the fixed 24-hour time-series for the generator maximum production."""

    # Device 2 (residential PV aggregation).
    s1 = np.zeros(25)
    s12 = np.linspace(0.5, 3.5, 7)
    s2 = 4 * np.ones(13)
    s23 = np.linspace(7.25, 36.75, 7)
    s3 = 30 * np.ones(13)
    P2 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1],
                         s1[:4]))

    # Device 4 (wind farm).
    s1 = 40 * np.ones(25)
    s12 = np.linspace(36.375, 14.625, 7)
    s2 = 11 * np.ones(13)
    s23 = np.linspace(14.725, 36.375, 7)
    s3 = 40 * np.ones(13)
    P4 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1],
                         s1[:4]))

    P10=np.copy(P4)+0.05
    P10 = np.tile(P10, (11, 1))*2.0

    ##################P_maxs = np.vstack((P2, P4, P10))
    P_maxs = np.vstack((P2, P4, P10))
    assert P_maxs.shape == (13, 96)

    return P_maxs



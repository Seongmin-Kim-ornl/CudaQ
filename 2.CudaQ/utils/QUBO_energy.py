import numpy as np
import pandas as pd


# energy state
def cal_QUBO_energy(x, Q):
    x = np.array(x)
    QUBO_energy = x@Q@np.transpose(x)
    energy = QUBO_energy.item()
    return energy


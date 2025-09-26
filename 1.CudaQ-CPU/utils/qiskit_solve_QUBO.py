# Qiskit imports
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler, SamplerV2
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

import numpy as np
import time
import warnings
warnings.simplefilter("ignore")
#from .QITE import QUBO_SOLVER

def solve_Q(Q):
    
    qp = QuadraticProgram()
    [qp.binary_var() for _ in range(Q.shape[0])]
    qp.minimize(quadratic=Q)
    optimizer = COBYLA()
    sampler = Sampler(backend_options={'device': "CPU", 'method': 'automatic'}, run_options={'shots': 1000})
    # sampler = SamplerV2()
    # sampler._backend.set_options(device=device, method='automatic') # SamplerV2 will not adhere to options dictionary passed. maybe a bug
    qaoa_mes = QAOA(optimizer=optimizer, sampler=sampler, reps=2)
    qaoa = MinimumEigenOptimizer(qaoa_mes)
    qaoa_result = qaoa.solve(qp)
    return qaoa_result.x


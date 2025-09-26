import numpy as np
import os

def load_QUBO(QUBO_size, ctn):
    #QUBO_file = open(f"../../3.QAOA_GPT/cuda-quantum/QAOA-GPT/0_QUBOs/QUBO_nodes{QUBO_size}_edgep95_{ctn}.txt", 'r')
    QUBO_file = open(f"./0_QUBOs/QUBO_nodes{QUBO_size}_edgep95_{ctn}.txt", 'r')
    #QUBO_file = open('./0_QUBOs/QUBO_'+str(QUBO_size)+'.txt', 'r') # import Q matrix
    A = QUBO_file.read()
    A = np.asmatrix(A)
    m = int(np.size(A)**(1/2))  # number of binary variables
    A = np.reshape(A, (m,m))
    Q = A
    Q = np.array(Q)
    return Q


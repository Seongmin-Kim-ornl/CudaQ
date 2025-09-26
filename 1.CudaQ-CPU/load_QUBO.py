import numpy as np
import os

def load_QUBO(QUBO_size):
    QUBO_file = open('QUBO_'+str(QUBO_size)+'.txt', 'r') # import Q matrix
    A = QUBO_file.read()
    A = np.asmatrix(A)
    m = int(np.size(A)**(1/2))  # number of binary variables
    A = np.reshape(A, (m,m))
    Q = A
    Q = np.array(Q)
    return Q

print(load_QUBO(100).shape)
print(load_QUBO(300).shape)
print(load_QUBO(500).shape)
print(load_QUBO(1000).shape)

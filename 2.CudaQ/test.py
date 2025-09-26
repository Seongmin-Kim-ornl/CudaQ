import numpy as np
import random, time, logging
import utils
#from test2 import solve_Q

def generate_random_qubo(n, density=1, weight_range=(-1, 1)):
        Q = np.zeros((n, n))
        
        # Fill diagonal elements (linear terms)
        for i in range(n):
            Q[i, i] = round(random.uniform(weight_range[0], weight_range[1]), 2)
        
        # Fill upper triangular part (quadratic terms)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    Q[i, j] = round(random.uniform(weight_range[0], weight_range[1]), 2)
        
        return Q
    

n_nodes = 18
Q_matrix = generate_random_qubo(n_nodes, density=1, weight_range=(-1, 1))
Q_matrix = np.array(Q_matrix)

tic = time.time()
bits = utils.solve_Q(Q_matrix)
print(f"solution: {bits}, time: {time.time() - tic}")
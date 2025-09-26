import numpy as np
import time, random
import cudaq
from cudaq import spin
import numpy as np
from typing import List
import networkx as nx
from pyqubo import Binary
from scipy.optimize import minimize


def solve_Q(Q):
    cudaq.set_target("nvidia") # "nvidia" for GPU, "qpp-cpu" for CPU
    max_iters = 5000
    num_layers = 1
    num_params = 2 * num_layers
    N = len(Q)
    problem_size = N
    
    # Step 1: Define binary variables
    variables = [Binary(f"x{i}") for i in range(N)]
    
    # Step 2: Build the Hamiltonian
    objective = 0
    
    # Add diagonal terms (linear)
    for i in range(N):
        objective += Q[i][i] * variables[i]
    
    # Add quadratic terms (upper off-diagonal only)
    for i in range(N):
        for j in range(i+1, N):
            if Q[i][j] != 0:
                objective += Q[i][j] * variables[i] * variables[j]
    
    # Step 3: Compile the model
    model = objective.compile()

    #print(f"---------- model compile done ----------")

    # Convert the pre-defined model to QUBO
    qubo, offset = model.to_qubo()
        
    def to_cudaq(model):    
        num_qubits = len(model.variables)
        h, J, offset = model.to_ising()
    
        # Map variables to qubit indices
        vars_set = set(h) | {v for pair in J for v in pair}
        var_to_index = {var: idx for idx, var in enumerate(sorted(vars_set))}
    
        # Initialization
        coeff_matrix = np.zeros((num_qubits, num_qubits), dtype=float)
        H = 0
    
        # Fill in linear terms (diagonal)
        for var, weight in h.items():
            idx = var_to_index[var]
            coeff_matrix[idx, idx] = float(weight)
            H += weight * spin.z(idx)
    
        # Fill in quadratic terms (off-diagonal)
        for (v1, v2), weight in J.items():
            i, j = var_to_index[v1], var_to_index[v2]
            coeff_matrix[i, j] = float(weight)
            coeff_matrix[j, i] = float(weight)  # symmetric
            H += weight * spin.z(i) * spin.z(j)
    
        H += offset
        flattened_coeff = coeff_matrix.flatten()
    
        return num_qubits, flattened_coeff, H
    
    
    num_qubits, flattened_coeff, H = to_cudaq(model)
    #print("Number of qubits:", num_qubits)
    #print("Flattened coefficient matrix:", flattened_coeff)
    #print("Hamiltonian:", H)
    
    
    
    
    @cudaq.kernel
    def kernel_qaoa(thetas: List[float]):
        """
        QAOA ansatz using a flattened coefficient matrix for linear and quadratic terms.
        """
        qubits = cudaq.qvector(num_qubits)
        h(qubits)
    
        for layer in range(num_layers):
            gamma = thetas[layer]
            beta = thetas[layer + num_layers]
    
            # Apply problem unitary
            for idx, coeff in enumerate(flattened_coeff):
                if coeff != 0:
                    i = idx // num_qubits  # Row index
                    j = idx % num_qubits   # Column index
    
                    if i == j:
                        # Linear term (diagonal)
                        rz(2.0 * coeff * gamma, qubits[i])
                    elif i < j:
                        # Quadratic term (off-diagonal)
                        x.ctrl(qubits[i], qubits[j])
                        rz(2.0 * coeff * gamma, qubits[j])
                        x.ctrl(qubits[i], qubits[j])
    
            # Apply mixer unitary
            for i in range(num_qubits):
                rx(2.0 * beta, qubits[i])
    
    
    
    # Specify the optimizer and its initial parameters. Make it repeatable.
    cudaq.set_random_seed(13)
    optimizer = cudaq.optimizers.COBYLA()
    optimizer.max_iterations = max_iters  # Increase if needed
    #optimizer.tol = 1e-4           # Increase tolerance to accept small changes
    np.random.seed(13)
    optimizer.initial_parameters = np.array([np.pi/4.0, np.pi/8.0]*num_layers)
    
    
    
    # Define the simulation target:

    
    # Define the objective, return ``
    def objective(parameters):
        return cudaq.observe(kernel_qaoa, H, parameters).expectation()
    
    from scipy.optimize import minimize
    
    # Initial parameter guess (same as before)
    initial_gamma = np.pi / 4.0
    initial_beta = np.pi / 8.0
    init_params = [initial_gamma, initial_beta] * num_layers

    #print(f"---------- QAOA CudaQ preparation done ----------")
    # Run optimization using scipy COBYLA
    tic = time.time()
    result = minimize(
        objective,
        x0=init_params,
        method='COBYLA',
        tol=1e-4,
        options={'maxiter': max_iters}
    )
    
    optimal_expectation = result.fun
    optimal_parameters = result.x
    
    #print(f"---------- QAOA estimator done ----------")     
    # Sample the final QAOA circuit with the optimized parameters
    sample_result = cudaq.sample(kernel_qaoa, 
                                 list(optimal_parameters), shots_count=10000)
    elapsed_time = time.time() - tic
    #print(f"---------- QAOA sampler done ----------")   
    # Print the most probable bitstring
    most_likely_binary = sample_result.most_probable()
    #print(f"---------- sample the best bitstring done ----------") 
    
    
    def reorder_labels(N):
        labels = [f"x{i}" for i in range(N+1)]  # base labels: x0, x1, ..., xN
        
        if N < 10:
            # just normal order
            return labels
        
        elif 10 < N < 20:
            # [x0, x1, x10, x11, x2, x3, ..., x9, x12, x13, ..., xN]
            # but you said "[x0, x1, x10, x11, x2, ... xN]"
            # let's insert all x10+ after x1, then continue from x2 to x9, then x12+ if any
            # better interpretation: after x1 insert all x10..xN, then continue from x2 to x9
    
            # split into parts
            part1 = labels[:2]          # x0, x1
            part2 = labels[10:N+1]      # x10, ..., xN
            part3 = labels[2:10]        # x2,..., x9
            return part1 + part2 + part3
        
        elif 20 < N < 30:
            # [x0, x1, x10..x19, x2, x21..xN, x3, x4, ...]
            part1 = labels[:2]          # x0, x1
            part2 = labels[10:20]       # x10..x19
            part3 = labels[2:3]         # x2
            part4 = labels[20:N+1]      # x21..xN
            part5 = labels[3:10]        # x3..x9 (example)
            # combine as asked:
            return part1 + part2 + part3 + part4 + part5
        
        else:
            # default: just return original
            return labels
    
    # Get current order of labels matching your binary_vector
    current_order = reorder_labels(problem_size-1)
    
    
    binary_array = np.array([int(bit) for bit in most_likely_binary], dtype=int)
    inverted_bit_array =  1 - binary_array
    
    # Check length consistency
    assert len(current_order) == len(inverted_bit_array), "Label and vector length mismatch"
    
    # Map label to current vector value
    label_to_value = {label: val for label, val in zip(current_order, inverted_bit_array)}
    
    # Sort labels by their numeric index to get natural order: x0, x1, x2, ...
    sorted_labels = sorted(current_order, key=lambda x: int(x[1:]))
    
    # Reorder vector to natural order
    sorted_vector = np.array([label_to_value[label] for label in sorted_labels])
    
    return sorted_vector


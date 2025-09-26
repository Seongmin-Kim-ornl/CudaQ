import numpy as np

def flip_bits(Q, x_optimal):
  # each end (0 & -1) is hard to be flipped, so manually do
  x_new_subQUBO = np.array(x_optimal)
  x_new_subQUBO[0] = 0
  QUBO_original = x_optimal@Q@np.transpose(x_optimal)
  QUBO_new = x_new_subQUBO@Q@np.transpose(x_new_subQUBO)
  if QUBO_original > QUBO_new:
      x_optimal = x_new_subQUBO
  
  x_new_subQUBO = np.array(x_optimal)
  x_new_subQUBO[0] = 1
  QUBO_original = x_optimal@Q@np.transpose(x_optimal)
  QUBO_new = x_new_subQUBO@Q@np.transpose(x_new_subQUBO)
  if QUBO_original > QUBO_new:
      x_optimal = x_new_subQUBO
      
  x_new_subQUBO = np.array(x_optimal)
  x_new_subQUBO[-1] = 0
  QUBO_original = x_optimal@Q@np.transpose(x_optimal)
  QUBO_new = x_new_subQUBO@Q@np.transpose(x_new_subQUBO)
  if QUBO_original > QUBO_new:
      x_optimal = x_new_subQUBO
      
  x_new_subQUBO = np.array(x_optimal)
  x_new_subQUBO[-1] = 1
  QUBO_original = x_optimal@Q@np.transpose(x_optimal)
  QUBO_new = x_new_subQUBO@Q@np.transpose(x_new_subQUBO)
  if QUBO_original > QUBO_new:
      x_optimal = x_new_subQUBO

  return x_optimal


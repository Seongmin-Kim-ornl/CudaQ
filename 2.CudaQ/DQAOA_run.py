from mpi4py import MPI
from DQAOA import DQAOA
import numpy as np
import random, time, logging
import utils

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# main
problem_size_list = [50, 100, 200, 300, 400, 500]
sub_QUBO_size_list = [4,6,8,10,12,14,16,18,20,22,24]

num_subQUBOs = size
num_DQAOA_iters_list = [10]


for problem_size in problem_size_list:
    print('='*77)
    #print('='*77)
    for ctn in range (5):
        Q = utils.load_QUBO(problem_size, ctn)
        for sub_QUBO_size in sub_QUBO_size_list:
            for num_DQAOA_iters in num_DQAOA_iters_list:
                energy_list = []
                time_list = []
                for iter in range(6):
                    if rank == 0:
                        tic = time.time()
                  
                    dqaoa = DQAOA(Q, sub_QUBO_size, num_subQUBOs, num_DQAOA_iters)
                    solution = dqaoa.run()
                
                    comm.Barrier()
                
                    if rank == 0:
                        elapsed_time = time.time() - tic
                        print(f"energy: {utils.cal_QUBO_energy(solution, Q)}")
                        print(f"elapsed time: {elapsed_time}")
                        energy_list.append(utils.cal_QUBO_energy(solution, Q))
                        time_list.append(elapsed_time)

                    comm.Barrier()
                
                if rank == 0:
                    print(f"======= DQAOA result =======")
                    print(f"problem size: {problem_size}, ctn: {ctn}, subQ size: {sub_QUBO_size}, iters: {num_DQAOA_iters}, num subQ: {num_subQUBOs}")
                    print(f"avg energy {np.mean(np.array(energy_list[1:6]))}, avg time: {np.mean(np.array(time_list[1:6]))}")
                    print(f"======= DQAOA result =======")

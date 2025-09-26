from mpi4py import MPI
import numpy as np
import random, time, logging
import utils

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class DQAOA:
    def __init__(self, Q, sub_QUBO_size, num_sub_QUBOs, num_DQAOA_iters):
        self.Q = Q
        self.problem_size = len(Q)
        self.sub_QUBO_size = sub_QUBO_size
        self.num_sub_QUBOs = num_sub_QUBOs
        self.num_DQAOA_iters = num_DQAOA_iters
        self.idx_list = list(range(self.problem_size))
        
        # Logger setup
        #logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        #self.logger = logging.getLogger(__name__)

    def initialize(self):
        np.random.seed(12345)
        self.x_optimal = np.random.choice([0, 1], size=self.problem_size)

    def decompose(self):
        random.seed(time.time())
        choicelist = random.sample(self.idx_list, self.sub_QUBO_size)
        choicelist.sort()
        rand_subQUBO = np.zeros((self.sub_QUBO_size, self.sub_QUBO_size))
        for sub_i in range(self.sub_QUBO_size):
            for sub_j in range(sub_i, self.sub_QUBO_size):
                rand_subQUBO[sub_i,sub_j] = self.Q[choicelist[sub_i], choicelist[sub_j]]
                    
        return rand_subQUBO, choicelist

    def solve(self, Q):
        # solve subQUBOs
        return utils.solve_Q(Q)

    def aggregate(self, x_sub, choicelist):
        for idx in range(self.sub_QUBO_size):
            x_new_subQUBO = np.array(self.x_optimal)
            x_new_subQUBO[choicelist[idx]] = x_sub[idx]
            
            QUBO_original = utils.cal_QUBO_energy(self.x_optimal, self.Q)
            QUBO_new = utils.cal_QUBO_energy(x_new_subQUBO, self.Q)
            if QUBO_original > QUBO_new:
                self.x_optimal = x_new_subQUBO

    def run(self):
        # Runs the distributed QAOA optimization using MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        MPI_size = comm.Get_size()
        self.initialize()

        for cycle in range(self.num_DQAOA_iters):
            if rank == 0:
                #print(f"Cycle {cycle+1} of {self.num_DQAOA_iters} started... {MPI.Wtime()}")
                subQUBO_list, choice_list = [], []
    
                # Generate all subproblems
                for dist_num in range(1, MPI_size):
                    subQUBO, choices = self.decompose()
                    comm.send(subQUBO, dest=dist_num)
                    comm.send(choices, dest=dist_num)
                    
            if rank != 0:
                recv_data_subQUBO = comm.recv(source=0)
                recv_data_choicelist = comm.recv(source=0)
                x_subQUBO = self.solve(recv_data_subQUBO)
                comm.send(x_subQUBO, dest=0)
                comm.send(recv_data_choicelist, dest=0)

            if rank == 0:
                for rank_recv_n_new in range(1, MPI_size):
                    x_subQUBO_rank = comm.recv(source=rank_recv_n_new)
                    choicelist = comm.recv(source=rank_recv_n_new)
                    self.aggregate(x_subQUBO_rank, choicelist)

        return self.x_optimal        


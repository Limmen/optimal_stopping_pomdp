import time
from optimal_stopping_pomdp.optimal_stopping_env.env_util import EnvUtil
import optimal_stopping_pomdp.util.util as util
import optimal_stopping_pomdp.pomdp_solvers.sondik_vi as sondik_vi
import numpy as np

if __name__ == '__main__':
    L=3
    R_INT = -5
    R_COST = -5
    R_SLA = 1
    R_ST = 5
    p=0.2
    num_observations = 10
    b1 = np.array([1,0,0])
    util.set_seed(1521245)

    Z = EnvUtil.observation_tensor(num_observations)
    T = EnvUtil.transition_tensor_single(l=L, p=p)
    R = EnvUtil.reward_tensor_single(R_SLA=R_SLA, R_INT=R_INT, R_ST=R_ST, L=L, R_COST=R_COST)
    S = EnvUtil.state_space()

    times = []
    values = []
    start = time.time()
    l=1
    while l <= L:
        print(f"l: {l}")
        Z = EnvUtil.observation_tensor(num_observations)[:,0,:,:]
        T = EnvUtil.transition_tensor_single(l=L, p=p)
        R = EnvUtil.reward_tensor_single(R_SLA=R_SLA, R_INT=R_INT, R_ST=R_ST, L=L, R_COST=R_COST)
        S = EnvUtil.state_space()
        A = EnvUtil.defender_actions()
        O = EnvUtil.observation_space(num_observations)
        sondik_vi.vi(P=T, Z=Z, R=R, T=100, gamma=0.95, n_states=len(S), n_actions=len(A), n_obs=len(O), b0=b1,
                     use_pruning=True)
import time
from optimal_stopping_pomdp.optimal_stopping_env.env_util import EnvUtil
import optimal_stopping_pomdp.util.util as util
import optimal_stopping_pomdp.pomdp_solvers.pomdp_hsvi as hsvi
import numpy as np

if __name__ == '__main__':
    env_name = "optimal-stopping-pomdp-v1"

    L=3
    R_INT = -5
    R_COST = -5
    R_SLA = 1
    R_ST = 5
    p=0.01
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

        with open(f'aleph_t_l_0.npy', 'wb') as f:
            np.save(f, np.asarray(list([list(np.zeros(len(S)))])))

        l_1_alpha_vectors = None
        with open(f'aleph_t_l_{l-1}.npy', 'rb') as f:
            l_1_alpha_vectors = np.load(f, allow_pickle=True)

        values, times = hsvi.hsvi(O=O,Z=Z,R=R,T=T,A=A,S=S,gamma=0.9, b0=b1, epsilon=0.01,
                             lp=True,prune_frequency=100, verbose=False,
                             number_of_simulations=100, l=l, max_exploration_depth=500,
                                  l_1_alpha_vectors=l_1_alpha_vectors, L=L,
                             start=start, values=values, times=times, p=p)
        print("values:")
        print(values)
        print("times:")
        print(times)
        l += 1
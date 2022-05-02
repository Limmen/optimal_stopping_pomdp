import random
from optimal_stopping_pomdp.optimal_stopping_env.optimal_stopping_env import OptimalStoppingEnv
from optimal_stopping_pomdp.dao.env_config import EnvConfig
from optimal_stopping_pomdp.optimal_stopping_env.env_util import EnvUtil
import numpy as np
import gym

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

    config = EnvConfig(
        env_name=env_name, A1=EnvUtil.attacker_actions(), A2=EnvUtil.defender_actions(), L=L, R_INT=R_INT,
        R_COST=R_COST, R_SLA=R_SLA, R_ST=R_ST, b1=b1, T=EnvUtil.transition_tensor(L=L, p=0.0),
        O=EnvUtil.observation_space(num_observations), Z=EnvUtil.observation_tensor(num_observations),
        R=EnvUtil.reward_tensor(R_SLA=R_SLA, R_INT=R_INT, R_COST=R_COST, L=L, R_ST=R_ST),
        S=EnvUtil.state_space(), p = p)
    env = gym.make(env_name, config=config)

    num_test_episodes = 10
    for i in range(num_test_episodes):
        o = env.reset()
        done = False
        t=1
        while not done:
            a1 = random.choice([0,1])
            o, r, done, info = env.step(a1)
            print(f"episode:{i}, t:{t}, o: {o}, r: {r}, done:{done}, info: {info}")
            t+=1


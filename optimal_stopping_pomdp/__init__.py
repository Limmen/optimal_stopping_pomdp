"""
Register OpenAI Envs
"""
import gym
from gym.envs.registration import register

register(
    id='optimal-stopping-pomdp-v1',
    entry_point='optimal_stopping_pomdp.optimal_stopping_env.optimal_stopping_env:OptimalStoppingEnv',
    kwargs={'config': None}
)
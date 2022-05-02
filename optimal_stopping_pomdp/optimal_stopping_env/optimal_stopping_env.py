from typing import Tuple, Dict, Union
import gym
import numpy as np
from optimal_stopping_pomdp.dao.env_config import EnvConfig
from optimal_stopping_pomdp.optimal_stopping_env.env_util import EnvUtil
from optimal_stopping_pomdp.dao.pomdp_state import POMDPState


class OptimalStoppingEnv(gym.Env):
    """
    OpenAI Gym Env for the POMDP of the defender when facing a static attacker
    (Hammar & Stadler 2022, https://arxiv.org/abs/2111.00289)
    """

    def __init__(self, config: EnvConfig):
        """
        Initializes the environment

        :param config: the environment configuration
        :param attacker_strategy: the strategy of the static attacker
        """

        self.config = config

        # Initialize environment state
        self.state = POMDPState(b1=self.config.b1, L=self.config.L)

        # Setup spaces
        self.observation_space = self.config.defender_observation_space()
        self.action_space = self.config.defender_action_space()

        # Setup Config
        self.viewer = None
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 50  # Video rendering speed
        }

        self.latest_attacker_obs = None
        # Reset
        self.reset()
        super().__init__()

    def step(self, a1 : int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Takes a step in the environment by executing the given action

        :param a1: the defender action
        :return: (obs, reward, done, info)
        """

        # Setup initial values
        done = False
        info = {}

        pi2 = EnvUtil.get_attacker_stage_strategy(s=self.state.s, config=self.config)

        # Compute r, s', b',o'
        a2 = EnvUtil.sample_attacker_action(pi2 = pi2, s=self.state.s)
        r = self.config.R[self.state.l - 1][a1][a2][self.state.s]
        self.state.s = EnvUtil.sample_next_state(l=self.state.l, a1=a1, a2=a2,
                                                          T=self.config.T,
                                                          S=self.config.S, s=self.state.s)
        o = max(self.config.O)
        if self.state.s == 2:
            done = True
        else:
            o = EnvUtil.sample_next_observation(Z=self.config.Z,
                                                         O=self.config.O, s_prime=self.state.s)
            self.state.b = EnvUtil.next_belief(o=o, a1=a1, b=self.state.b, pi2=pi2,
                                                        config=self.config,
                                                        l=self.state.l, a2=a2)

        # Update stops remaining
        self.state.l = self.state.l-a1

        # Populate info dict
        info["l"] = self.state.l
        info["s"] = self.state.s
        info["a1"] = a1
        info["a2"] = a2
        info["o"] = o

        # Get observations
        defender_obs = self.state.defender_observation()

        return defender_obs, r, done, info

    def reset(self, soft : bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resets the environment state, this should be called whenever step() returns <done>

        :return: initial observation
        """
        self.state.reset()
        defender_obs = self.state.defender_observation()
        return defender_obs

    def render(self, mode: str = 'human'):
        """
        Renders the environment
        Supported rendering modes:
          -human: render to the current display or terminal and return nothing. Usually for human consumption.
          -rgb_array: Return an numpy.ndarray with shape (x, y, 3),
                      representing RGB values for an x-by-y pixel image, suitable
                      for turning into a video.
        :param mode: the rendering mode
        :return: True (if human mode) otherwise an rgb array
        """
        raise NotImplemented("Rendering is not implemented for this environment")


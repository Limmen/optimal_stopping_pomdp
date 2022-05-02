from optimal_stopping_pomdp.optimal_stopping_env.optimal_stopping_env import OptimalStoppingEnv
from optimal_stopping_pomdp.dao.env_config import EnvConfig
from optimal_stopping_pomdp.optimal_stopping_env.env_util import EnvUtil
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import optimal_stopping_pomdp.util.util as util
import numpy as np
import gym
import time
import os


class PPOTrainingCallback(BaseCallback):
    """
    Callback for monitoring PPO training
    """
    def __init__(self, max_steps: int,
                 verbose=0,
                 eval_every: int = 100,
                 eval_batch_size: int = 10, save_every: int = 10, save_dir: str = ""):
        """

        :param max_steps:
        :param verbose:
        :param eval_every:
        :param eval_batch_size:
        :param save_every:
        :param save_dir:
        """
        super(PPOTrainingCallback, self).__init__(verbose)
        self.iter = 0
        self.eval_every = eval_every
        self.eval_batch_size = eval_batch_size
        self.max_steps = max_steps
        self.save_every = save_every
        self.save_dir = save_dir

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        ts = time.time()
        save_path = self.save_dir + f"/ppo_model{self.iter}_{ts}.zip"

        # Save model
        if self.iter % self.save_every == 0 and self.iter > 0:
            self.model.save(save_path)
            os.chmod(save_path, 0o777)

        # Eval model
        if self.iter % self.eval_every == 0:
            ts = time.time()
            o = self.training_env.reset()
            max_horizon = 200
            avg_rewards = []
            for i in range(self.eval_batch_size):
                done = False
                t = 0
                cumulative_reward = 0
                while not done and t <= max_horizon:
                    a, _ = self.model.predict(np.array(o), deterministic=False)
                    o, r, done, info = self.training_env.step(a)
                    cumulative_reward +=r
                    t+= 1
                avg_rewards.append(cumulative_reward)
            avg_R = np.mean(avg_rewards)
            print(f"[EVAL] Training iteration: {self.iter}, Average R:{avg_R}")
            self.training_env.reset()

        self.iter += 1


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

    env = Monitor(env)

    # Hparams
    num_neurons_per_hidden_layer = 64
    num_layers = 4
    policy_kwargs = dict(net_arch=[num_neurons_per_hidden_layer]*num_layers)
    steps_between_updates =1000
    learning_rate = 0.0005
    batch_size = 100
    device = "cpu"
    gamma = 0
    num_training_timesteps = int(1e6)
    verbose = 0

    # Set seed for reproducibility
    seed = 999
    util.set_seed(seed)
    cb = PPOTrainingCallback(eval_every=10, max_steps=500, verbose=1, eval_batch_size=10, save_every=1000,
                             save_dir="")

    # Train
    model = PPO("MlpPolicy", env, verbose=verbose,
                policy_kwargs=policy_kwargs, n_steps=steps_between_updates,
                batch_size=batch_size, learning_rate=learning_rate, seed=seed,
                device="cpu", gamma=gamma)
    model.learn(total_timesteps=num_training_timesteps, callback=cb)


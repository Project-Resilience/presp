"""
Evaluator for the CartPole gymnasium environment.
"""
import gymnasium
import numpy as np
import torch

from examples.cartpole.prescriptor import CartPolePrescriptor
from presp.evaluator import Evaluator


class CartPoleEvaluator(Evaluator):
    """
    Evaluator implementation for the CartPole gymnasium environment.
    We perform direct evolution so we do not implement update_predictor.
    evaluate_candidate simply runs the candidate prescriptor for n_steps in n_envs environments and returns the average.
    """
    def __init__(self, n_jobs: int, n_envs: int):
        super().__init__(outcomes=["score"], n_jobs=n_jobs)
        self.n_envs = n_envs

    def update_predictor(self, elites):
        pass

    def evaluate_candidate(self, candidate: CartPolePrescriptor):
        total_reward = 0
        for i in range(self.n_envs):
            env = gymnasium.make("CartPole-v1")
            obs, _ = env.reset(seed=i)
            episode_over = False
            while not episode_over:
                action = candidate.forward(torch.tensor(obs, dtype=torch.float32, device=candidate.device))
                obs, reward, done, truncated, _ = env.step(action.item())
                episode_over = done or truncated
                total_reward += reward
            env.close()
        return np.array([-1 * total_reward / self.n_envs])

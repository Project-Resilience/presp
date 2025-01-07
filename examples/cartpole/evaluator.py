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
    def __init__(self, n_steps: int, n_envs: int):
        super().__init__(outcomes=["score"])
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.env = gymnasium.make_vec("CartPole-v1", sutton_barto_reward=True, num_envs=n_envs)

    def update_predictor(self, elites):
        pass

    def evaluate_candidate(self, candidate: CartPolePrescriptor):
        total_rewards = np.zeros(self.n_envs)
        obs, _ = self.env.reset()
        for _ in range(self.n_steps):
            action = candidate.forward(torch.tensor(obs, dtype=torch.float32, device=candidate.device))
            obs, reward, _, _, _ = self.env.step(action)
            total_rewards += reward
        return np.array([-1 * (total_rewards / self.n_steps).mean()])

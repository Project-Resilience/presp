"""
Example implementation of a custom prescriptor. We wrap around the basic NNPrescriptor torch model to handle the
CartPole environment's observations.
"""
import numpy as np
import torch

from presp.prescriptor import NNPrescriptor


class CartPolePrescriptor(NNPrescriptor):
    """
    Wrapper around the NNPrescriptor to handle the CartPole environment (or just gym envs in general).
    We take in a numpy array of observation and return an action.
    This implementation assumes we only get 1 obs at a time which may not always be the case.
    """
    def forward(self, context: np.ndarray) -> torch.Tensor:
        context_tensor = torch.tensor(context, dtype=torch.float32, device=self.device)
        prob = super().forward(context_tensor)
        action = (prob > 0.5).int().item()
        return action

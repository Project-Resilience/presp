"""
A simple implementation of the NNPrescriptor used for the CartPole gymnasium environment.
"""
import torch

from presp.prescriptor import NNPrescriptor, NNPrescriptorFactory


class CartPolePrescriptor(NNPrescriptor):
    """
    The prescriptor for the CartPole gymnasium environment.
    Context is taken in as a torch tensor and passed through the model. Then the actions are reshaped according to how
    the gymnasium environment expects them.
    """
    def __init__(self, model_params: dict, device: str = "cpu"):
        super().__init__(model_params, device)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        logits = self.model(context)
        actions = (torch.sigmoid(logits) > 0.5).squeeze().int().cpu().numpy()
        return actions


class CartPolePrescriptorFactory(NNPrescriptorFactory):
    """
    CartPolePrescriptor factory. Hard-codes the model parameters since we're using a gym environment with a fixed
    observation/action space.
    """
    def __init__(self, device: str = "cpu"):
        model_params = {"in_size": 4, "hidden_size": 16, "out_size": 1}
        super().__init__(CartPolePrescriptor, model_params, device)

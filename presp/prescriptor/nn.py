"""
Implementation of Prescriptor using a simple 1 hidden layer neural network with Tanh activation.
"""
import copy
from pathlib import Path
from typing import Type

import torch

from presp.prescriptor.base import Prescriptor, PrescriptorFactory


class NNPrescriptor(Prescriptor):
    """
    Simple neural network implementation of a prescriptor.
    """
    def __init__(self, model_params: dict[str, int], device: str = "cpu"):
        super().__init__()
        self.model_params = model_params
        self.device = device

        self.model = torch.nn.Sequential(
            torch.nn.Linear(model_params["in_size"], model_params["hidden_size"]),
            torch.nn.Tanh(),
            torch.nn.Linear(model_params["hidden_size"], model_params["out_size"])
        )
        self.model.to(device)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.model(context)

    def save(self, path: Path):
        torch.save(self.model.state_dict(), path)


class NNPrescriptorFactory(PrescriptorFactory):
    """
    Factory to construct NNPrescriptors.
    TODO: Since we pass prescriptor_cls into super().__init__, pylance loses knowledge that prescriptor_cls should have
    a .model attribute...
    """
    def __init__(self, prescriptor_cls: Type[NNPrescriptor], model_params: dict[str, int], device: str = "cpu"):
        super().__init__(prescriptor_cls)
        self.model_params = model_params
        self.device = device

    def random_init(self) -> NNPrescriptor:
        """
        Orthogonally initializes a neural network.
        We orthogonally init all the linear layers except the last one which is standard normally initialized.
        Fill the biases with the standard normal distribution as well.
        """
        candidate = self.prescriptor_cls(self.model_params, self.device)
        for i, layer in enumerate(candidate.model):
            if isinstance(layer, torch.nn.Linear):
                # TODO: This relies on the model ending with a linear layer which we may not guarantee later!
                if i != len(candidate.model) - 1:
                    torch.nn.init.orthogonal_(layer.weight)
                else:
                    torch.nn.init.normal_(layer.weight, 0, 1)
                layer.bias.data.fill_(torch.normal(0, 1, size=(1,)).item())

        return candidate

    def crossover(self,
                  parents: list[NNPrescriptor],
                  mutation_rate: float,
                  mutation_factor: float) -> list[NNPrescriptor]:
        """
        Crossover two parents to create a child.
        Take a random 50/50 choice of either parent's weights.
        Then mutates the child.
        NOTE: The child is returned in a list to fit the abstract crossover method.
        """
        child = self.prescriptor_cls(self.model_params, self.device)
        parent1, parent2 = parents[0], parents[1]
        child.model = copy.deepcopy(parent1.model)
        for child_param, parent2_param in zip(child.model.parameters(), parent2.model.parameters()):
            mask = torch.rand(size=child_param.data.shape, device=self.device) < 0.5
            child_param.data[mask] = parent2_param.data[mask]
        self.mutate_(child, mutation_rate, mutation_factor)
        return [child]

    def mutate_(self, candidate: NNPrescriptor, mutation_rate: float, mutation_factor: float):
        """
        Mutates a prescriptor in-place with gaussian percent noise.
        """
        with torch.no_grad():
            for param in candidate.model.parameters():
                mutate_mask = torch.rand(param.shape, device=param.device) < mutation_rate
                noise = torch.normal(0,
                                     mutation_factor,
                                     param[mutate_mask].shape,
                                     device=param.device,
                                     dtype=param.dtype)
                param[mutate_mask] *= (1 + noise)

    def load(self, path: Path) -> Prescriptor:
        """
        Loads torch model from file.
        """
        candidate = self.prescriptor_cls(self.model_params, device=self.device)
        candidate.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        return candidate

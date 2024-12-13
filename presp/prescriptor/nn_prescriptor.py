"""
Implementation of Prescriptor using a simple 1 hidden layer neural network with Tanh activation.
"""
import copy
from pathlib import Path

from prescriptor import Prescriptor, PrescriptorFactory

import torch


class NNPrescriptor(Prescriptor):
    """
    Simple neural network implementation of a prescriptor.
    """
    def __init__(self, model_params: dict[str, int], device: str):
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
    Factory to construct NNPrescriptors
    """
    def __init__(self, model_params: dict[str, int], device: str):
        self.model_params = model_params
        self.device = device

    def random_init(self) -> NNPrescriptor:
        """
        Orthogonally initializes a neural network.
        """
        candidate = NNPrescriptor(self.model_params, self.device)
        for layer in candidate.model:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight)
                layer.bias.data.fill_(0.01)

        return candidate

    def crossover(self, parents: list[Prescriptor], mutation_rate: float, mutation_factor: float) -> list[Prescriptor]:
        """
        Crossover two parents to create a child.
        Take a random 50/50 choice of either parent's weights.
        Then mutates the child.
        NOTE: The child is returned in a list to fit the abstract crossover method.
        """
        child = NNPrescriptor(self.model_params, self.device)
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
                param[mutate_mask] += noise * param[mutate_mask]

    def load(self, path: Path) -> Prescriptor:
        """
        Loads torch model from file.
        """
        candidate = NNPrescriptor(self.model_params, device=self.device)
        candidate.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        return candidate

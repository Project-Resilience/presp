"""
Implementation of Prescriptor using a simple 1 hidden layer neural network with Tanh activation.
"""
import copy
from pathlib import Path

import torch

from presp.prescriptor.base import Prescriptor, PrescriptorFactory


class NNPrescriptor(Prescriptor):
    """
    Simple neural network implementation of a prescriptor.
    """
    def __init__(self, model_params: list[dict], device: str = "cpu"):
        super().__init__()
        self.model_params = [{**layer} for layer in model_params]
        self.device = device

        # Construct the model
        layers = []
        for layer in self.model_params:
            layer_type = layer.pop("type")
            if layer_type == "linear":
                layers.append(torch.nn.Linear(**layer))
            elif layer_type == "tanh":
                layers.append(torch.nn.Tanh(**layer))
            elif layer_type == "sigmoid":
                layers.append(torch.nn.Sigmoid(**layer))
            elif layer_type == "softmax":
                layers.append(torch.nn.Softmax(**layer))
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

        # Get our model params back after we destroy them building the model
        self.model_params = [{**layer} for layer in model_params]
        self.model = torch.nn.Sequential(*layers)
        self.model.to(device)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        We simply pass the context through our model here. The user should get exactly out of the prescriptor what
        they created using the config. They can process the output as they see fit.
        """
        output = self.model(context)
        return output


class NNPrescriptorFactory(PrescriptorFactory):
    """
    Factory to construct NNPrescriptors.
    """
    def __init__(self, model_params: dict[str, int], device: str = "cpu"):
        self.model_params = model_params
        self.device = device

    def random_init(self) -> NNPrescriptor:
        """
        Orthogonally initializes a neural network.
        We orthogonally init all the linear layers except the last one which is standard normally initialized.
        Fill the biases with the standard normal distribution as well.
        """
        candidate = NNPrescriptor(self.model_params, self.device)
        linear_layers = [layer for layer in candidate.model if isinstance(layer, torch.nn.Linear)]
        for i, layer in enumerate(linear_layers):
            if isinstance(layer, torch.nn.Linear):
                if i != len(linear_layers) - 1:
                    torch.nn.init.orthogonal_(layer.weight)
                else:
                    torch.nn.init.normal_(layer.weight, 0, 1)
                layer.bias.data.fill_(torch.normal(0, 1, size=(1,)).item())

        return candidate

    def crossover(self, parents: list[NNPrescriptor]) -> list[NNPrescriptor]:
        """
        Crossover two parents to create a child.
        Take a random 50/50 choice of either parent's weights.
        NOTE: The child is returned in a list to fit the abstract crossover method.
        """
        child = NNPrescriptor(self.model_params, self.device)
        parent1, parent2 = parents[0], parents[1]
        child.model = copy.deepcopy(parent1.model)
        for child_param, parent2_param in zip(child.model.parameters(), parent2.model.parameters()):
            mask = torch.rand(size=child_param.data.shape, device=self.device) < 0.5
            child_param.data[mask] = parent2_param.data[mask]
        return [child]

    def mutation(self, candidate: NNPrescriptor, mutation_rate: float, mutation_factor: float):
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

    def save(self, candidate: NNPrescriptor, path: Path):
        """
        Saves the torch model to file.
        """
        torch.save(candidate.model.state_dict(), path)

    def load(self, path: Path) -> Prescriptor:
        """
        Loads torch model from file.
        """
        candidate = NNPrescriptor(self.model_params, device=self.device)
        candidate.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        return candidate

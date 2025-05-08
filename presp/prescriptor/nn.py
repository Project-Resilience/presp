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
        """
        :param model_params: The parameters to construct the torch model with. A list of layers with their type and
        parameters. See below for how to use this.
        :param device: The torch device to load the model on to.
        """
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
            elif layer_type == "relu":
                layers.append(torch.nn.ReLU(**layer))
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
    def __init__(self,
                 prescriptor_cls: type[NNPrescriptor],
                 model_params: list[dict],
                 device: str = "cpu",
                 **kw_params):
        """
        :param prescriptor_cls: The class of the prescriptor to create. There can be a many:1 mapping between
        NNPrescriptor implementations and NNPrescriptorFactory so we can use the same factory for different
        prescriptors meaning we should take in their CLS here.
        :param model_params: The parameters to construct the NNPrescriptors with.
        :param device: What torch.device to load the models on to by default.
        :param kw_params: Any other parameters to pass to the prescriptor.
        """
        self.prescriptor_cls = prescriptor_cls
        self.model_params = model_params
        self.device = device
        self.kw_params = kw_params

    def random_init(self) -> NNPrescriptor:
        """
        Orthogonally initializes a neural network.
        We orthogonally init all the linear layers except the last one which is standard normally initialized.
        Fill the biases with the standard normal distribution as well.
        """
        candidate = self.prescriptor_cls(model_params=self.model_params, device=self.device, **self.kw_params)

        # NOTE: We need to move the model to CPU to init the weights, otherwise we get an error on MPS
        if candidate.device == "mps":
            candidate.model = candidate.model.to("cpu")

        linear_layers = [layer for layer in candidate.model if isinstance(layer, torch.nn.Linear)]
        for i, layer in enumerate(linear_layers):
            if isinstance(layer, torch.nn.Linear):
                if i != len(linear_layers) - 1:
                    torch.nn.init.orthogonal_(layer.weight)
                else:
                    torch.nn.init.normal_(layer.weight, 0, 1)
                layer.bias.data.fill_(torch.normal(0, 1, size=(1,)).item())

        # Return the weights back to the original device
        if candidate.device == "mps":
            candidate.model = candidate.model.to(candidate.device)

        return candidate

    def crossover(self, parents: list[NNPrescriptor]) -> list[NNPrescriptor]:
        """
        Crossover two parents to create a child.
        Take a random 50/50 choice of either parent's weights.
        NOTE: The child is returned in a list to fit the abstract crossover method.
        """
        child = self.prescriptor_cls(model_params=self.model_params, device=self.device, **self.kw_params)
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

    def save_population(self, population: list[NNPrescriptor], path: Path):
        """
        Put population into dict of cand_id -> NNPrescriptor and then save as a torch file.
        """
        pop_dict = {cand.cand_id: cand.model.state_dict() for cand in population}
        torch.save(pop_dict, path)

    def load_population(self, path: Path) -> dict[str, NNPrescriptor]:
        """
        Loads dict of torch weights from file and reconstructs the population into NNPrescriptors.
        """
        pop_dict = torch.load(path, map_location=self.device)
        population = {}
        for cand_id, model_state in pop_dict.items():
            candidate = self.prescriptor_cls(model_params=self.model_params, device=self.device, **self.kw_params)
            candidate.cand_id = cand_id
            candidate.model.load_state_dict(model_state)
            population[cand_id] = candidate
        return population

"""
TODO: Can we combine these into a single class that the user has to implement?
"""
from abc import ABC, abstractmethod
from pathlib import Path


class Prescriptor(ABC):
    """
    Prescriptor class that goes from context to actions.
    """
    def __init__(self):
        self.cand_id = ""
        self.parents = None
        self.metrics = None
        self.outcomes = None
        self.rank = None
        self.distance = None

    @abstractmethod
    def forward(self, context):
        """
        Generates actions from context.
        TODO: Is there a nicer way to have this be extensible?
        :param context: The input context to generate actions from. Currently this can be any type but maybe we should
        restrict this later.
        :return: TODO: Currently the return type isn't specified either. Perhaps this should be a numpy array?
        """


class PrescriptorFactory(ABC):
    """
    Interface in charge of creating prescriptors.
    Prescriptors should be able to be randomly initialized at the start of evolution, crossed over and mutated,
    and saved and loaded.
    """
    @abstractmethod
    def random_init(self) -> Prescriptor:
        """
        Creates a randomly initialized prescriptor model.
        """

    @abstractmethod
    def crossover(self, parents: list[Prescriptor]) -> list[Prescriptor]:
        """
        Crosses over N parents to make M children.
        :param parents: The list of parents to use in the crossover.
        """

    @abstractmethod
    def mutation(self, candidate: Prescriptor, mutation_rate: float, mutation_factor: float):
        """
        Mutates a prescriptor in-place.
        :param candidate: The candidate to mutate.
        :param mutation_rate: The rate at which to mutate each parameter.
        :param mutation_factor: The factor by which to mutate each parameter.
        """

    @abstractmethod
    def save(self, candidate: Prescriptor, path: Path):
        """
        Save the prescriptor to file.
        :param prescriptor: The prescriptor to save.
        :param path: The path to save the prescriptor to.
        """

    @abstractmethod
    def load(self, path: Path) -> Prescriptor:
        """
        Load a prescriptor from file.
        :param path: The path to load the prescriptor from.
        """

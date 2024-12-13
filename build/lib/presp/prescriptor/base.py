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
        :return: Currently the return type isn't specified either. Perhaps this should be a numpy array?
        """

    @abstractmethod
    def save(self, path: Path):
        """
        Save the prescriptor to file.
        :param path: The path to save the prescriptor to.
        """


class PrescriptorFactory(ABC):
    """
    Abstract class in charge of creating prescriptors.
    Implementations should store details used to create prescriptors.
    """
    @abstractmethod
    def random_init(self) -> Prescriptor:
        """
        Creates a randomly initialized prescriptor model.
        """

    @abstractmethod
    def crossover(self, parents: list[Prescriptor], mutation_rate: float, mutation_factor: float) -> list[Prescriptor]:
        """
        Crosses over N parents to make N children. Mutates the N children.
        TODO: Should we separate mutation out of this?
        :param parents: The list of parents to use in the crossover.
        :param mutation_rate: The rate at which to mutate each parameter.
        :param mutation_factor: The factor by which to mutate each parameter.
        """

    @abstractmethod
    def load(self, path: Path) -> Prescriptor:
        """
        Load a prescriptor from file.
        :param path: The path to load the prescriptor from.
        """

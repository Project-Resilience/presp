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
        """

    @abstractmethod
    def save(self, path: Path):
        """
        Save the prescriptor to file.
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
        """

    @abstractmethod
    def load(self, path: Path) -> Prescriptor:
        """
        Load a prescriptor from file.
        """

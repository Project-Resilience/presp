"""
A dummy prescriptor and factory for use in unit testing.
"""
from typing import Type

import numpy as np

from presp.prescriptor import Prescriptor, PrescriptorFactory


class DummyPrescriptor(Prescriptor):
    """
    A dummy prescriptor for testing purposes.
    This prescriptor stores a random float which it always returns as a prescription.
    """
    def __init__(self):
        super().__init__()
        self.number = np.random.rand(1)
        self.parents = []

    def forward(self, context):
        return self.number

    def save(self, path):
        with open(path, "w", encoding="utf-8") as file:
            file.write(str(self.number))


class DummyFactory(PrescriptorFactory):
    """
    A dummy factory for testing purposes.
    Returns the average of 2 numbers for crossover, adds some random gaussian noise for mutation.
    """
    def __init__(self, prescriptor_cls: Type[DummyPrescriptor]):
        super().__init__(prescriptor_cls)

    def random_init(self) -> DummyPrescriptor:
        return DummyPrescriptor()

    def crossover(self,
                  parents: list[DummyPrescriptor],
                  mutation_rate: float,
                  mutation_factor: float) -> list[DummyPrescriptor]:
        numb = np.mean([parent.number for parent in parents], axis=0)
        if np.random.rand(1) < mutation_rate:
            numb += np.random.normal(0, mutation_factor)
        child = DummyPrescriptor()
        # To keep track of ancestry for testing purposes
        child.parents = [parent.cand_id for parent in parents]
        child.number = numb
        return [child]

    def load(self, path) -> DummyPrescriptor:
        with open(path, "r", encoding="utf-8") as file:
            number = float(file.read())
        candidate = DummyPrescriptor()
        candidate.number = number
        return candidate

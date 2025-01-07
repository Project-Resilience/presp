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

    def forward(self, context):
        return self.number

    def save(self, path):
        pass


class DummyFactory(PrescriptorFactory):
    """
    A dummy factory for testing purposes.
    Returns the average of 2 numbers for crossover, adds some random gaussian noise for mutation.
    """
    def __init__(self, prescriptor_cls: Type[DummyPrescriptor]):
        super().__init__(prescriptor_cls)

    def random_init(self) -> DummyPrescriptor:
        return DummyPrescriptor()

    def crossover(self, parents: list[Prescriptor], mutation_rate: float, mutation_factor: float) -> list[Prescriptor]:
        numb = np.mean([parent.number for parent in parents], axis=0)
        if np.random.rand(1) < mutation_rate:
            numb += np.random.normal(0, mutation_factor)
        child = DummyPrescriptor()
        child.number = numb
        return [child]

    def load(self, path) -> Prescriptor:
        pass

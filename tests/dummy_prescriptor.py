"""
A dummy prescriptor and factory for use in unit testing.
"""
from pathlib import Path

import numpy as np

from presp.prescriptor import Prescriptor, PrescriptorFactory


class DummyPrescriptor(Prescriptor):
    """
    A dummy prescriptor for testing purposes.
    This prescriptor stores a random float which it always returns as a prescription.
    """
    def __init__(self):
        super().__init__()
        self.number = np.random.rand()
        self.parents = []

    def forward(self, context):
        return self.number


class DummyFactory(PrescriptorFactory):
    """
    A dummy factory for testing purposes.
    Returns the average of 2 numbers for crossover, adds some random gaussian noise for mutation.
    """
    def random_init(self) -> DummyPrescriptor:
        return DummyPrescriptor()

    def crossover(self, parents: list[DummyPrescriptor]) -> list[DummyPrescriptor]:
        numb = np.mean([parent.number for parent in parents], axis=0)
        child = DummyPrescriptor()
        # To keep track of ancestry for testing purposes
        child.parents = [parent.cand_id for parent in parents]
        child.number = numb
        return [child]

    def mutation(self, candidate: DummyPrescriptor, mutation_rate: float, mutation_factor: float):
        if np.random.rand() < mutation_rate:
            candidate.number += np.random.normal(0, mutation_factor)

    def save(self, candidate: DummyPrescriptor, path: Path):
        with open(path, "w", encoding="utf-8") as file:
            file.write(str(candidate.number))

    def load(self, path) -> DummyPrescriptor:
        with open(path, "r", encoding="utf-8") as file:
            number = float(file.read())
        candidate = DummyPrescriptor()
        candidate.number = number
        return candidate

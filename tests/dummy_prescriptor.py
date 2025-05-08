"""
A dummy prescriptor and factory for use in unit testing.
"""
from pathlib import Path

import numpy as np
import pandas as pd

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

    def save_population(self, population: list[DummyPrescriptor], path: Path):
        cand_ids = [cand.cand_id for cand in population]
        numbers = [cand.number for cand in population]
        df = pd.DataFrame({"cand_id": cand_ids, "number": numbers})
        df.to_csv(path, index=False)

    def load_population(self, path: Path) -> dict[str, DummyPrescriptor]:
        df = pd.read_csv(path)
        pop_dict = {}
        for _, row in df.iterrows():
            cand = DummyPrescriptor()
            cand.cand_id = row["cand_id"]
            cand.number = row["number"]
            pop_dict[cand.cand_id] = cand
        return pop_dict

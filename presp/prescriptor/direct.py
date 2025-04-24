"""
Direct representation of actions as a vector of floats from xl to xu.
"""
from pathlib import Path

import numpy as np

from presp.prescriptor import Prescriptor, PrescriptorFactory


class DirectPrescriptor(Prescriptor):
    """
    Direct evolution candidate. Simply a vector of floats between xl and xu.
    """
    def __init__(self, xl: np.ndarray, xu: np.ndarray):
        super().__init__()
        self.genome = np.mean(np.stack([xl, xu]), axis=0)

    def forward(self, _) -> np.ndarray:
        """
        Just straightforwardly returns the genome.
        """
        return self.genome


class DirectFactory(PrescriptorFactory):
    """
    Prescriptor factory handling the construction of direct evolution candidates.
    TODO: We do an extra operation here by creating 2 genomes then setting one to the other. Is there a cleaner
    way to do this?
    """
    def __init__(self, xl: np.ndarray, xu: np.ndarray):
        self.xl = xl
        self.xu = xu

    def random_init(self) -> DirectPrescriptor:
        """
        Creates a randomly initialized vector of floats between xl and xu uniformly.
        """
        genome = np.random.rand(*self.xl.shape)
        genome = genome * (self.xu - self.xl) + self.xl
        candidate = DirectPrescriptor(self.xl, self.xu)
        candidate.genome = genome
        return candidate

    def crossover(self, parents: list[DirectPrescriptor]) -> list[DirectPrescriptor]:
        """
        Crosses over 2 parents using uniform crossover to create a single child.
        """
        parent1 = parents[0].genome
        parent2 = parents[1].genome
        child_genome = np.where(np.random.rand(*parent1.shape) < 0.5, parent1, parent2)
        child = DirectPrescriptor(self.xl, self.xu)
        child.genome = child_genome
        return [child]

    def mutation(self, candidate: DirectPrescriptor, mutation_rate: float, mutation_factor: float):
        """
        Mutates the genome of the given candidate in place.
        When the mutation causes a parameter to go out of bounds, mirror it back into bounds.
            For example: mutating 0.9 to 1.2 would cause it to become 0.8
        """
        genome = candidate.genome
        mutate_mask = np.random.rand(*genome.shape) < mutation_rate
        noise = np.random.normal(0, mutation_factor, genome[mutate_mask].shape)
        genome[mutate_mask] *= (1 + noise)

        # Mirror genome back into bounds.
        genome = np.where(genome < self.xl, 2*self.xl-genome, genome)
        genome = np.where(genome > self.xu, 2*self.xu-genome, genome)

        # Is this necessary?
        candidate.genome = genome

    def save(self, candidate: DirectPrescriptor, path: Path):
        with open(path, "wb") as f:
            # Save the genome as a tensor
            np.save(f, candidate.genome)

    def load(self, path: Path) -> DirectPrescriptor:
        with open(path, "rb") as f:
            # Load the genome from a tensor
            genome = np.load(f)
        candidate = DirectPrescriptor(self.xl, self.xu)
        candidate.genome = genome
        return candidate

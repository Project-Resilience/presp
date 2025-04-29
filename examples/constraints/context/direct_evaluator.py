import numpy as np

from presp.evaluator import Evaluator
from presp.prescriptor import DirectPrescriptor


class DirectEvaluator(Evaluator):
    def __init__(self, n_jobs: int = 1):
        super().__init__(["f1", "f2"], n_jobs=n_jobs)

    def update_predictor(self, _):
        pass

    def f1(self, x1: float) -> float:
        return x1

    def f2(self, x2: float) -> float:
        return x2

    def c1(self, x1: float, x2: float, c: float) -> float:
        return -1 * (x1**2 + x2**2 - c**2)

    def c2(self, x1: float) -> float:
        return -1 * x1

    def c3(self, x2: float) -> float:
        return -1 * x2

    def evaluate_candidate(self, candidate: DirectPrescriptor) -> tuple[float, float]:
        x1 = candidate.genome[0]
        x2 = candidate.genome[1]

        f1 = self.f1(x1)
        f2 = self.f2(x2)

        g = np.array([self.c1(x1, x2, 10), self.c2(x1), self.c3(x2)])
        g = np.maximum(g, 0)
        g = np.sum(g)

        return np.array([f1, f2]), g

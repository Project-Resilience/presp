import numpy as np

from presp.evaluator import Evaluator
from presp.prescriptor import DirectPrescriptor


class BNHEvaluator(Evaluator):

    def __init__(self):
        super().__init__(["f1", "f2"], n_jobs=1)

    def update_predictor(self, _):
        pass

    def f1(self, x1: float, x2: float) -> float:
        return 4 * x1 ** 2 + 4 * x2 ** 2

    def f2(self, x1: float, x2: float) -> float:
        return (x1 - 5) ** 2 + (x2 - 5) ** 2

    def c1(self, x1: float, x2: float) -> float:
        return (x1 - 5) ** 2 + x2 ** 2 - 25

    def c2(self, x1: float, x2: float) -> float:
        return -1 * ((x1 - 8) ** 2 + (x2 + 3) ** 2 - 7.7)

    def c3(self, x1: float):
        return x1 - 5

    def c4(self, x1: float):
        return -1 * x1

    def c5(self, x2: float):
        return x2 - 3

    def c6(self, x2: float):
        return -1 * x2

    def evaluate_candidate(self, candidate: DirectPrescriptor):
        x1 = candidate.genome[0]
        x2 = candidate.genome[1]

        # Objectives
        f1 = self.f1(x1, x2)
        f2 = self.f2(x1, x2)
        f = np.array([f1, f2])

        # Constraints
        c1 = self.c1(x1, x2)
        c2 = self.c2(x1, x2)
        c3 = self.c3(x1)
        c4 = self.c4(x1)
        c5 = self.c5(x2)
        c6 = self.c6(x2)

        g = np.array([c1, c2, c3, c4, c5, c6])
        g = np.sum(np.maximum(0, g))

        return f, g

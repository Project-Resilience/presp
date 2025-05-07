from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from examples.constraints.context.circle_evaluator import CircleEvaluator
from examples.constraints.context.triangle_evaluator import TriangleEvaluator
from presp.prescriptor import NNPrescriptor, NNPrescriptorFactory

class Experimenter:
    def __init__(self, results_dir: Path):
        with open(results_dir / "config.yml", "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        if self.config["problem"] == "triangle":
            self.evaluator = TriangleEvaluator(**self.config["eval_params"])
        elif self.config["problem"] == "circle":
            self.evaluator = CircleEvaluator(**self.config["eval_params"])
        else:
            raise ValueError("Problem not supported")
        self.factory = NNPrescriptorFactory(NNPrescriptor, **self.config["prescriptor_params"])
        self.results_dir = results_dir

    def get_gen_df(self, gen: int, pareto: bool = True) -> pd.DataFrame:
        gen_df = pd.read_csv(self.results_dir / f"{gen}.csv")
        if pareto:
            gen_df = gen_df[gen_df["rank"] == 1]
        return gen_df

    def get_candidate_solutions(self, cand_id: str, test: bool = False) -> torch.Tensor:
        candidate = self.factory.load(self.results_dir / f"{cand_id.split('_')[0]}/{cand_id}")
        actions = self.evaluator.prescribe(candidate, test=test)
        return actions.cpu().numpy()

    def compute_hypervolume_2d(self, pareto_front: np.ndarray, ref_point: np.ndarray) -> float:
        """
        Assumes minimization.
        Pareto front is n x 2
        """
        hypervolume = 0
        pareto_front = np.sort(pareto_front, axis=0)
        for i in range(pareto_front.shape[0]):
            x1 = pareto_front[i, 0]
            x2 = pareto_front[i+1, 0] if i < pareto_front.shape[0] - 1 else ref_point[0]

            y1 = pareto_front[i, 1]
            y2 = ref_point[1]

            volume = (x2 - x1) * (y2 - y1)
            volume = max(volume, 0)
            hypervolume += volume

        return hypervolume

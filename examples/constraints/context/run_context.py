from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from presp.evolution import Evolution
from presp.prescriptor import DirectFactory, NNPrescriptorFactory, NNPrescriptor
from examples.constraints.context.context_evaluator import ContextEvaluator
from examples.constraints.context.direct_evaluator import DirectEvaluator


def run():
    with open("examples/constraints/context/config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    evaluator = ContextEvaluator(**config["eval_params"])
    factory = NNPrescriptorFactory(NNPrescriptor, **config["prescriptor_params"])

    save_path = Path(config["evolution_params"]["save_path"])
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "config.yml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    evolution = Evolution(**config["evolution_params"], evaluator=evaluator, prescriptor_factory=factory)
    evolution.run_evolution()



if __name__ == "__main__":
    run()

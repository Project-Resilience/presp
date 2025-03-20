# presp
Open-source ESP implementation for Project Resilience

## Evolution

The evolutionary step assumes it is given a sorted, evaluated population.
1. (Optional) Elites are taken from the population and used to update the evaluator (full ESP loop).
2. The next population is created using crossover/mutation of the current population.
3. The next population is evaluated and sorted.

## Usage
The `Evolution` class is the core of the library and takes in many evolution parameters as well as the `Evaluator` and the `PrescriptorFactory`.

3 things have to be implemented to use the library: the Evaluator, the Prescriptor, and the PrescriptorFactory.
- The `Prescriptor` is an individual in the population whose `forward` function must be implemented for use the in the `Evaluator`.
- The `PrescriptorFactory` must implement crossover/mutation for evolution.
- The `Evaluator` takes in a population of `Prescriptor`s and attaches metrics to each individual.

A basic example would look like this:
```python
from presp.evolution import Evolution

from examples.evaluator import ImplementedEvaluator
from examples.prescriptor import ImplementedFactory

config = {...}
evaluator = ImplementedEvaluator(**config["eval_params"])
prescriptor_factory = ImplementedFactory(**config["prescriptor_params"])

evolution = Evolution(**config["evolution_params"], evaluator=evaluator, prescriptor_factory=prescriptor_factory)

evolution.run_evolution()
```

## Example
For a more full-fledged example see `examples/cartpole` for a simple implementation of direct evolution (not ESP) on the CartPole gymnasium environment.
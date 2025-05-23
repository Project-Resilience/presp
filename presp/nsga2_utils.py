"""
Utilities for NSGA-II implementation.
"""
from presp.prescriptor import Prescriptor


# pylint: disable=invalid-name
def fast_non_dominated_sort(population: list[Prescriptor]) -> list[list[Prescriptor]]:
    """
    Fast non-dominated sort algorithm.
    Sets rank of candidates in-place.
    :param population: The population to sort.
    :return: A list of fronts.
    """
    population_size = len(population)
    S = [[] for _ in range(population_size)]    # Set of solutions dominated by p
    n = [0 for _ in range(population_size)]     # Number of solutions dominating p
    front = [[]]                                # Final Pareto fronts
    rank = [0 for _ in range(population_size)]  # Rank of solution

    for p in range(population_size):
        S[p] = []
        n[p] = 0
        for q in range(population_size):
            # If p dominates q
            if dominates(population[p], population[q]):
                # Add q to the set of solutions dominated by p
                S[p].append(q)
            elif dominates(population[q], population[p]):
                # Increment the domination counter of p
                n[p] = n[p] + 1
        # p belongs to the first front
        if n[p] == 0:
            rank[p] = 0
            front[0].append(p)

    # Initialize the front counter
    i = 0
    while front[i]:
        # Used to store the members of the next front
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                # q belongs to the next front
                if n[q] == 0:
                    rank[q] = i+1
                    Q.append(q)
        i = i+1
        front.append(Q)

    # With this implementation the final front will be empty
    front.pop()

    # Convert front indices to population
    candidate_fronts = []
    for f in front:
        cands = []
        for idx in f:
            cands.append(population[idx])
        candidate_fronts.append(cands)

    # Manually increment all ranks by 1 to match NSGA-II convention
    rank = [r + 1 for r in rank]

    # Set all ranks
    for candidate, r in zip(population, rank):
        candidate.rank = r

    return candidate_fronts


def calculate_crowding_distance(front: list[Prescriptor]):
    """
    Set crowding distance of each candidate in front in-place.
    :param front: The list of prescriptors to calculate crowding distance for.
    """
    n_objectives = len(front[0].metrics)
    for candidate in front:
        candidate.distance = 0
    for m in range(n_objectives):
        # Sort by the mth objective
        sorted_front = sorted(front, key=lambda candidate: candidate.metrics[m])

        obj_min = sorted_front[0].metrics[m]
        obj_max = sorted_front[-1].metrics[m]

        # Edges have infinite distance
        sorted_front[0].distance = float('inf')
        sorted_front[-1].distance = float('inf')
        # If all candidates have the same value, they have a distance of 0 so we can skip this process
        if obj_max != obj_min:
            for i in range(1, len(sorted_front) - 1):
                dist = sorted_front[i+1].metrics[m] - sorted_front[i-1].metrics[m]
                # Normalize distance by objective range
                sorted_front[i].distance += dist / (obj_max - obj_min)


def dominates(candidate1: Prescriptor, candidate2: Prescriptor) -> bool:
    """
    Determine if one individual dominates another.
    First we check feasibility dominance by counting constraint violations.
    One individual dominates another if it's doing better in at least one objective and better than or equal to in all
    the rest.
    metrics are always minimized: lower is better.
    :param candidate1: First candidate in comparison.
    :param candidate2: Second candidate in comparison.
    :return: Whether candidate 1 dominates candidate 2.
    """
    # Domination begins with constraint violation
    if candidate1.cv < candidate2.cv:
        return True
    if candidate1.cv > candidate2.cv:
        return False

    # If both candidates are feasible or have the same constraint violation, we compare metrics
    better = False
    for obj1, obj2 in zip(candidate1.metrics, candidate2.metrics):
        if obj1 > obj2:
            return False
        if obj1 < obj2:
            better = True
    return better

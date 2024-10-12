import numpy as np
from argparse import ArgumentParser
from functools import partial
import matplotlib.pyplot as plt
import datetime
from copy import copy, deepcopy
import os


parser = ArgumentParser(
    description="Solving 0-1 Knapsack problem using genetic approach.",
    allow_abbrev=False,
)
parser.add_argument("-c", "--config", default="config_1.txt",
                    help="Must be located in the ./ps2-idai610-config/ dir")
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("-s", "--selection", default="roulette",
                    help="One of ['roulette', 'tournament']. Default roulette")
parser.add_argument("-ts", "--tournament-size", type=int, default=3,
                    help="If tournament selection, specify tournament size." \
                         " Default 3")
parser.add_argument("--disable-crossover", action="store_true",
                    help="Disable crossover.")
parser.add_argument("--disable-mutation", action="store_true",
                    help="Disable mutation.")
parser.add_argument("-cr", "--crossover-rate", type=float, default=1.0,
                    help="Must be in [0, 1]. Default 1.0")
parser.add_argument("-mr", "--mutation-rate", type=float, default=0.05,
                    help="Must be in [0, 1]. Default 0.05")
parser.add_argument("-er", "--elitism-rate", type=float, default=0.,
                    help="Must be in [0, 1]. Defaults 0")
parser.add_argument("-ps", "--population-size", type=int, default=0,
                    help="Defaults to the one in config file")
parser.add_argument("-ngens", "--n-generations", type=int, default=0,
                    help="Defaults to the one in config file")
parser.add_argument("-f", "--fitness", type=str, default="sum",
                    help="One of ['sum', 'penalty']")
parser.add_argument("-v", "--visualize", action="store_true",
                    help="Whether to save visualizations.")
parser.add_argument("-sc", "--stop", type=str, default=None,
                    help="One of ['notimproving']")
args = parser.parse_args()


def load_config(config: str):
    """
    Loads data from given configs.
    
    Args:
        config: str, Path to config file.

    Returns:
        population: 2d array, Contains the population
        items: 2d array, each row represents an items, first column corresponds
            to weight while second corresponds to value. 
        chromosome_length: int, Length of the chromosome
        knapsack_capacity: int, Maximum weight the knapsack can support
        final_gen: int, Final generation (stopping condition)
    """
    rng = np.random.default_rng(63842)
    # Populate the random variables
    with open("./ps2-idai610-config/" + config, 'r') as f:
        lines = f.readlines()
    population_size, chromosome_length, final_gen, knapsack_capacity = map(
        int,
        [lines[i].strip() for i in range(4)])
    items = np.asarray(
        [tuple(map(int, line.strip().split())) for line in lines[4:]])

    return (
        items,
        population_size,
        chromosome_length,
        knapsack_capacity,
        final_gen
    )


def initialize_population(
        rng: np.random.Generator,
        population_size: int,
        chromosome_length: int,
        gene_prob: float
):
    # noinspection PyTypeChecker
    population = rng.binomial(
        n=1,
        p=gene_prob,
        size=(population_size, chromosome_length))
    return population


def sum_fitness(
        population: np.array,
        items: np.array,
        knapsack_capacity: int
):
    def compute_fitness_per_individual(individual: np.array):
        weight = np.dot(individual, items[:, 0])
        if weight > knapsack_capacity:
            return 0.
        else:
            return np.dot(individual, items[:, 1]).astype(np.float64)

    fitness_scores = np.apply_along_axis(
        compute_fitness_per_individual,
        axis=1,
        arr=population)
    return fitness_scores


def penalty_based_fitness(
        population: np.array,
        items: np.array,
        knapsack_capacity: int,
        penalty_coefficient: float,
):
    def compute_fitness_per_individual(individual: np.array):
        weight = np.dot(individual, items[:, 0])
        value = np.dot(individual, items[:, 1])
        violation = max(0, weight - knapsack_capacity)
        p = 1 / (1 + penalty_coefficient * violation)
        return value * p

    fitness_scores = np.apply_along_axis(
        compute_fitness_per_individual,
        axis=1,
        arr=population)
    return fitness_scores


def roulette_wheel_selection(
        rng: np.random.Generator,
        fitness_scores: np.array,
        population_size: int,
):
    """
    Roulette Wheel Selection

    Args:
        rng: numpy random generator instance.
        fitness_scores: 1d array of fitness score for each individual
        population_size: int, Size of the population.

    Returns:
        A tuple of indices for two parents.
        (parent_1_idx, parent_2_idx)
    """
    idx = np.arange(population_size)
    fitness_scores = np.copy(fitness_scores) + 1e-8
    fitness_probs = fitness_scores / np.sum(fitness_scores)
    parent_1_idx = rng.choice(
        a=idx,
        p=fitness_probs)
    idx = np.delete(idx, parent_1_idx)
    fitness_scores = np.delete(fitness_scores, parent_1_idx)
    fitness_probs = fitness_scores / np.sum(fitness_scores)
    parent_2_idx = rng.choice(
        a=idx,
        p=fitness_probs)
    return parent_1_idx, parent_2_idx


def tournament_selection(
        rng: np.random.Generator,
        fitness_scores: np.array,
        population_size: int,
        tournament_size: int,
):
    """
    Tournament Selection.

    Args:
        rng: numpy random generator instance.
        fitness_scores: 1d numpy array containing fitness scores for each
            individual
        population_size: int, Size of the population.
        tournament_size: int, size of each tournament.
    
    Returns:
        A tuple of indices for two parents.
        (parent_1_idx, parent_2_idx)
    """
    participants = rng.choice(np.arange(population_size), size=tournament_size)
    parent_1_idx = participants[np.argmax(fitness_scores[participants])]
    parent_2_idx = parent_1_idx
    while parent_1_idx == parent_2_idx:
        participants = rng.choice(
            np.arange(population_size), size=tournament_size)
        parent_2_idx = participants[np.argmax(fitness_scores[participants])]
    return parent_1_idx, parent_2_idx


def single_point_crossover(
        rng: np.random.Generator,
        parent_1: np.array,
        parent_2: np.array
):
    perform_crossover = rng.uniform(0., 1.) < args.crossover_rate
    if perform_crossover:
        crossover_idx = rng.integers(0, len(parent_1))
        child_1 = np.concatenate([
            parent_1[:crossover_idx],
            parent_2[crossover_idx:]
        ])
        child_2 = np.concatenate([
            parent_1[:crossover_idx],
            parent_2[crossover_idx:]
        ])
    else:
        child_1 = parent_1
        child_2 = parent_2

    return child_1, child_2


def point_mutation(
        rng: np.random.Generator,
        chromosome: np.array,
        mutation_rate: float,
):
    for i in range(len(chromosome)):
        chromosome[i] ^= rng.uniform(0., 1.) < mutation_rate
    return chromosome


def simple_stopping(curr_gen, final_gen, **kwargs):
    return curr_gen < final_gen


def stop_if_not_improving(
        curr_gen: int,
        final_gen: int,
        curr_best: int,
        prev_best: int,
        patience: int = 5):
    if "nii" not in stop_if_not_improving.__dict__:
        stop_if_not_improving.nii = 0
    if curr_gen < final_gen:
        if curr_best > prev_best:
            stop_if_not_improving.nii = 0
            return True
        else:
            stop_if_not_improving.nii += 1
            if stop_if_not_improving.nii >= patience:
                print(
                    f"Stopping Reason: No improvement for {patience} generations.")
                return False
            return True
    return False


# Evolution
def evolve(rng: np.random.Generator, p_size=None, verbose=True):
    curr_gen = 0
    gene_prob = 0.05 if args.config == "config_2.txt" else 0.5
    mutation_rate = args.mutation_rate
    (
        items,
        population_size,
        chromosome_length,
        knapsack_capacity,
        final_gen
    ) = load_config(args.config)

    if args.population_size:
        population_size = args.population_size

    if p_size is not None:
        population_size = p_size

    population = initialize_population(
        rng,
        population_size,
        chromosome_length,
        gene_prob
    )

    if args.fitness == "sum":
        fitness_fn = partial(
            sum_fitness,
            items=items,
            knapsack_capacity=knapsack_capacity,
        )
    elif args.fitness == "penalty":
        penalty_coefficient = np.mean(items[:, 1])
        fitness_fn = partial(
            penalty_based_fitness,
            items=items,
            knapsack_capacity=knapsack_capacity,
            penalty_coefficient=penalty_coefficient,
        )
    else:
        raise ValueError("Invalid value for fitness.")

    if args.selection == "roulette":
        selection_fn = partial(
            roulette_wheel_selection,
            population_size=population_size)
    elif args.selection == "tournament":
        selection_fn = partial(
            tournament_selection,
            population_size=population_size,
            tournament_size=args.tournament_size,
        )
    else:
        raise ValueError("Invalid value for selection.")

    crossover_fn = single_point_crossover
    mutation_fn = point_mutation

    stopping_criterion_fn = simple_stopping
    if args.stop == "notimproving":
        print("Set to not improving criteria.")
        stopping_criterion_fn = stop_if_not_improving

    final_gen = final_gen if not args.n_generations else args.n_generations - 1

    fitness_scores = fitness_fn(population)
    if verbose:
        print(f"Generation {curr_gen} | Best Ind Score: {np.max(fitness_scores)} "
              f"| Overall Avg: {np.mean(fitness_scores)}\n")
    best_curr_idx = np.argmax(fitness_scores)
    best_overall = dict(
        individual=population[best_curr_idx],
        fitness=fitness_scores[best_curr_idx],
        avg_fitness=np.mean(fitness_scores),
        generation=0,
    )
    avg_fitness_per_generation = [np.mean(fitness_scores)]
    best_fitness_per_generation = [np.max(fitness_scores)]
    num_active_gene_of_best_curr = [np.sum(best_overall["individual"])]
    best_fitness_overall = copy(best_fitness_per_generation)
    num_active_genes_of_best_overall = copy(num_active_gene_of_best_curr)

    n_elites = int(args.elitism_rate * population_size)
    n_elites += 1 if (n_elites % 2 != 0) else 0  # convert odd to even

    prev_best = -np.inf
    while (
            stopping_criterion_fn(
                curr_gen,
                final_gen,
                curr_best=np.max(fitness_scores),
                prev_best=prev_best)
    ):
        if (curr_gen + 1) % 10 == 0:
            mutation_rate = min(mutation_rate + 0.05, 0.2)
        parents = population
        elites = parents[np.argsort(fitness_scores)][::-1][:n_elites]
        children = np.zeros(
            (population_size - n_elites, chromosome_length),
            dtype=np.int64)
        n_children = 0
        while n_children < (population_size - n_elites):
            # Selection
            parent_1_idx, parent_2_idx = selection_fn(
                rng,
                fitness_scores)
            child_1 = parents[parent_1_idx]
            child_2 = parents[parent_2_idx]

            # Crossover
            if not args.disable_crossover:
                child_1, child_2 = crossover_fn(
                    rng,
                    child_1,
                    child_2)

            # Mutation
            if not args.disable_mutation:
                child_1 = mutation_fn(rng, child_1, mutation_rate)
                child_2 = mutation_fn(rng, child_2, mutation_rate)

            children[n_children, :] = child_1
            n_children += 1
            children[n_children, :] = child_2
            n_children += 1

        population = np.concatenate([elites, children], axis=0)
        fitness_scores = fitness_fn(population)
        curr_gen += 1
        best_curr_idx = np.argmax(fitness_scores)
        prev_best = best_overall["fitness"]
        if fitness_scores[best_curr_idx] > best_overall["fitness"]:
            best_overall["individual"] = population[best_curr_idx]
            best_overall["fitness"] = fitness_scores[best_curr_idx]
            best_overall["generation"] = curr_gen

        avg_fitness_per_generation.append(np.mean(fitness_scores))
        best_fitness_per_generation.append(np.max(fitness_scores))
        num_active_gene_of_best_curr.append(np.sum(population[best_curr_idx]))
        best_fitness_overall.append(best_overall["fitness"])
        num_active_genes_of_best_overall.append(np.sum(best_overall["individual"]))

        if verbose:
            print(
                f"Generation {curr_gen} | Best Ind Score: {np.max(fitness_scores):.0f} "
                f"| Overall Avg: {np.mean(fitness_scores):.4f} "
                f"| Best Overall Score: {best_overall['fitness']:.0f}\n")

        # if np.sum(fitness_scores) == 0 or np.sum(fitness_scores) == 1:
        #    print("The solution has converged due to the death of population.")
        #    break

    if args.visualize:
        if not os.path.exists("./viz/"):
            os.mkdir("./viz/")
        fig, ax = plt.subplots()
        x_idx = np.arange(len(avg_fitness_per_generation))
        ax.plot(x_idx, avg_fitness_per_generation,
                label="Average Current Gen Fitness Score")
        ax.plot(x_idx, best_fitness_per_generation,
                label="Best Current Gen Fitness Score")
        ax.plot(x_idx, best_fitness_overall,
                label="Best Fitness Score Overall")
        plt.suptitle("Generations History")
        plt.title(f"config: {args.config} | selection: {args.selection} | "
                  f"fitness method: {args.fitness} | Population Size: {population_size} | "
                  f"crossover rate: {0. if args.disable_crossover else args.crossover_rate:.2f} | "
                  f"mutation rate: {0. if args.disable_mutation else mutation_rate:.2f}",
                  fontsize=6)
        ax.set_xlabel("Generation #")
        ax.set_ylabel("Fitness Score")
        ax.legend()
        filename = "./viz/generation_history_scores_run.png"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename.rsplit('.', 1)[0]}_{timestamp}.{filename.rsplit('.', 1)[1]}"
        plt.savefig(filename)
        print(f"Successfully saved {filename} to disk.")

        # Now plot genes
        fig, ax = plt.subplots()
        ax.plot(x_idx, num_active_gene_of_best_curr,
                label="Number of Active Genes Of Curr Best")
        ax.plot(x_idx, num_active_genes_of_best_overall,
                label="Number of Active Genes Of Overall Best")
        plt.suptitle("Active Genes History")
        plt.title(f"config: {args.config} | selection: {args.selection} | "
                  f"fitness method: {args.fitness} | Population Size: {population_size} | "
                  f"crossover rate: {0. if args.disable_crossover else args.crossover_rate:.2f} | "
                  f"mutation rate: {0. if args.disable_mutation else mutation_rate:.2f}",
                  fontsize=6)
        ax.set_xlabel("Generation #")
        ax.set_ylabel("Active Genes")
        ax.legend()
        filename = "./viz/generation_history_genes_run.png"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename.rsplit('.', 1)[0]}_{timestamp}.{filename.rsplit('.', 1)[1]}"
        plt.savefig(filename)
        print(f"Successfully saved {filename} to disk.")

    return population, best_overall


if __name__ == "__main__":
    rng = np.random.default_rng(args.seed)
    _, best = evolve(rng)
    print(f"Best occurs in the generation: {best['generation']}")

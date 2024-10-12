import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import datetime
import os


parser = ArgumentParser(
    description="Solving 0-1 Knapsack problem using Tabu Search.",
    allow_abbrev=False,
)
parser.add_argument("--n-iters", type=int, default=0,
                    help="Defaults to config.")
parser.add_argument("--tabu-size", type=int, default=10,
                    help="Size of the tabu list")
parser.add_argument("--verbose", action="store_true",
                    help="Print progress")
parser.add_argument("--visualize", action="store_true",
                    help="Generation Visualizations")
parser.add_argument("--initial-solution", type=str, default="random",
                    help="One of ['random', 'greedy']")
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


def compute_value(curr_solution: np.array, values: np.array):
    """
    Computes value of a given solution.
    Assumes the user will check for capacity constraints downstream.

    Args:
        curr_solution: Array containing current solution
        values: Array of items item values

    Returns:
        An integer representing the value of a given solution.
    """
    return np.dot(values, curr_solution)



def compute_weight(curr_solution: np.array, weights: np.array):
    """
    Computes weight of a given solution.
    Assumes the user will check for capacity constraints downstream.

    Args:
        curr_solution: Array containing current solution
        weights: Array of items weights

    Returns:
        An integer representing the weight of a given solution.
    """
    return np.dot(weights, curr_solution)


def random_initial_solution(rng: np.random.Generator,
                            weights: np.array,
                            capacity: int,
                            random_prob: float = 0.5):
    while True:
        random_solution = rng.binomial(
            n=1,
            p=random_prob,
            size=len(weights),
        )
        if compute_weight(random_solution, weights) <= capacity:
            return random_solution


def greedy_initial_solution(values: np.array, weights: np.array, capacity: int):
    """
    Chooses the initial solution based on the highest value to weight ratio.

    Args:
        values: Array of items item values
        weights: Array of items weights
        capacity: int, Max capacity of the knapsack

    Returns:
        Array containing an initial solution.
    """
    n = len(values)
    solution = np.zeros(n)
    v_to_w_ratios = []
    for i in range(n):
        if weights[i] > 0:
            v_to_w_ratios.append((i, values[i] / weights[i]))
        else:
            v_to_w_ratios.append((i, np.inf))
    v_to_w_ratios = sorted(v_to_w_ratios, key=lambda x: x[1], reverse=True)
    cum_weight = 0

    for i, _ in v_to_w_ratios:
        if cum_weight + weights[i] <= capacity:
            solution[i] = 1
            cum_weight += weights[i]
        else:
            break
    return solution


def get_neighbors(curr_solution: np.array, weights: np.array, capacity: int):
    """
    Generates neighbors of current solution by flipping one bit at a time.

    Args:
        curr_solution: Array representing current solution.
        weights: Array of items weights
        capacity: int, Max capacity of the knapsack

    Returns:
        A list containing neighbors of current solution.
    """
    neighbors = []
    for i in range(len(curr_solution)):
        neighbor = np.copy(curr_solution)
        neighbor[i] = 1 - curr_solution[i]
        # Check if the neighbor is valid i.e. satisfies the capacity constraint
        if compute_weight(neighbor, weights) <= capacity:
            neighbors.append(neighbor)
    return neighbors


def tabu_search(
        rng: np.random.Generator,
        values: np.array,
        weights: np.array,
        capacity: int,
        n_iters: int,
        tabu_size: int = 10,
        verbose: bool = True,
        random_prob: float = 0.5,
        config:str="",
):
    """
    Performs the forbidden search

    Args:
        values: Array of items item values
        weights: Array of items weights
        capacity: int, Max capacity of the knapsack
        n_iters: int, Maximum number of iterations
        tabu_size: int, Max size of the tabu list
        verbose: bool, Print progress

    Returns:
        Returns the best solution.
    """
    if args.initial_solution == "greedy":
        curr_solution = greedy_initial_solution(values, weights, capacity)
    elif args.initial_solution == "random":
        curr_solution = random_initial_solution(rng, weights, capacity, random_prob)
    else:
        raise ValueError("Invalid value for initial solution.")

    best_solution = np.copy(curr_solution)
    best_value = compute_value(best_solution, values)
    tabu_list = []

    best_curr_value_per_iter = []
    best_overall_value_per_iter = []
    curr_n_items_in_bag_per_iter = []
    best_n_items_in_bag_per_iter = []

    for iter in range(n_iters):
        neighbors = get_neighbors(curr_solution, weights, capacity)
        best_neighbor = None
        best_neighbor_value = -np.inf

        for neighbor in neighbors:
            neighbor_value = compute_value(neighbor, values)
            if (
                    (neighbor_value > best_neighbor_value)
                    and
                    (str(id(neighbor)) not in tabu_list or neighbor_value > best_value)
            ):
                best_neighbor = neighbor
                best_neighbor_value = neighbor_value

        if best_neighbor is None:
            curr_n_items_in_bag_per_iter.append(0)
            best_curr_value_per_iter.append(0)
            break

        curr_solution = best_solution

        if best_neighbor_value > best_value:
            best_solution = best_neighbor
            best_value = best_neighbor_value

        tabu_list.append(str(id(curr_solution)))
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        best_overall_value_per_iter.append(best_value)
        best_n_items_in_bag_per_iter.append(sum(best_solution))
        best_curr_value_per_iter.append(best_neighbor_value)
        curr_n_items_in_bag_per_iter.append(sum(best_neighbor))

        if verbose:
            print(f"Iter: {iter} | "
                  f"Curr Value: {compute_value(curr_solution, values)} | "
                  f"Best Value: {best_value}")

    if args.visualize:
        if not os.path.exists("./viz/"):
            os.mkdir("./viz/")
        fig, ax = plt.subplots()
        x_idx = np.arange(len(best_curr_value_per_iter))
        ax.plot(x_idx, best_curr_value_per_iter,
                label="Best Current Value")
        ax.plot(x_idx, best_overall_value_per_iter,
                label="Best Overall Value")
        plt.suptitle("Solutions History")
        plt.title(f"config: {config} | "
                  f"initial_solution: {args.initial_solution}",
                  fontsize=10)
        ax.set_xlabel("Iteration #")
        ax.set_ylabel("Knapsack Value")
        ax.legend()
        filename = f"./viz/tabu_{config}_solution_history_run.png"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename.rsplit('.', 1)[0]}_{timestamp}.{filename.rsplit('.', 1)[1]}"
        plt.savefig(filename)
        print(f"Successfully saved {filename} to disk.")

        # Now plot n items in bag
        fig, ax = plt.subplots()
        ax.plot(x_idx, curr_n_items_in_bag_per_iter,
                label="n_items_in_curr_best")
        ax.plot(x_idx, best_n_items_in_bag_per_iter,
                label="n_items_in_best_overall")
        plt.suptitle("N Items in Knapsack")
        plt.title(f"config: {config} | "
                  f"initial_solution: {args.initial_solution}",
                  fontsize=10)
        ax.set_xlabel("Iteration #")
        ax.set_ylabel("N Items")
        ax.legend()
        filename = f"./viz/tabu_{config}_n_items_history_run.png"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename.rsplit('.', 1)[0]}_{timestamp}.{filename.rsplit('.', 1)[1]}"
        plt.savefig(filename)
        print(f"Successfully saved {filename} to disk.")

    return best_solution, best_value


if __name__ == "__main__":
    rng = np.random.default_rng(7777)
    for config in ["config_1.txt", "config_2.txt"]:
        print("Processing {}".format(config))
        items, _, _, capacity, n_iters = load_config(config)
        n_iters += 1
        if args.n_iters > 0:
            n_iters = args.n_iters
        weights, values = items[:, 0], items[:, 1]
        best_solution, best_value = tabu_search(
            rng,
            values=values,
            weights=weights,
            capacity=capacity,
            n_iters=n_iters,
            tabu_size=args.tabu_size,
            verbose=args.verbose,
            random_prob=0.05 if config == "config_2.txt" else 0.5,
            config=config,
        )

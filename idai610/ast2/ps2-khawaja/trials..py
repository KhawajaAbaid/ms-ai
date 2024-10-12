from knapsack_01_genetic import (evolve, stop_if_not_improving, sum_fitness,
                                 load_config)
import numpy as np
import pandas as pd


n_trials = 30
population_sizes = np.arange(50, 1001, step=50)
final_values = []
final_weights = []
final_no_items = []
stopping_fn = stop_if_not_improving
fitness_fn = sum_fitness

(
    items,
    population_size,
    chromosome_length,
    knapsack_capacity,
    final_gen
) = load_config('config_1.txt')


def run_trials(rng: np.random.Generator, p_size):
    print("Trial: ", end="")
    for i in range(n_trials):
        print(i, end=" ")
        final_population, _ = evolve(rng, p_size=p_size, verbose=False)
        scores = fitness_fn(final_population, items, knapsack_capacity)
        max_idx = np.argmax(scores)
        best_value = scores[max_idx]
        best_individual = final_population[max_idx]
        best_weight = np.sum(items[:, 0][np.flatnonzero(best_individual)])
        n_items = sum(best_individual)

        final_values.append(best_value)
        final_weights.append(best_weight)
        final_no_items.append(n_items)
    total_value_str = f"{np.mean(final_values):.2f} +- {np.std(final_values):.2f}"
    max_weight_str = f"{np.max(final_weights)}"
    n_items = f"{int(np.mean(final_no_items))}"
    print()
    return total_value_str, max_weight_str, n_items


table = {"Population_Size": [], "Total_Value": [], "Max_Weight": [],
         "N_Items_in_Best": []}

rng = np.random.default_rng(38416)
for ps in population_sizes:
    print("Population Size:", ps)
    total_value, max_weight, n_items = run_trials(rng, ps)
    table["Population_Size"].append(ps)
    table["Total_Value"].append(total_value)
    table["Max_Weight"].append(max_weight)
    table["N_Items_in_Best"].append(n_items)

print(table)
# # table_df = pd.DataFrame(table)
# table_df = pd.DataFrame(data={
#     'Population_Size': pd.Series(table['Population_Size'], dtype='int'),
#     # 'Total_Value': pd.Series(table['Total_Value'], dtype='object'),
#     # 'Max_Weight': pd.Series(table['Max_Weight'], dtype='object'),
#     # 'N_Items_in_Best': pd.Series(table['N_Items_in_Best'], dtype='object')
# })
# table_df.to_csv("table.csv")
# print(table_df)


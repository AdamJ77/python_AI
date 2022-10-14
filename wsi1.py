import numpy as np
import itertools as it
from matplotlib import pyplot as plt

import random
import time

weights = np.array([8, 3, 5, 2])
capacity = 9
profits = np.array([16, 8, 9, 6])


class KnapSack:
  def __init__(self, profits, weights, capacity):
    self.profits = profits
    self.weights = weights
    self.capacity = capacity

  def solve_knapsack_brute_force(self):
    """
    might as well sort zip list with lambda on occurrences if 0 then stops already iterating, if 1 it still goes
    is sorting actually faster than going through each 0?
    """
    start_time = time.monotonic()
    variations = it.product(range(2), repeat=len(profits))
    max_sum_profits = 0
    max_sum_weight = 0
    final_indexes = []
    for variation in variations:
      curr_sum_profit = 0
      curr_sum_weight = 0
      indexes = []
      for index, (occurrence, profit, weight) in enumerate(zip(variation, self.profits, self.weights)):
        if occurrence:
          if curr_sum_weight + weight > self.capacity:
            break
          curr_sum_weight += weight
          curr_sum_profit += profit
          indexes.append(index)
      if curr_sum_profit > max_sum_profits:
        max_sum_profits = curr_sum_profit
        max_sum_weight = curr_sum_weight
        final_indexes = indexes.copy()
    running_time = time.monotonic() - start_time
    return {"Max profit": max_sum_profits, "Max weight": max_sum_weight, "Indexes": final_indexes}, running_time

  def solve_knapsack_pw_ratio(self):
    pass

def get_plot_brute_force(knapsack, n_elements):
  times = []
  n_profits = []
  for _ in range(n_elements):
    rand_profit = random.randint(1, 20)
    rand_weight = random.randint(1, 10)
    knapsack.profits = np.append(knapsack.profits, rand_profit)
    knapsack.weights = np.append(knapsack.weights, rand_weight)
    n_profits.append(len(knapsack.weights))
    info, time_run = knapsack.solve_knapsack_brute_force()
    times.append(time_run)
  plt.plot(n_profits, times, linewidth=2.0)
  plt.ylabel("Time")
  plt.xlabel("Number of elements")
  plt.show()
  plt.savefig("Plot.png")


if __name__ == "__main__":
  knapsack = KnapSack(profits, weights, capacity)
  information_value, time_run = knapsack.solve_knapsack_brute_force()
  print(information_value)
  get_plot_brute_force(knapsack, 10)
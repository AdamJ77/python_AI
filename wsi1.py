import numpy as np
import itertools as it

weights = np.array([8, 3, 5, 2])
capacity = 9
profits = np.array([16, 8, 9, 6])

class KnapSack:
  def __init__(self, profits, weights, capacity):
    self.profits = profits
    self.weights = weights
    self.capacity = capacity

  def solve_knapsack_brute_force(self):
    variations = it.product(range(2), repeat=len(profits))
    for variation in variations:
        print(variation)

  def solve_knapsack_pw_ratio(self):
    pass


if __name__ == "__main__":
  knapsack = KnapSack(profits, weights, capacity)
  knapsack.solve_knapsack_brute_force()
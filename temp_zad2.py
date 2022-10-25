import numpy as np
import matplotlib.pyplot as plt
import random

class Chromosome:
  def __init__(self, length: int, array=None): #if array is None it should be initialized with random binary vector
    if array is None:
      self.array = np.empty((length, 1), dtype=int)
      val_gen = (1 if random.randint(0,1) else 0 for _ in range(length))
      for val in val_gen:
        self.array = np.insert(self.array, int, val)

  def decode(self, lower_bound, upper_bound, aoi):
    pass

  def mutation(self, probability):
    pass

  def crossover(self, other):
    pass


if __name__ == "__main__":
    chrm = Chromosome(16)
    print(chrm.array)
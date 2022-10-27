import numpy as np
import matplotlib.pyplot as plt
import random



def min_max_norm(val, min_val, max_val, new_min, new_max):
  return (val - min_val) * (new_max - new_min) / (max_val - min_val) + new_min


class Chromosome:
  def __init__(self, length, array=None): #if array is None it should be initialized with random binary vector
    self.length = length
    if array is None:
      self.array = np.array([], dtype=int)
      val_gen = (1 if random.randint(0,1) else 0 for _ in range(length))
      for val in val_gen:
        self.array = np.append(self.array, val)
    else:
      self.array = array

  def decode(self, lower_bound, upper_bound, aoi):
    array_to_decode = self.array[lower_bound : upper_bound]
    str_bin = "".join(str(bite) for bite in array_to_decode)
    val = int(str_bin, 2)       # change string binary to int decimal
    max_val = 2 ** (self.length) - 1
    norm = min_max_norm(val, 0, max_val, aoi[0], aoi[1])
    return str_bin, val, max_val, round(norm, 3)

  def mutation(self, probability) -> None:
    prob = random.random()
    if prob < probability:
      rand_index = self.get_random_index(self.length)
      self.array[rand_index] = 0 if self.array[rand_index] else 1

  def crossover(self, other):
    point_of_split = self.get_random_index(self.length)
    child1 = np.append(self.array[:point_of_split + 1], other.array[point_of_split + 1:])
    child2 = np.append(self.array[point_of_split + 1:], other.array[:point_of_split + 1])
    return child1, child2

  @staticmethod
  def get_random_index(upper_limit) -> int:
    return random.randint(0, upper_limit - 1)



class GeneticAlgorithm:
  def __init__(self, chromosome_length: int, obj_func_num_args: int, objective_function, aoi, population_size=1000,
               tournament_size=2, mutation_probability=0.05, crossover_probability=0.8, num_steps=30):
    assert chromosome_length % obj_func_num_args == 0, "Number of bits for each argument should be equal"
    self.chromosome_lengths = chromosome_length
    self.obj_func_num_args = obj_func_num_args
    self.bits_per_arg = int(chromosome_length / obj_func_num_args)
    self.objective_function = objective_function
    self.aoi = aoi
    self.tournament_size = tournament_size
    self.mutation_probability = mutation_probability
    self.crossover_probability = crossover_probability
    self.num_steps = num_steps

  def eval_objective_func(self, chromosome):
    pass

  def tournament_selection(self):
    pass

  def reproduce(self, parents):
    pass

  def plot_func(self, trace):
    X = np.arange(-2, 3, 0.05)
    Y = np.arange(-4, 2, 0.05)
    X, Y = np.meshgrid(X, Y)
    Z = 1.5 - np.exp(-X ** (2) - Y ** (2)) - 0.5 * np.exp(-(X - 1) ** (2) - (Y + 2) ** (2))
    plt.figure()
    plt.contour(X, Y, Z, 10)
    cmaps = [[ii / len(trace), 0, 0] for ii in range(len(trace))]
    plt.scatter([x[0] for x in trace], [x[1] for x in trace], c=cmaps)
    plt.show()

  def run(self):
    pass


if __name__ == "__main__":
    chrm = Chromosome(8, [1, 0, 0, 0, 0, 1, 1, 1])
    print(chrm.array)

    val = chrm.decode(0, 9, [0, 1])
    print(val)

    # print(min_max_norm(135, 0, 255, 0, 1))


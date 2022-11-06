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
      val_gen = (1 if random.randint(0,1) else 0 for _ in range(self.length))
      for val in val_gen:
        self.array = np.append(self.array, val)
    else:
      self.array = array


  def decode(self, lower_bound, upper_bound, aoi):
    """Upper bound not included"""
    array_to_decode = self.array[lower_bound : upper_bound]
    str_bin = "".join(str(bite) for bite in array_to_decode)
    val = int(str_bin, 2)       # change string binary to int decimal
    max_val = 2**(self.length) - 1
    norm = min_max_norm(val, 0, max_val, aoi[0], aoi[1])
    return norm


  def mutation(self, probability):
    prob = random.random()
    if prob < probability:
      rand_index = self.get_random_index(self.length)
      self.array[rand_index] = 0 if self.array[rand_index] else 1



  @staticmethod
  def get_random_index(upper_limit) -> int:
    return random.randint(0, upper_limit - 1)


  def crossover(self, other, prop_of_mutation):
    """
    Point of split after which there is split
    point_of_split:
    0 - switch arrays
    1-7 - before this index is slice point
    """
    point_of_split = self.get_random_index(self.length)
    arr_child1 = np.append(self.array[:point_of_split], other.array[point_of_split:])
    arr_child2 = np.append(other.array[:point_of_split], self.array[point_of_split:])

    # create new children + mutate them
    child1 = Chromosome(self.length, arr_child1)
    child1.mutation(prop_of_mutation)

    child2 = Chromosome(self.length, arr_child2)
    child2.mutation(prop_of_mutation)
    # print(point_of_split)
    return (child1, child2)



""" GENETIC ALGORITHM """

class GeneticAlgorithm:
  def __init__(self, chromosome_length: int, obj_func_num_args: int, objective_function, aoi, population_size=20,
               tournament_size=2, mutation_probability=0.05, crossover_probability=0.8, num_steps=4):
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
    self.population_size = population_size

  def eval_objective_func(self, chromosome: Chromosome) -> float:
    """Get arguments by decoding and return objective function value"""
    args = self.decode_chrom(chromosome)
    obj_func_val = self.objective_function(*args)
    return obj_func_val  # unneccessary round


  def decode_chrom(self, chromosome: Chromosome):
    """Returns list of arguments by decoding array"""
    bpa = self.bits_per_arg
    return [chromosome.decode(i * bpa, (i+1) * bpa, self.aoi) for i in range(self.obj_func_num_args)]

  def reproduce(self, parents: list):
    prob = random.random()
    if prob < self.crossover_probability:
      parents = parents[0].crossover(parents[1], self.mutation_probability)
    return parents

  def tournament_selection(self, population: np.array):
    new_pop = np.empty(shape=(self.population_size, 2), dtype=list)

    for index in range(self.population_size):
      contestants = np.array([random.choice(population) for _ in range(self.tournament_size)])
      survivor = min(contestants, key = lambda cand: cand[1])
      new_pop[index] = survivor
    return new_pop

  def run(self):
    chrom_length = self.chromosome_lengths
    trace_of_best = []

    population1 = np.array([[Chromosome(chrom_length), 0] for _ in range(self.population_size)])
    # population = np.empty(shape=(self.population_size, 2), dtype=list)
    # for index in range(self.population_size):
    #     population[index][0] = Chromosome(chrom_length)
    # return population

    # evaluate all
    self.evaluate_all(population1)
    # save current best value
    current_best_value = self.find_best_individual(population1, trace_of_best)

    for _ in range(self.num_steps):
      # create new population after tournament
      new_pop = self.tournament_selection(population1)

      # reproducing (crossing + mutation)
      for index in range(0, self.population_size, 2):
        parent1 = new_pop[index]
        parent2 = new_pop[index+1]
        child1, child2 = self.reproduce([parent1[0], parent2[0]])
        population1[index][0] = child1
        population1[index+1][0] = child2

      self.evaluate_all(population1)
      population_best_value = self.find_best_individual(population1, trace_of_best)

      if population_best_value <= current_best_value:
        current_best_value = population_best_value

    self.plot_func(trace_of_best)

    return current_best_value

  def evaluate_all(self, population):
    """Update individuals value"""
    for index, (chrom, val) in enumerate(population):
        population[index][1] = self.eval_objective_func(chrom)

  def find_best_individual(self, population, trace):
    """Return best population value AND append its arguments to trace"""
    best_ind_of_pop = min(population, key = lambda cand: cand[1])
    bop_args = self.decode_chrom(best_ind_of_pop[0])
    trace.append(bop_args)
    return best_ind_of_pop[1]

  def plot_func(self, trace):
    X = np.arange(-2, 3, 0.05)
    Y = np.arange(-4, 2, 0.05)
    X, Y = np.meshgrid(X, Y)
    Z = 1.5 - np.exp(-X ** (2) - Y ** (2)) - 0.5 * np.exp(-(X - 1) ** (2) - (Y + 2) ** (2))
    plt.figure()
    plt.contour(X, Y, Z, 10)
    cmaps = [[ii / len(trace), 0, 0] for ii in range(len(trace))]
    plt.scatter([x[0] for x in trace], [x[1] for x in trace], c=cmaps)
    plt.savefig("ga.png")



def objective_function_f(*args):
    x1, x2 = args
    return (1.5 - np.exp(-x1 ** (2) - x2 ** (2)) - 0.5 * np.exp(-(x1 - 1) ** (2) - (x2 + 2) ** (2)))






if __name__ == "__main__":
  ch_length = 8
  n_arg = 2
  # aoi = [0,1]

  chrm1 = Chromosome(ch_length)
  # print(chrm1.array)

  # chrm2 = Chromosome(ch_length)
  # print(chrm2.array)

  # child1, child2 = chrm1.crossover(chrm2, 0)
  # print(child1.array)
  # print(child2.array)
  # print(chrm1.decode(0, 8, aoi))

  ga = GeneticAlgorithm(ch_length, n_arg, objective_function_f, [0,100])

  # # print(ga.eval_objective_func(chrm1))
  print(ga.run())

  # arr = np.array([['a', 1],
  #                 ['b', 8],
  #                 ['c', 3]])
  # val = arr[:, 1].max()
  # print(val)
  # print(ga.eval_objective_func(chrm1))
  # print(ga.decode_chrom(chrm1))
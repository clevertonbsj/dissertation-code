# -*- coding: utf-8 -*-
import numpy as np
#from deap import base
#from deap import creator
#from deap import algorithms
#from deap import tools
import random as rd
import plotly.io as pio
import plotly.express as px
import mlrose_hiive as mlrose


class Product():
  def __init__(self, name, space, price):
    self.name = name
    self.space = space
    self.price = price
'''
prodlist = []
prodlist.append(Product('Fridge A', 0.751, 999.90))
prodlist.append(Product('Cellphone', 0.00000899, 2199.12))
prodlist.append(Product('TV 55', 0.400, 4346.99))
prodlist.append(Product("TV 50", 0.290, 3999.90))
prodlist.append(Product("TV 42", 0.200, 2999.00))
prodlist.append(Product("Notebook A", 0.00350, 2499.90))
prodlist.append(Product("Elçectric Fan", 0.496, 199.90))
prodlist.append(Product("Microwave A", 0.0424, 308.66))
prodlist.append(Product("Microwave B", 0.0544, 429.90))
prodlist.append(Product("Microwave C", 0.0319, 299.29))
prodlist.append(Product("Fridge B", 0.635, 849.00))
prodlist.append(Product("Fridge C", 0.870, 1199.89))
prodlist.append(Product("Notebook B", 0.498, 1999.90))
prodlist.append(Product("Notebook C", 0.527, 3999.00)) 
spaces = []
prices = []
names = []
for product in prodlist:
  spaces.append(product.space)
  prices.append(product.price)
  names.append(product.name)
'''
prodlist = [('Fridge A', 0.751, 999.90), ('Cellphone', 0.00000899, 2199.12), 
            ('TV 55', 0.400, 4346.99), ("TV 50", 0.290, 3999.90), 
            ("TV 42", 0.200, 2999.00), ("Notebook A", 0.00350, 2499.90),
            ("Elçectric Fan", 0.496, 199.90), ("Microwave A", 0.0424, 308.66), 
            ("Microwave B", 0.0544, 429.90), ("Microwave C", 0.0319, 299.29), 
            ("Fridge B", 0.635, 849.00), ("Fridge C", 0.870, 1199.89), 
            ("Notebook B", 0.498, 1999.90), ("Notebook C", 0.527, 3999.00)]
limit = 3
popu_size = 20
mutation_proba = 0.01
number_of_gens = 100

def fitness(sol):
  cost = 0
  sum_spaces = 0
  for i in range(len(sol)):
    if sol[i] == 1:
      cost += prodlist[i][2]
      sum_spaces += prodlist[i][1]
  if sum_spaces > limit:
    cost = 1
  return cost #for DEAP add a , after cost

''' DEAP Implementation
toolbox = base.Toolbox()
creator.create('FitnessMax', base.Fitness, weights = (1.0,))
creator.create('Individual', list, fitness = creator.FitnessMax)

toolbox.register('attr_bool', rd.randint, 0, 1)
toolbox.register('Individual', tools.initRepeat, creator.Individual, toolbox.attr_bool, n = 14)
toolbox.register('pop', tools.initRepeat, list, toolbox.Individual)
toolbox.register('evaluate', fitness)
toolbox.register('mate', tools.cxOnePoint)
toolbox.register('mutate', tools.mutFlipBit, indpb = 0.01)
toolbox.register('select', tools.selRoulette)


pop = toolbox.pop(n = pop_size)
crossover_prob = 1.0

statistics = tools.Statistics(key = lambda individual: individual.fitness.values)
statistics.register('Max', np.max)
statistics.register('Min', np.min)
statistics.register('Mean', np.mean)
statistics.register('Std', np.std)

pop, info = algorithms.eaSimple(pop, toolbox, crossover_prob, mutation_prob,
                                number_of_gens, statistics)

best_sol = tools.selBest(pop, 1)
for individual in best_sol:
    print(individual)
    print(individual.fitness)
'''
#MLROSe Implementation
fitnessf = mlrose.CustomFitness(fitness)
problem = mlrose.DiscreteOpt(length=len(prodlist), fitness_fn=fitnessf, maximize=True, max_val=2)
best_sol_array, best_sol_fitness, best_sol_curve = mlrose.genetic_alg(problem, pop_size=20, mutation_prob=0.01)
print(best_sol_array, best_sol_fitness)
for i in range(len(best_sol_array)):
    if best_sol_array[i] == 1:
        print('Name: ', prodlist[i][0],
              '- Price: ', prodlist[i][2])

pio.renderers.default = 'svg'
#figure = px.line(x = range(0, 101), y = info.select('Max'), title = 'Genetic Algorithm results')
#figure.show()
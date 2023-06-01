# -*- coding: utf-8 -*-
from random import random
import plotly.express as px
import plotly.io as pio

''' creating the Product Class'''

class Product():
  def __init__(self, name, space, price):
    self.name = name
    self.space = space
    self.price = price
    
'''Creating and filling the product list'''

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

'''Creating the Individual class'''

class Individual():
  def __init__(self, spaces, prices, space_limit, generation=0):
    self.spaces = spaces
    self.prices = prices
    self.space_limit = space_limit
    self.score_evaluation = 0
    self.used_space = 0
    self.generation = generation
    self.chromosome = []
    
    #creating the initial chromosome with random values
    for i in range(len(spaces)):
      if random() < 0.5:
        self.chromosome.append('0')
      else:
        self.chromosome.append('1')
    
  #creating the fitness function
  def fitness(self):
    score = 0
    sum_spaces = 0
    for i in range(len(self.chromosome)):
      if self.chromosome[i] == '1':
        score += self.prices[i]
        sum_spaces += self.spaces[i]
    if sum_spaces > self.space_limit:
      score = 1
    self.score_evaluation = score
    self.used_space = sum_spaces
    
  #creating the crossover function
  def crossover(self, other_individual):
    cutoff = round(random() * len(self.chromosome))

    child1 = other_individual.chromosome[0:cutoff] + self.chromosome[cutoff::]
    child2 = self.chromosome[0:cutoff] + other_individual.chromosome[cutoff::]
    children = [Individual(self.spaces, self.prices, self.space_limit, self.generation + 1),
                Individual(self.spaces, self.prices, self.space_limit, self.generation + 1)]
    children[0].chromosome = child1
    children[1].chromosome = child2
    return children  
  
  #creating the mutation function
  def mutation(self, rate):
      for i in range(len(self.chromosome)):
          if random() < rate:
              if self.chromosome[i] == '1':
                  self.chromosome[i] = '0'
              else:
                  self.chromosome[i] = '1'
      return self
    
'''Creating the Genetic Algortihm class'''

class GeneticAlgorithm():
    def __init__(self, pop_size):
        self.pop_size = pop_size
        self.population = []
        self.generation = 0
        self.best_sol = None
        self.list_of_sol = []
        
    #creating a function to initialize the pop
    def init_pop(self, spaces, prices, space_limit):
        for i in range(self.pop_size):
            self.population.append(Individual(spaces, prices, space_limit))
        self.best_sol = self.population[0]
        
    #creating the function to order the pop
    def order_pop(self):
        self.population = sorted(self.population, key = lambda population: \
                                 population.score_evaluation, reverse = True)

    #creating the best individual function
    def best_individual(self, individual):
        if individual.score_evaluation > self.best_sol.score_evaluation:
            self.best_sol = individual

    #sum of eval
    def sum_evaluation(self):
        sum = 0
        for individual in self.population:
            sum += individual.score_evaluation
        return sum

    #Selecting the parents
    def select_parent(self, sum_evaluation):
        parent = -1
        random_val = random() * sum_evaluation
        sum = 0
        i = 0
        while i < len(self.population) and sum < random_val:
            sum += self.population[i].score_evaluation
            parent += 1
            i += 1
        return parent
    
    #visualizing the generation
    def visualize_gen(self):
        best = self.population[0]
        print('Generation: ', self.population[0].generation,
              'Total Price: ', best.score_evaluation,
              'Space: ', best.used_space,
              'Chromosome: ', best.chromosome)
    
    #creating the solve function
    def solve(self, mutation_probability, number_of_gens, spaces, prices, limit):
        self.init_pop(spaces, prices, limit)
        for individual in self.population:
            individual.fitness()
        self.order_pop()
        #self.visualize_gen()
        for generation in range(number_of_gens):
            sum = self.sum_evaluation()
            new_pop = []
            for new_individuals in range(0, self.pop_size, 2):
                parent1 = self.select_parent(sum)
                parent2 = self.select_parent(sum)
                children = self.population[parent1].crossover(self.population[parent2])
                new_pop.append(children[0].mutation(mutation_probability))
                new_pop.append(children[1].mutation(mutation_probability))
            self.population = list(new_pop)
            for individual in self.population:
                individual.fitness()
            self.order_pop()
            #self.visualize_gen()
            best = self.population[0]
            self.list_of_sol.append(best.score_evaluation)
            self.best_individual(best)
        print('Best Solution - Generation: ', self.best_sol.generation,
              'Total Price: ', self.best_sol.score_evaluation,
              'Space: ', self.best_sol.used_space,
              'Chromosome: ', self.best_sol.chromosome)
        return self.best_sol.chromosome
    
'''testing the code'''
spaces = []
prices = []
names = []
for product in prodlist:
  spaces.append(product.space)
  prices.append(product.price)
  names.append(product.name)
limit = 3
'''
individual1 = Individual(spaces, prices, limit)
for i in range(len(prodlist)):
  print(individual1.chromosome[i])
  if individual1.chromosome[i] == '1':
    print('Name: ', prodlist[i].name)
individual1.fitness()
print('Score: ', individual1.score_evaluation)
print('Used space: ', individual1.used_space)
print('Chromosome: ', individual1.chromosome)
'''
pop_size = 20
'''
ga = GeneticAlgorithm(pop_size)
ga.init_pop(spaces, prices, limit)
for individual in ga.population:
    individual.fitness()
ga.order_pop()
ga.best_individual(ga.population[0])
sum = ga.sum_evaluation()

new_pop = []
'''
mutation_prob = 0.01
'''
for new_individuals in range (0, ga.pop_size, 2):
    parent1 = ga.select_parent(sum)
    parent2 = ga.select_parent(sum)
    children = ga.population[parent1].crossover(ga.population[parent2])
    new_pop.append(children[0].mutation(mutation_prob))
    new_pop.append(children[1].mutation(mutation_prob))
'''
number_of_gens = 100
ga = GeneticAlgorithm(pop_size)
result = ga.solve(mutation_prob, number_of_gens, spaces, prices, limit)
print(result)
for i in range(len(prodlist)):
    if result[i] == '1':
        print('Name: ', prodlist[i].name,
              '- Price: ', prodlist[i].price)

pio.renderers.default = 'svg'
figure = px.line(x = range(0, 100), y = ga.list_of_sol, title = 'Genetic Algorithm results')
figure.show()




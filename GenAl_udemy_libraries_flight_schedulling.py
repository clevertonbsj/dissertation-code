# -*- coding: utf-8 -*-
import numpy as np
#from deap import base
#from deap import creator
#from deap import algorithms
#from deap import tools
#import random as rd
import mlrose_hiive as mlrose

people = [('Lisbon', 'LIS'),
          ('Madrid', 'MAD'),
          ('Paris', 'CDG'),
          ('Dublin', 'DUB'),
          ('Brussels', 'BRU'),
          ('London', 'LHR')]

destination = 'FCO'
flights = {('BRU', 'FCO'): [('6:12', '10:22', 230),
  ('7:53', '11:37', 433),
  ('9:08', '12:12', 364),
  ('10:30', '14:57', 290),
  ('12:19', '15:25', 342),
  ('13:54', '18:02', 294),
  ('15:44', '18:55', 382),
  ('16:52', '20:48', 448),
  ('18:26', '21:29', 464),
  ('20:07', '23:27', 473)],
 ('CDG', 'FCO'): [('6:25', '9:30', 335),
  ('7:34', '9:40', 324),
  ('9:15', '12:29', 225),
  ('11:28', '14:40', 248),
  ('12:05', '15:30', 330),
  ('14:01', '17:24', 338),
  ('15:34', '18:11', 326),
  ('17:07', '20:04', 291),
  ('18:23', '21:35', 134),
  ('19:53', '22:21', 173)],
 ('DUB', 'FCO'): [('6:17', '8:26', 89),
  ('8:04', '10:11', 95),
  ('9:45', '11:50', 172),
  ('11:16', '13:29', 83),
  ('12:34', '15:02', 109),
  ('13:40', '15:37', 138),
  ('15:27', '17:18', 151),
  ('17:11', '18:30', 108),
  ('18:34', '19:36', 136),
  ('20:17', '22:22', 102)],
 ('FCO', 'BRU'): [('6:09', '9:49', 414),
  ('7:57', '11:15', 347),
  ('9:49', '13:51', 229),
  ('10:51', '14:16', 256),
  ('12:20', '16:34', 500),
  ('14:20', '17:32', 332),
  ('15:49', '20:10', 497),
  ('17:14', '20:59', 277),
  ('18:44', '22:42', 351),
  ('19:57', '23:15', 512)],
 ('FCO', 'CDG'): [('6:33', '9:14', 172),
  ('8:23', '11:07', 143),
  ('9:25', '12:46', 295),
  ('11:08', '14:38', 262),
  ('12:37', '15:05', 170),
  ('14:08', '16:09', 232),
  ('15:23', '18:49', 150),
  ('16:50', '19:26', 304),
  ('18:07', '21:30', 355),
  ('20:27', '23:42', 169)],
 ('FCO', 'DUB'): [('6:39', '8:09', 86),
  ('8:23', '10:28', 149),
  ('9:58', '11:18', 130),
  ('10:33', '12:03', 74),
  ('12:08', '14:05', 142),
  ('13:39', '15:30', 74),
  ('15:25', '16:58', 62),
  ('17:03', '18:03', 103),
  ('18:24', '20:49', 124),
  ('19:58', '21:23', 142)],
 ('FCO', 'LHR'): [('6:58', '9:01', 238),
  ('8:19', '11:16', 122),
  ('9:58', '12:56', 249),
  ('10:32', '13:16', 139),
  ('12:01', '13:41', 267),
  ('13:37', '15:33', 142),
  ('15:50', '18:45', 243),
  ('16:33', '18:15', 253),
  ('18:17', '21:04', 259),
  ('19:46', '21:45', 214)],
 ('FCO', 'LIS'): [('6:19', '8:13', 239),
  ('8:04', '10:59', 136),
  ('9:31', '11:43', 210),
  ('11:07', '13:24', 171),
  ('12:31', '14:02', 234),
  ('14:05', '15:47', 226),
  ('15:07', '17:21', 129),
  ('16:35', '18:56', 144),
  ('18:25', '20:34', 205),
  ('20:05', '21:44', 172)],
 ('FCO', 'MAD'): [('6:03', '8:43', 219),
  ('7:50', '10:08', 164),
  ('9:11', '10:42', 172),
  ('10:33', '13:11', 132),
  ('12:08', '14:47', 231),
  ('14:19', '17:09', 190),
  ('15:04', '17:23', 189),
  ('17:06', '20:00', 95),
  ('18:33', '20:22', 143),
  ('19:32', '21:25', 160)],
 ('LHR', 'FCO'): [('6:08', '8:06', 224),
  ('8:27', '10:45', 139),
  ('9:15', '12:14', 247),
  ('10:53', '13:36', 189),
  ('12:08', '14:59', 149),
  ('13:40', '15:38', 137),
  ('15:23', '17:25', 232),
  ('17:08', '19:08', 262),
  ('18:35', '20:28', 204),
  ('20:30', '23:11', 114)],
 ('LIS', 'FCO'): [('6:11', '8:31', 249),
  ('7:39', '10:24', 219),
  ('9:15', '12:03', 99),
  ('11:08', '13:07', 175),
  ('12:18', '14:56', 172),
  ('13:37', '15:08', 250),
  ('15:03', '16:42', 135),
  ('16:51', '19:09', 147),
  ('18:12', '20:17', 242),
  ('20:05', '22:06', 261)],
 ('MAD', 'FCO'): [('6:05', '8:32', 174),
  ('8:25', '10:34', 157),
  ('9:42', '11:32', 169),
  ('11:01', '12:39', 260),
  ('12:44', '14:17', 134),
  ('14:22', '16:32', 126),
  ('15:58', '18:40', 173),
  ('16:43', '19:00', 246),
  ('18:48', '21:45', 246),
  ('19:50', '22:24', 269)]}


schedule = [1,2, 3,2, 7,3, 6,3, 2,4, 5,3]

def print_sch(schedule):
    flight_id = -1
    total_price = 0
    for i in range(len(schedule)//2):
        name = people[i][0]
        origin = people[i][1]
        flight_id += 1
        going = flights[(origin, destination)][schedule[flight_id]]
        total_price += going[2]
        flight_id += 1
        returning = flights[(destination, origin)][schedule[flight_id]]
        total_price += returning[2]
        print(name, origin, going[0], going[1], going[2],
              returning[0], returning[1], returning[2])
        print(total_price)

def fitness_deap(schedule):
    flight_id = -1
    total_price = 0
    for i in range(0,6):
        origin = people[i][1]
        flight_id += 1
        going = flights[(origin, destination)][schedule[flight_id]]
        total_price += going[2]
        flight_id += 1
        returning = flights[(destination, origin)][schedule[flight_id]]
        total_price += returning[2]
    return total_price,
    
def fitness_mlrose(schedule):
    flight_id = -1
    total_price = 0
    for i in range(0,6):
        origin = people[i][1]
        flight_id += 1
        going = flights[(origin, destination)][schedule[flight_id]]
        total_price += going[2]
        flight_id += 1
        returning = flights[(destination, origin)][schedule[flight_id]]
        total_price += returning[2]
    return total_price
'''
toolbox = base.Toolbox()
creator.create('Fitnessmin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness = creator.Fitnessmin)
toolbox.register('attr_int', rd.randint, a=0, b=9)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_int, n=12)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('evaluate', fitness_deap)
toolbox.register('mate', tools.cxOnePoint)
toolbox.register('mutate', tools.mutFlipBit, indpb = .01)
toolbox.register('select', tools.selTournament, tournsize = 3)

pop = toolbox.population(n=500)
cross_prob = .7
mut_prob = .3
gen_n = 100

statistics = tools.Statistics(key = lambda individual: individual.fitness.values)
statistics.register('Max', np.max)
statistics.register('Min', np.min)
statistics.register('Mean', np.mean)
statistics.register('Std', np.std)

pop, info = algorithms.eaSimple(pop, toolbox, cross_prob, mut_prob,
                                gen_n, statistics)

best_sol = tools.selBest(pop, 1)
for individual in best_sol:
    print(individual)
    print(individual.fitness)
print_sch(individual)
'''

fitness = mlrose.CustomFitness(fitness_mlrose)
problem = mlrose.DiscreteOpt(length=12, fitness_fn=fitness, maximize=False, max_val=10)
best_sol_array, best_sol_fitness, best_sol_curve = mlrose.genetic_alg(problem, pop_size=500, mutation_prob=0.3)
print(best_sol_array, best_sol_fitness)
print_sch(best_sol_array)
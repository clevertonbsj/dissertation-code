# -*- coding: utf-8 -*-
"""
Created on Fri May 26 09:57:49 2023

@author: cbsj0
"""
import random as rd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.autograd import Variable
from timeit import default_timer as timer
import pandas as pd
def create_excel(Task_dictionary, file_path, sheetname, indexes):
    tofile = pd.DataFrame(Task_dictionary, index = indexes)
    tofile.to_excel(file_path, sheet_name=sheetname)

def read_excel(filepath, sheetname):
    dataframe = pd.read_excel(filepath, sheet_name=sheetname, index_col= 0)
    tasks = dataframe.to_dict()
    for i in range(1, len(tasks)+1):
        task = []
        for j in range(len(tasks[i])):
            task.append(tasks[i][j])
        tasks[i] = task
    return tasks
    

def create_graph(warehouse_number, delivery_number):
    '''creating the weighted adjacency matrix and squaring it'''
    adjacency_matrix = []
    matrix = warehouse_number + delivery_number + 1
    for i in range(matrix):
        adjacency_matrix.append([])     
        for j in range(matrix):
            if i==0 and j > 0 and j <= warehouse_number:
                adjacency_matrix[i].append(1)
            elif i == 0 and j == 0 or j == 0 or j > warehouse_number:
                adjacency_matrix[i].append(0)
            elif i > 0 and i <= warehouse_number:
                adjacency_matrix[i].append(0)
            elif j > warehouse_number:
                adjacency_matrix[i].append(0)
            else:
                adjacency_matrix[i].append(abs(rd.randint(1, 5)))
    '''transforming the adjacency matrix into a numpy array to create a graph'''
    adjacency_matrix = np.array(adjacency_matrix)
    Graph = nx.from_numpy_array(adjacency_matrix)
    return Graph, adjacency_matrix

def create_tasks(warehouse_number, delivery_number, Warehouses, Delivery_points):
    '''Creating empty lists and the tasks dict'''
    Load = []
    Unload = []
    Load_t0 = []
    Load_tf = []
    Loading_t = []
    Unload_t0 = []
    Unload_tf = []
    Unloading_t = []
    Task_dictionary = {}
    for i in range((warehouse_number*delivery_number)*100):        
        '''filling the lists and the dictionary'''
        Load.append(Warehouses[rd.randint(0, len(Warehouses)-1)])
        Unload.append(Delivery_points[rd.randint(0, len(Delivery_points)-1)])
        Load_t0.append(abs(rd.randint(0, 10)))
        Load_tf.append(abs(rd.randint(5, 10)))
        Loading_t.append(abs(rd.randint(1, 5)))
        Unload_t0.append(abs(rd.randint(5, 10)))
        Unload_tf.append(abs(rd.randint(5, 10)))
        Unloading_t.append(abs(rd.randint(1, 5)))
        Task_dictionary[i+1] = [Load[i], Unload[i], Load_t0[i], Load_tf[i],
                                Loading_t[i], Unload_t0[i], Unload_tf[i], 
                                Unloading_t[i]]
    temp = []
    res = dict()
    for key, val in Task_dictionary.items():
        if val not in temp:
            temp.append(val)
            res[key] = val
    
    return res

def create_queue(queue_max_size, time_slot_minutes, Task_dictionary):
    task_q = []
    task_times = []
    
    '''creating the task queue by assigning random tasks'''
    for i in range(queue_max_size):
        task_q.append(list(Task_dictionary.keys())\
                      [rd.randint(0, len(Task_dictionary)-1)])
        task_times.append(Task_dictionary[task_q[i]][5] + Task_dictionary[task_q[i]][3] + \
                          Task_dictionary[task_q[i]][2] + Task_dictionary[task_q[i]][6])
        
    '''verifying that the tasks can be completed in the given time slot'''
    while sum(task_times) > time_slot_minutes:
        task_q.pop()
        task_times.pop()
    return task_q

def order_queue_edd(task_q, elapsed_time, Task_dictionary):
    task_q_ord = []
    tasks_dues = []
    '''
    running through the list and organizing it based on the earliest delivery
    time
    '''
    for i in range(len(task_q)):
        tasks_dues.append(elapsed_time + Task_dictionary[task_q[i]][2])
    task_q_ord = [x for _,x in sorted(zip(tasks_dues, task_q))]
    return task_q_ord


def order_queue(task_queue, task_order):
    task_queue = [x for _,x in sorted(zip(task_order, task_queue))]
    return task_queue

def Do(adjacency_matrix, task_queue, tasks_dictionary, elapsed_time,
         current_position, warehouse_number, trigger):
    idle_task = []
    start_times = []
    Load = True
    changed = False
    i = 0
    curr_pos = current_position
    complete, failed, time_taken = [], [], []

    #checking if we should load the product on the AGV
    while i < len(task_queue):
        if Load == True:
            if changed == False:
                time_start_task = elapsed_time
                start_times.append(time_start_task)
            time_start = time_start_task
            curr_task = task_queue[i]
            load_pos = tasks_dictionary[curr_task][0]
            unload_pos = tasks_dictionary[curr_task][1] + warehouse_number  
            loading_time = tasks_dictionary[curr_task][4]
            unloading_time = tasks_dictionary[curr_task][7]
            load_start = tasks_dictionary[curr_task][2] + time_start_task
            load_end = tasks_dictionary[curr_task][3] + load_start
            unload_start = tasks_dictionary[curr_task][5] + time_start_task 
            unload_end = tasks_dictionary[curr_task][6] + unload_start
            move_time = adjacency_matrix[curr_pos][load_pos] 
            elapsed_time += move_time
            idle = 0
            if changed == True:
                changed = False
            if changed == False:
                if trigger[i] >=9:
                    time_start_task = elapsed_time
                    start_times.append(time_start_task)
                    changed = True
        #checking if the AGV arrived before the opening of the load window
        #if it did it stays there until the opening of the window
       
            if elapsed_time < load_start:
                idle += load_start - elapsed_time
                elapsed_time = load_start
                elapsed_time += loading_time
                idle += loading_time
                curr_pos = load_pos
                Load = False
                if changed == False:
                    if trigger[i] >=5:
                        time_start_task = elapsed_time
                        start_times.append(time_start_task)
                        changed = True
                
                #checking if the AGV arrived during it's intended loading window
                #if it did the task proceeds as usual
                
            elif elapsed_time >= load_start and elapsed_time <= load_end:
                elapsed_time += loading_time
                idle += loading_time
                curr_pos = load_pos
                Load = False
                if changed == False:
                    if trigger[i] >=5:
                        time_start_task = elapsed_time
                        start_times.append(time_start_task)
                        changed = True
                
                #checking if the AGV failed to arrive during it's time window
                #if it did it returns to the neutral point (node 0) while adding a 
                #penalty to it's score
                    
            else:
                failed.append(task_queue[i])
                elapsed_time += 2 * move_time
                i += 1


        else :
            move_time = adjacency_matrix[unload_pos][curr_pos]
            elapsed_time += move_time
        #checking if the AGV arrived before the opening of the unloading window
        #if it did it stays there until the opening of the window and add
        #the time taken to complete the task and the completed tasks to their
        #respective lists
        
            if elapsed_time < unload_start:
                idle += unload_start - elapsed_time
                elapsed_time = unload_start
                elapsed_time += unloading_time
                idle += unloading_time
                curr_pos = unload_pos
                Load = True
                complete.append(task_queue[i])
                time_taken.append(elapsed_time - time_start)
                idle_task.append(idle)
                if changed == False:
                    if trigger[i] >=3:
                        time_start_task = elapsed_time
                        start_times.append(time_start_task)
                        changed = True
                i += 1
                
                
        #checking if the AGV arrived during it's intended unloading window
        #if it did the task proceeds as usual and add the time taken to 
        #complete the task and the completed tasks to their respective lists
        
            elif elapsed_time >= unload_start and elapsed_time <= unload_end:
                elapsed_time += unloading_time
                idle += unloading_time
                curr_pos = unload_pos
                Load = True
                complete.append(task_queue[i])
                time_taken.append(elapsed_time - time_start)
                idle_task.append(idle)
                if changed == False:
                    if trigger[i] >=3:
                        time_start_task = elapsed_time
                        start_times.append(time_start_task)
                        changed = True
                i += 1
                
        #checking if the AGV failed to arrive during it's time window
        #if it did it returns to the neutral point (node 0) while adding a 
        #penalty to it's score
        
            else:
                elapsed_time += 2 * move_time + loading_time +\
                    adjacency_matrix[0][load_pos]
                curr_pos = 0
                Load = True
                failed.append(task_queue[i])
                i += 1
                

    return task_queue, idle_task, elapsed_time, curr_pos, complete, failed, time_taken, start_times

def plot(complete_tasks, time_taken_per_task, idle_time_per_task, Title):
    X_axis = np.arange(len(complete_tasks))
    width = .4
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    time_tasks = {
        'Tempo levado por tarefa': tuple(time_taken_per_task),
        'Tempo em ócio na tarefa': tuple(idle_time_per_task)
        }    
    for attribute, time in time_tasks.items():
        offset = width * multiplier
        rects = ax.bar(X_axis + offset, time, width, label = attribute)
        ax.bar_label(rects, padding = 1)
        multiplier +=1
    
    ax.set_ylabel('Tempo (min)')
    ax.set_xlabel('Tarefa concluida')
    ax.set_xticks(X_axis + width/2, complete_tasks)
    ax.legend(loc = 'upper right', ncols = 2)
    ax.set_title(Title)
    ax.set_ylim(0, 60)
    plt.show()

def newplot(G1, G2, G3, Title, lim):
    X_axis = np.arange(len(G1))
    width = .2
    multiplier = 0
    fig, ax = plt.subplots(layout = 'constrained')
    means = {
        'Earliest Due Date': tuple(G1),
        'Algortimo Genético': tuple(G2),
        'Rede Neural + Earliest Due Date': tuple(G3)
        }
    for attribute, mean in means.items():
        offset = (width * multiplier)
        rects = ax.bar(X_axis+ offset, mean, width, label = attribute)
        ax.bar_label(rects, padding = 1, fontsize = 8)
        multiplier += 1.5
    

    ax.set_ylabel('Tempo (min)')
    ax.set_xticks(X_axis + width*1.5, range(1, len(G1) + 1))
    ax.legend(loc = 'upper right', ncols = 1)
    ax.set_title(Title)
    ax.set_ylim(0, lim)
    plt.show()

def newplot2(G1, G2, G3, Title, lim):
    X_axis = np.arange(len(G1))
    width = .2
    multiplier = 0
    fig, ax = plt.subplots(layout = 'constrained')
    means = {
        'Earliest Due Date': tuple(G1),
        'Algortimo Genético': tuple(G2),
        'Rede Neural + Earliest Due Date': tuple(G3)
        }
    for attribute, mean in means.items():
        offset = (width * multiplier)
        rects = ax.bar(X_axis+ offset, mean, width, label = attribute)
        ax.bar_label(rects, padding = 1, fontsize = 8)
        multiplier += 1.5
    
    ax.set_xticks(X_axis + width*1.5, range(1, len(G1) + 1))
    ax.legend(loc = 'upper right', ncols = 1)
    ax.set_title(Title)
    ax.set_ylim(0, lim)
    plt.show()
    
        
class Individual():
    def __init__(self, idle, tasks, current_position, warehouse_number, adjacency_matrix,
                 tasks_dictionary, elapsed_time, generation = 0):
        self.idle = idle
        self.tasks = tasks
        self.score_evaluation = 0
        self.generation = generation
        self.current_position = current_position
        self.warehouse_number = warehouse_number
        self.adjacency_matrix = adjacency_matrix
        self.tasks_dictionary = tasks_dictionary
        self.elapsed_time = elapsed_time
        self.chromossome = []
        
                
        #creating the initial chromossome with a random task order
        for i in range(len(tasks)):
            self.chromossome.append(i)
        
        for i in range(len(tasks)): #randomly swapping chromossome positions
            rand = abs(rd.randint(0, len(tasks)-1))
            self.chromossome[i], self.chromossome[rand] = \
                self.chromossome[rand], self.chromossome[i]
    #creating the fitness function
    def fitness(self, trigger):
        idle = 0
        Load = True
        i = 0
        task_queue = [x for _,x in sorted(zip(self.chromossome, self.tasks))]
        curr_pos = self.current_position
        elapsed_time = self.elapsed_time
        failed = []
        changed = False

        while i < len(task_queue): #keeps the function running while going through task queue
           # print(trigger,i)
            if changed == False:
                time_start_task = self.elapsed_time    
            curr_task = task_queue[i]
            load_pos = self.tasks_dictionary[curr_task][0]
            unload_pos = self.tasks_dictionary[curr_task][1] + self.warehouse_number  
            loading_time = self.tasks_dictionary[curr_task][4]
            unloading_time = self.tasks_dictionary[curr_task][7]
            load_start = self.tasks_dictionary[curr_task][2] + time_start_task
            load_end = self.tasks_dictionary[curr_task][3] + load_start
            unload_start = self.tasks_dictionary[curr_task][5] + time_start_task
            unload_end = self.tasks_dictionary[curr_task][6] + unload_start
            if changed == True:
                changed = False
            if changed == False:
                if trigger[i] >=9:
                    time_start_task = elapsed_time
                    changed = True

            #checking if we should load the product on the AGV
            
            if Load == True:
                move_time = self.adjacency_matrix[curr_pos][load_pos] 
                elapsed_time += move_time
            
                #checking if the AGV arrived before the opening of the load window
                #if it did it stays there until the opening of the window
                
                if elapsed_time < load_start:
                    idle += load_start - elapsed_time
                    elapsed_time = load_start
                    elapsed_time += loading_time
                    idle += loading_time
                    curr_pos = load_pos
                    Load = False
                    if changed == False:
                        if trigger[i] >=5:
                            time_start_task = elapsed_time
                            changed = True
                
                #checking if the AGV arrived during it's intended loading window
                #if it did the task proceeds as usual
                
                elif elapsed_time >= load_start and elapsed_time <= load_end:
                    elapsed_time += loading_time
                    idle += loading_time
                    curr_pos = load_pos
                    Load = False
                    if changed == False:
                        if trigger[i] >=5:
                            time_start_task = elapsed_time
                            changed = True
                
                #checking if the AGV failed to arrive during it's time window
                #if it did it returns to the neutral point (node 0) while adding a 
                #penalty to it's score
                
                else:
                    elapsed_time += 2 * move_time
                    i += 1
                    failed.append(curr_task)


            else:
                move_time = self.adjacency_matrix[unload_pos][curr_pos]
                elapsed_time += move_time
                #checking if the AGV arrived before the opening of the unloading window
                #if it did it stays there until the opening of the window and add
                #the time taken to complete the task and the completed tasks to their
                #respective lists
                
                if elapsed_time < unload_start:
                    idle += unload_start - elapsed_time
                    elapsed_time = unload_start
                    elapsed_time += unloading_time
                    idle += unloading_time
                    curr_pos = unload_pos
                    Load = True
                    if changed == False:
                        if trigger[i] >=3:
                            time_start_task = elapsed_time
                            changed = True
                    i += 1
                    
                #checking if the AGV arrived during it's intended unloading window
                #if it did the task proceeds as usual and add the time taken to 
                #complete the task and the completed tasks to their respective lists
                    
                elif elapsed_time >= unload_start and elapsed_time <= unload_end:
                    elapsed_time += unloading_time
                    idle += unloading_time
                    curr_pos = unload_pos
                    Load = True
                    if changed == False:
                        if trigger[i] >=3:
                            time_start_task = elapsed_time
                            changed = True
                    i += 1
                    
                #checking if the AGV failed to arrive during it's time window
                #if it did it returns to the neutral point (node 0) while adding a 
                #penalty to it's score
                
                else:
                    elapsed_time += 2 * move_time + loading_time +\
                        self.adjacency_matrix[0][load_pos]
                    i += 1
                    curr_pos = 0
                    Load = True
                    failed.append(curr_task)
                    
        for task in failed:
            idle += 99
        self.score_evaluation = elapsed_time + idle

  #creating the crossover function
    def crossover(self, other_individual):
        size = len(self.chromossome)
        
        #choosing random start and end positions for the crossover
        start, end = sorted([rd.randrange(size) for _ in range(2)])
        
        #replicating the chromossome into the children
        child1, child2 = [-1] * size, [-1] * size #signalling empty positions
        child1_inh, child2_inh = [], [] #lists for the inherited genes
        for i in range(start, end+1):
            child1[i] = self.chromossome[i]
            child2[i] = other_individual.chromossome[i]
            child1_inh.append(self.chromossome[i])
            child2_inh.append(other_individual.chromossome[i])
        
        #doing the actual crossover
        curr_p1_pos, curr_p2_pos = 0, 0
        fixed_pos = list(range(start, end+1))
        
        i=0
        while i < size:
            if i in fixed_pos:
                i+=1
                continue
            curr_gene1  = child1[i] 
            if curr_gene1 == -1: #current gene of the first child is empty
                p2_trait = other_individual.chromossome[i]
                while p2_trait in child1_inh: #checking if the current gene is already inherited
                    curr_p2_pos += 1
                    if curr_p2_pos >= size:
                        break
                    p2_trait = other_individual.chromossome[curr_p2_pos]
                    
                child1[i] = p2_trait
                child1_inh.append(p2_trait)
            curr_gene2 = child2[i]
            if curr_gene2 == -1: #current gene of the second child is empty
                p1_trait = self.chromossome[i]
                while p1_trait in child2_inh: #checking if the current gene is already inherited
                    curr_p1_pos += 1
                    if curr_p1_pos >= size:
                        break
                    p1_trait = self.chromossome[curr_p1_pos]
                child2[i] = p1_trait
                child2_inh.append(p1_trait)
            
            i+=1
        children = [Individual(self.idle, self.tasks, self.current_position, 
                               self.warehouse_number, self.adjacency_matrix, 
                               self.tasks_dictionary, self.elapsed_time, self.generation+1),
                    Individual(self.idle, self.tasks, self.current_position, 
                               self.warehouse_number, self.adjacency_matrix, 
                               self.tasks_dictionary, self.elapsed_time, self.generation+1)]
        children[0].chromossome = child1
        children[1].chromossome = child2
        return children
    
    #creating the mutation function
    def mutation(self, rate):
        for i in range(len(self.chromossome)):
            if rd.random() < rate: #verifying if mutation occurs
                rand = abs(rd.randint(0, len(self.chromossome)-1)) 
                self.chromossome[i], self.chromossome[rand] = \
                    self.chromossome[rand], self.chromossome[i] #swapping random positions of the chromossome as a mutation
        return self

class GeneticAlgorithm():
    def __init__(self, population_size):
        self.population_size = population_size
        self.population = []
        self.generation = 0
        self.best_sol = None
        self.list_of_solutions = []
        
    #creating a function to initialize the population
    def init_pop(self, idle, tasks, current_position, warehouse_number, 
                 adjacency_matrix, tasks_dictionary, elapsed_time):
        for i in range(self.population_size):
            self.population.append(Individual(idle, tasks, current_position, 
                                              warehouse_number, adjacency_matrix, 
                                              tasks_dictionary, elapsed_time))
        self.best_sol = self.population[0]
    
    #creating the best individual function
    def best_individual(self, individual):
        if individual.score_evaluation < self.best_sol.score_evaluation:
            self.best_sol = individual
            
    #creating a function to order the population based on their scores
    def order_population(self):
        self.population = sorted(self.population, key = lambda population: \
                                 population.score_evaluation)
        
        
    #creating a function to do the sum of the evaluation
    def sum_evaluation(self):
        sum = 0 
        for individual in self.population:
            sum += individual.score_evaluation
        return sum
    
    #creating the function to select the parents
    def select_parents(self, sum_evaluation):
        parent = -1
        rand = rd.random() * sum_evaluation
        sum = 0
        i = 0
        while i < len(self.population) and sum < rand:
            sum += self.population[i].score_evaluation
            parent += 1
            i += 1
        return parent
    
    #creating the function to visualize the generation


    #creating the solving function
    def solve(self, mutation_probability, number_of_generations, idle, tasks,
              curr_pos, warehouse_number, adjacency_matrix, tasks_dictionary, 
              elapsed_time, trigger):
        
        self.init_pop(idle, tasks, curr_pos, warehouse_number, adjacency_matrix,
                      tasks_dictionary, elapsed_time)
        for individual in self.population:
            individual.fitness(trigger)
        self.order_population()
        for generation in range(number_of_generations):
            sum = self.sum_evaluation()
            new_population = []
            for individual in range(0, self.population_size, 2):
                p1 = self.select_parents(sum)
                p2 = self.select_parents(sum)
                children = self.population[p1].crossover(self.population[p2])
                new_population.append(children[0])
                new_population.append(children[1])
            self.population = new_population
            for individual in self.population:
                individual.fitness(trigger)
            self.order_population()
            best = self.population[0]
            self.list_of_solutions.append(best.score_evaluation)
            self.best_individual(best)
            #print(best.chromossome, best.score_evaluation, best.generation)
        #print('Best Solution - Generation', self.best_sol.generation,
        #      'Score', self.best_sol.score_evaluation,
        #      'Chromossome', self.best_sol.chromossome)
        return self.best_sol.chromossome
                
class brain(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(brain, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.input_size*3)
        self.fc2 = nn.Linear(self.input_size*3, self.input_size*2)
        self.fc3 = nn.Linear(self.input_size*2, self.input_size)
        self.fc4 = nn.Linear(self.input_size, round(self.output_size*.5))
        self.fc5 = nn.Linear(round(self.output_size*.5), round(self.output_size*.75))
        self.fc6 = nn.Linear(round(self.output_size*.75), self.output_size)
        
        
    def forward(self, state):
        a = func.relu(self.fc1(state))
        a = func.relu(self.fc2(a))
        a = func.relu(self.fc3(a))
        a = func.relu(self.fc4(a))
        a = func.relu(self.fc5(a))
        a = func.relu(self.fc6(a))
        bias = func.sigmoid(a)
        return bias

class ReplayMem(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = list(zip(*rd.sample(self.memory, batch_size)))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


class Dqn():
    
    def __init__(self, task_queue, task_dict, adjacency_matrix, num_load_pt, 
                 gamma, input_vector, bias, current_position, loss):
        self.input_vector = []
        self.current_position = current_position
        self.input_vector = input_vector
        self.bias = bias
        self.elapsed_time = 0
        self.idle_time = 0
        self.gamma = gamma
        self.reward_window = []
        self.model = brain(len(self.input_vector), len(self.bias))
        self.memory = ReplayMem(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = .001)
        self.last_state = torch.Tensor(len(self.input_vector)).unsqueeze(0)
        self.last_bias = self.bias
        self.last_reward = 0
        self.loss = loss
    
    def create_bias(self, state):
        bias = self.model(state)
        return bias
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_bias):
        bias = self.model(batch_state).squeeze(1)
        next_bias = self.model(batch_next_state).detach()
        tgt = self.gamma * next_bias + batch_reward
        td_loss = func.smooth_l1_loss(bias, tgt)
        self.loss.append(td_loss.item())
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph = True)
        self.optimizer.step()

    def update(self, reward, last_state, loss):
        new_state = torch.Tensor(last_state).float().unsqueeze(0)
        checker = [1.] * len(self.last_bias)
        if self.last_bias == checker:
            self.memory.push((self.last_state, new_state, torch.Tensor(self.last_bias).unsqueeze(0), torch.Tensor([self.last_reward]).unsqueeze(0)))
        else:
            self.memory.push((self.last_state, new_state, torch.Tensor(self.last_bias), torch.Tensor([self.last_reward]).unsqueeze(0)))
        self.bias = self.create_bias(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_bias = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_bias)
        self.last_bias = self.bias
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return self.bias
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)

    def save(self):
        torch.save({'state_dict' : self.model.state_dict(), 
                    'optimizer' : self.optimizer.state_dict()},
                   'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print('Loading...')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('done')
        else:
            print('no save file found')

def check_brain():
    return os.path.isfile('last_brain.pth')

def create_queue_ordered_unbiased(Task_dictionary, max_task_number, time_slot_duration):
    tasks = []
    scores = []
    task_times = []
    for i in range(max_task_number):
        rand_task = rd.randint(1, len(Task_dictionary))
        tasks.append(rand_task)
        t_time_l = Task_dictionary[rand_task][2] + Task_dictionary[rand_task][3]
        t_time_u = Task_dictionary[rand_task][5] + Task_dictionary[rand_task][4]
        task_times.append(t_time_l + t_time_u)
    tasks = [ x for _,x in sorted(zip(task_times, tasks))]
    task_times.sort()
    while sum(task_times) > time_slot_duration:
        del tasks[-1], task_times[-1]
    for i in range(len(tasks)):
        scores.append(task_times[i]/sum(task_times))
    scores.sort(reverse=True)
    return tasks, scores

def order_queue_biased(Task_dictionary, task_queue, bias):
    temp_scores = []
    task_times = []
    task_scores = [1] * len(task_queue)
    tasks = task_queue
    for task in tasks:
        t_time_l = Task_dictionary[task][2] + Task_dictionary[task][3]
        t_time_u = Task_dictionary[task][5] + Task_dictionary[task][4]
        task_times.append(t_time_l + t_time_u)
    tasks = [ x for _,x in sorted(zip(task_times, tasks))]
    bias = [ x for _,x in sorted(zip(task_times, bias))]
    task_times.sort()
    for i in range(len(task_queue)):
        temp_scores.append((task_times[i]/max(task_times)))
    temp_scores.sort(reverse=True)
    for i in range(len(task_queue)):
        task_scores[i] = temp_scores[i] + bias[i]
    tasks = [x for _,x in sorted(zip(task_scores, tasks), reverse= True)]
    bias = [x for _,x in sorted(zip(task_scores, bias), reverse=True)]
    task_scores.sort(reverse=True)
    return tasks, bias, task_scores

def order_queue_biased_ga(task_queue, bias, mut_prob, n_gen, idle_time, start_pos, w, matrix, 
                          Tasks, ela_time, trigger, pop_size):
    task_order = GeneticAlgorithm(pop_size).solve(mut_prob, n_gen, idle_time, task_queue, start_pos, w, matrix,
                          Tasks, ela_time, trigger)
    task_queue = [x for _,x in sorted(zip(task_order, task_queue))]
    bias = [x for _,x in sorted(zip(task_order, bias))]
    task_scores = []
    for i in range(len(task_queue)):
        task_scores.append(i+1)
    for i in range(len(task_scores)):
        task_scores[i] = task_scores[i]/max(task_scores)
    task_scores.sort(reverse=True)
    for i in range(len(task_scores)):
        task_scores[i] += bias[i]
    task_queue = [x for _,x in sorted(zip(task_scores, task_queue), reverse= True)]
    bias = [x for _,x in sorted(zip(task_scores, bias), reverse=True)]
    task_scores.sort(reverse=True)
    return task_queue, bias, task_scores

def order_queue_biased_pure(task_queue, bias):
    task_queue = [x for _,x in sorted(zip(bias, task_queue), reverse= True)]
    bias = sorted(bias, reverse=True)
    return task_queue, bias

def add_random_task(task_queue, Task_dictionary, task_scores, bias, trigger):
    rand_task = rd.choice(list(Task_dictionary))
    trigger.append(rd.randint(1, 10))
    task_queue.append(rand_task)
    bias = bias.tolist()
    bias = bias[0]
    bias.append(1)
    task_queue, bias, task_scores = order_queue_biased(Task_dictionary, task_queue, bias)
    return task_queue, bias, task_scores, trigger

def add_random_task_ga(task_queue, Task_dictionary, task_scores, bias, trigger,
                       mut_prob, n_gen, idle_time, start_pos, w, matrix, 
                       ela_time, pop_size):
    rand_task = rd.choice(list(Task_dictionary))
    trigger.append(rd.randint(1, 10))
    task_queue.append(rand_task)
    bias = bias.tolist()
    bias = bias[0]
    bias.append(1)
    task_queue, bias, task_scores = order_queue_biased_ga(task_queue, bias, 
                                                          mut_prob, n_gen, 
                                                          idle_time, start_pos, 
                                                          w, matrix, Task_dictionary,
                                                          ela_time, trigger,
                                                          pop_size)
    return task_queue, bias, task_scores, trigger

def add_random_task_pure(task_queue, Task_dictionary, bias, trigger):
    rand_task = rd.randint(1, len(Task_dictionary))
    trigger.append(rd.randint(1, 10))
    task_queue.append(rand_task)
    bias = bias.tolist()
    bias = bias[0]
    bias.append(1)
    task_queue = [x for _,x in sorted(zip(bias, task_queue), reverse= True)]
    bias = sorted(bias, reverse=True)
    return task_queue, bias, trigger

def check_time(elapsed_time, max_window):
    if elapsed_time >= max_window:
        return True

def create_input(task_queue, task_dict, bias, current_position, elapsed_time, 
                 adjacency_matrix, warehouse_number):
    #task_load_move = []
    task_load_start = []
    task_load_end = []
    #task_unload_move = []
    task_unload_start = []
    task_unload_end = []
    input_v = []
    for i in range(len(task_queue)):
        task = task_queue[i]
        #load_pos = task_dict[task][0]
        #unload_pos = task_dict[task][1] + warehouse_number
        #task_load_move.append(adjacency_matrix[current_position][load_pos])
        task_load_start.append(task_dict[task][2] + elapsed_time)
        task_load_end.append(task_dict[task][3] + task_load_start[i])
        #task_unload_move.append(adjacency_matrix[unload_pos][load_pos])
        task_unload_start.append(task_dict[task][5] + task_load_end[i])
        task_unload_end.append(task_dict[task][6] + task_unload_start[i])
    for i in range(len(task_queue)):
        #input_v.append(task_queue[i])
        #input_v.append(task_load_move[i]/max(task_load_move))
        input_v.append(task_load_start[i]/max(task_load_start))
        input_v.append(task_load_end[i]/max(task_load_end))
        #input_v.append(task_unload_move[i]/max(task_unload_move))
        input_v.append(task_unload_start[i]/max(task_unload_start))
        input_v.append(task_unload_end[i]/max(task_unload_end))
        #input_v.append(bias[i])
    #input_v.append(current_position)
    return input_v

def Test(adjacency_matrix, task_queue, tasks_dictionary, elapsed_time, idling, 
         current_position, warehouse_number, complete, failed, changed, trigger, time_start_task):
    i = 0
    trigger2 = trigger
    idle = idling
    Load = True
    if changed == False:
        time_start_task = elapsed_time
    curr_pos = current_position
    curr_task = task_queue[0]
    load_pos = tasks_dictionary[curr_task][0]
    unload_pos = tasks_dictionary[curr_task][1] + warehouse_number  
    loading_time = tasks_dictionary[curr_task][4]
    unloading_time = tasks_dictionary[curr_task][7]
    load_start = tasks_dictionary[curr_task][2] + time_start_task
    load_end = tasks_dictionary[curr_task][3] + load_start
    unload_start = tasks_dictionary[curr_task][5] + load_end 
    unload_end = tasks_dictionary[curr_task][6] + unload_start
    if changed == True:
        changed = False
    if changed == False:
        if trigger2[0] >=9:
            time_start_task = elapsed_time
            del trigger2[0]
            changed = True

    #checking if we should load the product on the AGV
    while i <1:
        if Load == True:
            move_time = adjacency_matrix[curr_pos][load_pos] 
            elapsed_time += move_time
        
        #checking if the AGV arrived before the opening of the load window
        #if it did it stays there until the opening of the window
       
            if elapsed_time < load_start:
                idle += load_start - elapsed_time
                elapsed_time = load_start
                elapsed_time += loading_time
                idle += loading_time
                curr_pos = load_pos
                Load = False
                if changed == False:
                    if trigger2[0] >=5:
                        time_start_task = elapsed_time
                        del trigger2[0]
                        changed = True
            
            #checking if the AGV arrived during it's intended loading window
            #if it did the task proceeds as usual
           
            elif elapsed_time >= load_start and elapsed_time <= load_end:
                    elapsed_time += loading_time
                    idle += loading_time
                    curr_pos = load_pos
                    Load = False
                    if changed == False:
                        if trigger2[0] >=5:
                            time_start_task = elapsed_time
                            del trigger2[0]
                            changed = True
        
         #checking if the AGV failed to arrive during it's time window
         #if it did it returns to the neutral point (node 0) while adding a 
         #penalty to it's score
         
            else:
                idle += 99
                failed.append(task_queue[0])
                elapsed_time += 2 * move_time
                del task_queue[0]
                break
        else :
            move_time = adjacency_matrix[unload_pos][curr_pos]
            elapsed_time += move_time
        #checking if the AGV arrived before the opening of the unloading window
        #if it did it stays there until the opening of the window and add
        #the time taken to complete the task and the completed tasks to their
        #respective lists
        
            if elapsed_time < unload_start:
                idle += unload_start - elapsed_time
                elapsed_time = unload_start
                elapsed_time += unloading_time
                idle += unloading_time
                curr_pos = unload_pos
                i += 1
                Load = True
                complete.append(task_queue[0])
                del task_queue[0]
                if changed == False:
                    if trigger2[0] >=3:
                        time_start_task = elapsed_time
                        del trigger2[0]
                        changed = True
            
                
                
        #checking if the AGV arrived during it's intended unloading window
        #if it did the task proceeds as usual and add the time taken to 
        #complete the task and the completed tasks to their respective lists
        
            elif elapsed_time >= unload_start and elapsed_time <= unload_end:
                elapsed_time += unloading_time
                idle += unloading_time
                curr_pos = unload_pos
                i += 1
                Load = True
                complete.append(task_queue[0])
                del task_queue[0]
                if changed == False:
                    if trigger2[0] >=3:
                        time_start_task = elapsed_time
                        del trigger2[0]
                        changed = True
            
                
        #checking if the AGV failed to arrive during it's time window
        #if it did it returns to the neutral point (node 0) while adding a 
        #penalty to it's score
        
            else:
                elapsed_time += 2 * move_time + loading_time +\
                    adjacency_matrix[0][load_pos]
                i += 1
                curr_pos = 0
                Load = True
                failed.append(task_queue[0])
                del task_queue[0]
    
    if changed == False:
        del trigger2[0]
    return task_queue, idle, elapsed_time, curr_pos, complete, failed, changed, trigger2, time_start_task

            
def Train_model(task_q, task_scores, adjacency_matrix, trainning_duration, Task_dictionary, 
                warehouse_number, gamma, trigger, loss):
    elapsed_time = 0
    idling = 0
    current_position = 0
    test_trigger = trigger
    complete, failed = [], []
    bias = [1] * len(task_q)
    input_vector = create_input(task_q, Task_dictionary, bias, current_position, 
                                elapsed_time, adjacency_matrix, warehouse_number)
    last_reward = 0
    scores = []
    brain = Dqn(task_q, Task_dictionary, adjacency_matrix, warehouse_number, 
        gamma, input_vector, bias, current_position, loss)
    if check_brain():
        brain.load()
    t_d = trainning_duration * 3600
    t_f = 0
    t_i = timer()
    changed = False
    time_start_task = 0
    while t_f - t_i <= t_d:
        last_state = input_vector
        bias = brain.update(last_reward, last_state, loss)
        scores.append(brain.score())
        last_idle = idling
        time_start = elapsed_time
        last_len_complete = len(complete)
        last_len_failed = len(failed)
        task_q, idling, elapsed_time, current_position, complete, failed, changed, test_trigger, time_start_task =\
        Test(adjacency_matrix, task_q, Task_dictionary, elapsed_time, idling, 
             current_position, warehouse_number, complete, failed, changed, test_trigger, time_start_task)
        bias = bias[0][0:-1]
        bias = bias.unsqueeze(0)

        task_q, bias, task_scores, test_trigger = add_random_task(task_q, Task_dictionary,
                                                    task_scores, bias, test_trigger)
        task_q, bias, task_scores = order_queue_biased(Task_dictionary, task_q, bias)
        input_vector = create_input(task_q, Task_dictionary, bias, current_position, 
                                    elapsed_time, adjacency_matrix, warehouse_number)
        if len(complete) > last_len_complete:
            last_reward += 10
        if len(failed) > last_len_failed:
            last_reward -= 10
        last_reward -= ((elapsed_time - time_start)/100) - ((idling - last_idle)/ 10)
        t_f = timer()
    brain.save()
    return scores, last_reward, brain.loss
    
def create_bias(task_queue, Task_dictionary, current_position, elapsed_time,
                adjacency_matrix, warehouse_number, gamma, last_reward,
                scores, loss):
    bias = [1] * len(task_queue)
    input_vector = create_input(task_queue, Task_dictionary, bias, 
                                current_position, elapsed_time, 
                                adjacency_matrix, warehouse_number)
    brain = Dqn(task_queue, Task_dictionary, adjacency_matrix, warehouse_number, 
        gamma, input_vector, bias, current_position, loss)
    if check_brain():
        brain.load()
    bias = brain.update(last_reward, input_vector, loss)
    scores.append(brain.score())
    brain.save()
    return bias, scores

def Train_model_ga(adjacency_matrix, trainning_duration, Task_dictionary, 
                   warehouse_number, gamma, n_gen, pop_size, mut_prob):
    elapsed_time = 0
    idling = 0
    current_position = 0
    test_trigger = []
    complete, failed = [], []
    task_q, task_scores = create_queue_ordered_unbiased(Task_dictionary, 20, 240)
    bias = [1] * len(task_q)
    for i in range(len(task_q)):
        test_trigger.append(rd.randint(1, 10))
    input_vector = create_input(task_q, Task_dictionary, bias, current_position, 
                                elapsed_time, adjacency_matrix, warehouse_number)
    last_reward = 0
    scores = []
    brain = Dqn(task_q, Task_dictionary, adjacency_matrix, warehouse_number, 
        gamma, input_vector, bias, current_position)
    t_d = trainning_duration * 3600
    t_f = 0
    t_i = timer()
    changed = False
    time_start_task = 0
    while t_f - t_i <= t_d:
        last_state = input_vector
        bias = brain.update(last_reward, last_state)
        scores.append(brain.score())
        last_idle = idling
        time_start = elapsed_time
        last_len_complete = len(complete)
        last_len_failed = len(failed)
        task_q, idling, elapsed_time, current_position, complete, failed, changed, test_trigger, time_start_task =\
        Test(adjacency_matrix, task_q, Task_dictionary, elapsed_time, idling, 
             current_position, warehouse_number, complete, failed, changed, test_trigger, time_start_task)
        bias = bias[0][0:-1]
        bias = bias.unsqueeze(0)
        
        
        task_q, bias, task_scores, test_trigger = add_random_task_ga(task_q, Task_dictionary,
                                                                     task_scores, bias, test_trigger,
                                                                     mut_prob, n_gen, idling, 
                                                                     current_position, warehouse_number, 
                                                                     adjacency_matrix, elapsed_time, pop_size)
        task_q, bias, task_scores = order_queue_biased_ga(task_q, bias, mut_prob,
                                                          n_gen, idling, current_position,
                                                          warehouse_number, adjacency_matrix,
                                                          Task_dictionary, elapsed_time, test_trigger, pop_size)
                
        input_vector = create_input(task_q, Task_dictionary, bias, current_position, 
                                    elapsed_time, adjacency_matrix, warehouse_number)
        if len(complete) > last_len_complete:
            last_reward += 10
        if len(failed) > last_len_failed:
            last_reward -= 10
        last_reward -= ((elapsed_time - time_start)/100) - ((idling - last_idle)/ 10)
        t_f = timer()
    #brain.save()
    return scores, last_reward
    
def Train_model_pure(adjacency_matrix, trainning_duration, Task_dictionary, warehouse_number, gamma, trigger, loss):
    elapsed_time = 0
    idling = 0
    current_position = 0
    test_trigger = trigger
    complete, failed = [], []
    task_q, task_scores = create_queue_ordered_unbiased(Task_dictionary, 20, 240)
    bias = [1] * len(task_q)
    input_vector = create_input(task_q, Task_dictionary, bias, current_position, 
                                elapsed_time, adjacency_matrix, warehouse_number)
    last_reward = 0
    scores = []
    brain = Dqn(task_q, Task_dictionary, adjacency_matrix, warehouse_number, 
        gamma, input_vector, bias, current_position, loss)
    t_d = trainning_duration * 3600
    t_f = 0
    t_i = timer()
    changed = False
    time_start_task = 0
    while t_f - t_i <= t_d:
        last_state = input_vector
        bias = brain.update(last_reward, last_state, loss)
        scores.append(brain.score())
        last_idle = idling
        time_start = elapsed_time
        last_len_complete = len(complete)
        last_len_failed = len(failed)
        task_q, idling, elapsed_time, current_position, complete, failed, changed, test_trigger, time_start_task =\
        Test(adjacency_matrix, task_q, Task_dictionary, elapsed_time, idling, 
             current_position, warehouse_number, complete, failed, changed, test_trigger, time_start_task)
        bias = bias[0][0:-1]
        bias = bias.unsqueeze(0)

        task_q, bias, test_trigger = add_random_task_pure(task_q, Task_dictionary,
                                                          bias, test_trigger)
        task_q, bias = order_queue_biased_pure(task_q, bias)
        input_vector = create_input(task_q, Task_dictionary, bias, current_position, 
                                    elapsed_time, adjacency_matrix, warehouse_number)
        if len(complete) > last_len_complete:
            last_reward += 10
        if len(failed) > last_len_failed:
            last_reward -= 10
        last_reward -= ((elapsed_time - time_start)/100) - ((idling - last_idle)/ 10)
        t_f = timer()
    #brain.save()
    return scores, last_reward, brain.loss

def plot_losses(losses, name):
    plt.plot(losses)
    plt.title(name)
    plt.xlabel('')
    plt.ylabel('')
    plt.show()
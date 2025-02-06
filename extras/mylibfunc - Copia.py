# -*- coding: utf-8 -*-
"""
Created on Fri May 26 09:57:49 2023

@author: cbsj0
"""
from math import floor, ceil
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
            if j==0 and i > 0 and i <= warehouse_number:
                adjacency_matrix[i].append(1)
            elif j>=i:
                adjacency_matrix[i].append(0)
            elif j == 0:
                adjacency_matrix[i].append(0)
            else:
                adjacency_matrix[i].append(abs(rd.randint(1, 5)))
    
    for i in range(matrix):    
        for j in range(matrix):
            if j > i:
                adjacency_matrix[i][j] = adjacency_matrix[j][i]
                
    '''transforming the adjacency matrix into a numpy array to create a graph'''
    adjacency_matrix = np.array(adjacency_matrix)
    Graph = nx.from_numpy_array(adjacency_matrix)
    return Graph, adjacency_matrix

def create_tasks(warehouse_number, delivery_number, Warehouses, Delivery_points, A = 8, B = 5):
    '''Creating empty lists and the tasks dict'''
    Load = []
    Unload = []
    Load_t0 = []
    Load_tf = []
    Loading_t = []
    Unload_t0 = []
    Unload_tf = []
    Unloading_t = []
    Type = []
    Task_dictionary = {}
    for i in range((warehouse_number*delivery_number)\
                   +abs(rd.randint(0, warehouse_number*delivery_number))):        
        '''filling the lists and the dictionary'''
        aux = round(rd.random() * 10, 2)
        Load.append(Warehouses[rd.randint(0, len(Warehouses)-1)])
        Unload.append(Delivery_points[rd.randint(0, len(Delivery_points)-1)])
        Load_t0.append(abs(rd.randint(0, 10)))
        Load_tf.append(abs(rd.randint(5, 10)))
        Loading_t.append(abs(rd.randint(1, 5)))
        Unload_t0.append(abs(rd.randint(5, 10)))
        Unload_tf.append(abs(rd.randint(5, 10)))
        Unloading_t.append(abs(rd.randint(1, 5)))
        if aux >= A:
            Type.append('A')
        elif A > aux >= B:
            Type.append('B')
        else:
            Type.append('C')
        Task_dictionary[i+1] = [Load[i], Unload[i], Load_t0[i], Load_tf[i],
                                Loading_t[i], Unload_t0[i], Unload_tf[i], 
                                Unloading_t[i], Type[i]]
        
    aux = list({k: v for k, v in Task_dictionary.items() if v[-1] == 'A'}.keys())
    if len(aux) < len(Task_dictionary) * .15:
        diff = len(Task_dictionary) * .15 - len(aux)
        eligible_ts = list({k: v for k, v in Task_dictionary.items() if v[-1] != 'A'}.keys())
        chosen_ts = rd.choices(eligible_ts, k = ceil(diff))
        for task in chosen_ts:
            Task_dictionary[task][-1] = "A"
            
    return Task_dictionary

def create_queue(queue_max_size, time_slot_minutes, Task_dictionary, A = 9, B = 6.5):
    task_q = []
    task_times = []
    a_tasks = list({k: v for k, v in Task_dictionary.items() if v[-1] == 'A'}.keys())
    b_tasks = list({k: v for k, v in Task_dictionary.items() if v[-1] == 'B'}.keys())
    c_tasks = list({k: v for k, v in Task_dictionary.items() if v[-1] == 'C'}.keys())
    '''creating the task queue by assigning random tasks'''
    for i in range(queue_max_size):
        aux = round(rd.random() * 10, 2)
        if aux >= A:
            task_q.append(rd.choices(c_tasks, k = 1))
        elif A > aux >= B:
            task_q.append(rd.choices(b_tasks, k = 1))
        else:
            task_q.append(rd.choices(a_tasks, k = 1))
        task = task_q[-1][0]
        task_times.append(Task_dictionary[task][5] + Task_dictionary[task][3] + \
                          Task_dictionary[task][2] + Task_dictionary[task][6])
        
    '''verifying that the tasks can be completed in the given time slot'''
    while sum(task_times) > time_slot_minutes:
        task_q.pop()
        task_times.pop()
        
    return task_q

def order_queue_edd(task_q, elapsed_time, Task_dictionary, capacity, n_agents):
    tasks_dues = []
    task_q_ord = []
    
    '''
    running through the list and organizing it based on the earliest delivery
    time
    '''
    for i in range(len(task_q)):
        tasks_dues.append(elapsed_time + Task_dictionary[task_q[i][0]][2])
    total_task_q_ord = [x for _,x in sorted(zip(tasks_dues, task_q))]
    
    while len(total_task_q_ord) < capacity * n_agents:
        total_task_q_ord.append('Dummy')
    
    for j in range(n_agents):
        task_q_ord = total_task_q_ord[j::n_agents]
            
    return task_q_ord

def order_queue(task_queue, task_order):
    total_task_queue = [x for _,x in sorted(zip(task_order, task_queue))]
    return total_task_queue

def Do(adjacency_matrix, non_split_tasks, tasks_dictionary, elapsed_time,
         current_position, warehouse_number, trigger, n_agents, agent_capacity):
    idle_task = []
    start_times = []
    Load = True
    changed = False
    i = 0
    curr_pos = current_position
    prev_pos = 0
    complete, failed, time_taken = [], [], []
    idle = 0
    split_size = int(len(non_split_tasks) / n_agents)
    cargo = {}
    for i in range(n_agents):
        cargo[i] = [0] * agent_capacity
       
    changed = False
    for agent in range(n_agents):
        curr_pos = current_position[agent]
        failed = []
        i = 0
        task_queue = non_split_tasks[agent * split_size : (agent + 1) * split_size]
        agent_is_full = False
        
        while i < len(task_queue): #keeps the function running while going through task queue
            if changed == False:
                time_start_task = elapsed_time
            curr_task, task_type = task_queue[i].split('.')
            curr_task = curr_task.lstrip('[')
            curr_task = curr_task.rstrip(']')                
            
            if curr_task == 'Dummy':
                load_pos = current_position[agent]
                unload_pos = 0
                loading_time = 0
                unloading_time = 0
                load_start = 0 + time_start_task
                load_end = 99 + load_start
                unload_start = 0 + time_start_task
                unload_end = 99 + unload_start
            else:
                curr_task = int(curr_task)
                load_pos = tasks_dictionary[curr_task][0]
                unload_pos = tasks_dictionary[curr_task][1] + warehouse_number  
                loading_time = tasks_dictionary[curr_task][4]
                unloading_time = tasks_dictionary[curr_task][7]
                load_start = tasks_dictionary[curr_task][2] + time_start_task
                load_end = tasks_dictionary[curr_task][3] + load_start
                unload_start = tasks_dictionary[curr_task][5] + time_start_task
                unload_end = tasks_dictionary[curr_task][6] + unload_start
            
            if changed == True:
                changed = False
            
            if changed == False:
                if trigger[floor(i/2)] >=9:
                    time_start_task = elapsed_time
                    changed = True
                  
            if task_type == 'L':
                Load = True
            else:
                Load = False
            #checking if we should load the product on the AGV

            if Load == True:
                
                if load_pos == prev_pos and task_queue[i]!=task_queue[i-1]:
                    elapsed_time += 5
                
                move_time = adjacency_matrix[curr_pos][load_pos] 
                elapsed_time += move_time
                
                #checking if the AGV arrived before the opening of the load window
                #if it did it stays there until the opening of the window
                if elapsed_time < load_start:
                    
                    idle += load_start - elapsed_time
                    elapsed_time = load_start
                    elapsed_time += loading_time
                    idle += loading_time
                    prev_pos = curr_pos
                    curr_pos = load_pos
                    
                    if not agent_is_full: #If the agent is not full Load
                        for j in  range(len(cargo[agent])):
                            if cargo[agent][j] == 0:
                                cargo[agent][j] = curr_task
                                break
                        complete.append(task_queue[i])
                        time_taken.append(elapsed_time - time_start_task)
                        idle_task.append(idle)
                                
                    else: #If the agent is full apply fail
                        failed.append(str(curr_task) + '.L')
                        
                    if changed == False:
                        if trigger[floor(i/2)] >=5:
                            time_start_task = elapsed_time
                            changed = True
                    
                    if 0 not in cargo[agent]:
                        agent_is_full = True

                    
                    i += 1
                #checking if the AGV arrived during it's intended loading window
                #if it did the task proceeds as usual

                elif elapsed_time >= load_start and elapsed_time <= load_end:
                    
                    elapsed_time += loading_time
                    idle += loading_time
                    prev_pos = curr_pos
                    curr_pos = load_pos
                    
                    if not agent_is_full: #If the agent is not full Load
                        for j in  range(len(cargo[agent])):
                            if cargo[agent][j] == 0:
                                cargo[agent][j] = curr_task
                                break
                        complete.append(task_queue[i])
                        time_taken.append(elapsed_time - time_start_task)
                        idle_task.append(idle)
                                
                    else: #If the agent is full fail
                        failed.append(str(curr_task) + '.L')
                    
                    if changed == False:
                        if trigger[floor(i/2)] >=5:
                            time_start_task = elapsed_time
                            changed = True
                    
                    if 0 not in cargo[agent]:
                        agent_is_full = True
                    
                    
                    i += 1
                    
                #checking if the AGV failed to arrive during it's time window
                #if it did it returns to the neutral point (node 0) while adding a 
                #penalty to it's score

                else:
                    elapsed_time += 2 * move_time + loading_time
                    if changed == False:
                        if trigger[floor(i/2)] >=5:
                            time_start_task = elapsed_time
                            changed = True
                    idle += 99
                    i += 1
                    failed.append(str(curr_task) + '.L')


            else:
                if unload_pos == prev_pos:
                    elapsed_time += 5
                
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
                    prev_pos = curr_pos
                    curr_pos = unload_pos
                    
                    if curr_task not in cargo[agent]:
                        failed.append(str(curr_task) + '.U')
                    else:
                        for j in range(len(cargo[agent])):
                            if cargo[agent][j] == curr_task:
                                cargo[agent][j] = 0
                                agent_is_full = False
                                break
                        complete.append(task_queue[i])
                        time_taken.append(elapsed_time - time_start_task)
                        idle_task.append(idle)
                        
                    
                    if changed == False:
                        if trigger[floor(i/2)] >=3:
                            time_start_task = elapsed_time
                            changed = True
                    
                    i += 1

                #checking if the AGV arrived during it's intended unloading window
                #if it did the task proceeds as usual and add the time taken to 
                #complete the task and the completed tasks to their respective lists

                elif elapsed_time >= unload_start and elapsed_time <= unload_end:
                    
                    elapsed_time += unloading_time
                    idle += unloading_time
                    prev_pos = curr_pos
                    curr_pos = unload_pos
                    
                    
                    if curr_task not in cargo[agent]:
                        failed.append(str(curr_task) + '.U')
                    else:
                        for j in range(len(cargo[agent])):
                            if cargo[agent][j] == curr_task:
                                cargo[agent][j] = 0
                                agent_is_full = False
                                break
                        complete.append(task_queue[i])
                        time_taken.append(elapsed_time - time_start_task)
                        idle_task.append(idle)
                        
                    
                    if changed == False:
                        if trigger[floor(i/2)] >=3:
                            time_start_task = elapsed_time
                            changed = True
                    
                    i += 1

                #checking if the AGV failed to arrive during it's time window
                #if it did it returns to the neutral point (node 0) while adding a 
                #penalty to it's score

                else:
                    elapsed_time += 2 * move_time + loading_time +\
                        adjacency_matrix[0][load_pos] + 5
                    i += 1
                    curr_pos = 0
                    failed.append(str(curr_task) + '.U')
                    
                    if changed == False:
                        if trigger[floor(i/2)] >=3:
                            time_start_task = elapsed_time
                            changed = True
                    
                    if curr_task in cargo[agent]:
                        for j in range(len(cargo[agent])):
                            if cargo[agent][j] == curr_task:
                                cargo[agent][j] = 0
                                agent_is_full = False
                                break
                    
                    

    return non_split_tasks, idle_task, elapsed_time, curr_pos, true_complete(complete), failed, time_taken, start_times
    
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
    def __init__(self, idle, task_queue, current_position, warehouse_number, adjacency_matrix,
                 tasks_dictionary, elapsed_time, capacity, n_agents, generation = 0):
        self.idle = idle
        self.tasks = task_queue
        self.score_evaluation = 0
        self.generation = generation
        self.current_position = current_position
        self.warehouse_number = warehouse_number
        self.adjacency_matrix = adjacency_matrix
        self.tasks_dictionary = tasks_dictionary
        self.elapsed_time = elapsed_time
        self.agent_capacity = capacity
        self.n_agents = n_agents
        self.cargo = []
        self.chromossome = []
        
        #filling the cargo with 0 to indicate that it's empty
        for i in range(n_agents):
            aux = [0] * capacity
            self.cargo.append(aux)
                    
        #creating the initial chromossome with a random task order
        for i in range(len(self.tasks)):
            self.chromossome.append(i)
        

        for i in range(len(task_queue)): #randomly swapping chromossome positions
            rand = abs(rd.randint(0, len(task_queue)-1))
            self.chromossome[i], self.chromossome[rand] = \
                self.chromossome[rand], self.chromossome[i]
        
    #creating the fitness function
    def fitness(self, trigger):
        idle = 0
        split_size = self.agent_capacity * 2
        total_task_queue = [x for _,x in sorted(zip(self.chromossome, self.tasks))]
        curr_pos = self.current_position
        elapsed_time = self.elapsed_time
        changed = False
        n_agents  = self.n_agents
        prev_pos = -1
        failed = []
        for agent in range(n_agents):            
            i = 0
            split_point = split_size * (agent + 1)
            
            if agent == 0:
                task_queue = total_task_queue[0:split_point]
                last_split = split_point
            else:
                task_queue = total_task_queue[last_split:split_point]
                last_split = split_point
                
            
            
            agent_is_full = False
            
            while i < len(task_queue): #keeps the function running while going through task queue
                if changed == False:
                    time_start_task = self.elapsed_time
                curr_task, task_type = task_queue[i].split('.')
                curr_task = curr_task.lstrip('[')
                curr_task = curr_task.rstrip(']')                
                
                if curr_task == 'Dummy':
                    load_pos = self.current_position
                    unload_pos = 0
                    loading_time = 0
                    unloading_time = 0
                    load_start = 0 + time_start_task
                    load_end = 99 + load_start
                    unload_start = 0 + time_start_task
                    unload_end = 99 + unload_start
                else:
                    curr_task = int(curr_task)
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
                    if trigger[floor(i/2)] >=9:
                        time_start_task = elapsed_time
                        changed = True
                      
                if task_type == 'L':
                    Load = True
                else:
                    Load = False
                #checking if we should load the product on the AGV

                if Load == True:
                    
                    if load_pos == prev_pos and task_queue[i]!=task_queue[i-1]:
                        elapsed_time += 5
                    
                    move_time = self.adjacency_matrix[curr_pos][load_pos] 
                    elapsed_time += move_time

                    #checking if the AGV arrived before the opening of the load window
                    #if it did it stays there until the opening of the window
                    
                    
                        
                    if elapsed_time < load_start:
                        
                        idle += load_start - elapsed_time
                        elapsed_time = load_start
                        elapsed_time += loading_time
                        idle += loading_time
                        prev_pos = curr_pos
                        curr_pos = load_pos
                    
                        
                        if not agent_is_full: #If the agent is not full Load
                            for j in  range(len(self.cargo[agent])):
                                if self.cargo[agent][j] == 0:
                                    self.cargo[agent][j] = curr_task
                                    break
                                    
                        else: #If the agent is full apply heavy penalty
                            idle += 99
                            failed.append(str(curr_task) + '.L')
                            
                        if changed == False:
                            if trigger[floor(i/2)] >=5:
                                time_start_task = elapsed_time
                                changed = True
                        
                        if 0 not in self.cargo[agent]:
                            agent_is_full = True

                        i += 1
                    #checking if the AGV arrived during it's intended loading window
                    #if it did the task proceeds as usual

                    elif elapsed_time >= load_start and elapsed_time <= load_end:
                        
                        elapsed_time += loading_time
                        idle += loading_time
                        prev_pos = curr_pos
                        curr_pos = load_pos
                        
                        if not agent_is_full: #If the agent is not full Load
                            for j in  range(len(self.cargo[agent])):
                                if self.cargo[agent][j] == 0:
                                    self.cargo[agent][j] = curr_task
                                    break
                                    
                        else: #If the agent is full apply heavy penalty
                            idle += 99
                            failed.append(str(curr_task) + '.L')
                        
                        if changed == False:
                            if trigger[floor(i/2)] >=5:
                                time_start_task = elapsed_time
                                changed = True
                        
                        if 0 not in self.cargo[agent]:
                            agent_is_full = True
                        
                        i += 1
                        
                    #checking if the AGV failed to arrive during it's time window
                    #if it did it returns to the neutral point (node 0) while adding a 
                    #penalty to it's score

                    else:
                        elapsed_time += 2 * move_time + loading_time
                        if changed == False:
                            if trigger[floor(i/2)] >=5:
                                time_start_task = elapsed_time
                                changed = True
                        idle += 99
                        i += 1
                        failed.append(str(curr_task) + '.L')


                else:
                    
                    if unload_pos == prev_pos:
                        elapsed_time += 5
                    
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
                        prev_pos = curr_pos
                        curr_pos = unload_pos
                        
                        if curr_task not in self.cargo[agent]:
                            failed.append(str(curr_task) + '.U')
                            idle += 99
                        else:
                            for j in range(len(self.cargo[agent])):
                                if self.cargo[agent][j] == curr_task:
                                    self.cargo[agent][j] = 0
                                    agent_is_full = False
                                    break
                        
                        if changed == False:
                            if trigger[floor(i/2)] >=3:
                                time_start_task = elapsed_time
                                changed = True
                        i += 1

                    #checking if the AGV arrived during it's intended unloading window
                    #if it did the task proceeds as usual and add the time taken to 
                    #complete the task and the completed tasks to their respective lists

                    elif elapsed_time >= unload_start and elapsed_time <= unload_end:
                        
                        elapsed_time += unloading_time
                        idle += unloading_time
                        prev_pos = curr_pos
                        curr_pos = unload_pos
                        
                        if curr_task not in self.cargo[agent]:
                            failed.append(str(curr_task) + '.U')
                            idle += 99
                        else:
                            for j in range(len(self.cargo[agent])):
                                if self.cargo[agent][j] == curr_task:
                                    self.cargo[agent][j] = 0
                                    agent_is_full = False
                                    break
                        
                        if changed == False:
                            if trigger[floor(i/2)] >=3:
                                time_start_task = elapsed_time
                                changed = True
                        i += 1

                    #checking if the AGV failed to arrive during it's time window
                    #if it did it returns to the neutral point (node 0) while adding a 
                    #penalty to it's score

                    else:
                        elapsed_time += 2 * move_time + loading_time +\
                            self.adjacency_matrix[0][load_pos] + 99
                        
                        if changed == False:
                            if trigger[floor(i/2)] >=3:
                                time_start_task = elapsed_time
                                changed = True
                        
                        if curr_task not in self.cargo[agent]:
                            failed.append(str(curr_task) + '.U')
                            idle += 99
                        else:
                            for j in range(len(self.cargo[agent])):
                                if self.cargo[agent][j] == curr_task:
                                    self.cargo[agent][j] = 0
                                    agent_is_full = False
                                    break
                        
                        idle += 99
                        i += 1
                        curr_pos = 0
                        failed.append(str(curr_task) + '.U')
        
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
                               self.tasks_dictionary, self.elapsed_time, self.agent_capacity,
                               self.n_agents, self.generation+1),
                    Individual(self.idle, self.tasks, self.current_position, 
                               self.warehouse_number, self.adjacency_matrix, 
                               self.tasks_dictionary, self.elapsed_time, self.agent_capacity,
                               self.n_agents, self.generation+1)]
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
                 adjacency_matrix, tasks_dictionary, elapsed_time, capacity, n_agents):
        for i in range(self.population_size):
            self.population.append(Individual(idle, tasks, current_position, 
                                              warehouse_number, adjacency_matrix, 
                                              tasks_dictionary, elapsed_time, capacity, n_agents))
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
              elapsed_time, trigger, capacity, n_agents):
        
        self.init_pop(idle, tasks, curr_pos, warehouse_number, adjacency_matrix,
                      tasks_dictionary, elapsed_time, capacity, n_agents)
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

def get_complete_percent(complete_tasks, task_q):
    
    L = []
    U = []
    C = []
    
    for task in complete_tasks:
        task_id, task_type = task.split('.')
        if task_type == 'L':
            L.append(task_id)
        else:
            U.append(task_id)
    
    for i in range(len(L)):
        for j in range(len(U)):
            if L[i] == U[j]:
                C.append(L[i])
                U[j] = None
                break
                
    return round(len(C)/ (len(task_q)/2), 2)
            
class selector_brain(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(master_brain, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, int(self.input_size*2))
        self.fc2 = nn.Linear(int(self.input_size*2), int(self.input_size*1.8))
        self.fc3 = nn.Linear(int(self.input_size*1.8), int(self.input_size*1.6))
        self.fc4 = nn.Linear(int(self.input_size*1.6), int(self.input_size*1.4))
        self.fc5 = nn.Linear(int(self.input_size*1.4), int(self.input_size*1.2))
        self.fc6 = nn.Linear(int(self.input_size*1.2), int(self.input_size*1))
        self.fc7 = nn.Linear(int(self.input_size*1), int(self.input_size*.8))
        self.fc8 = nn.Linear(int(self.input_size*.8), int(self.input_size*.6))
        self.fc9 = nn.Linear(int(self.input_size*.6), int(self.output_size))
        # self.fc10 = nn.Linear(int(self.output_size), int(self.output_size*1.2))
        # self.fc11 = nn.Linear(int(self.output_size*1.2), int(self.output_size*1.4))
        # self.fc12 = nn.Linear(int(self.output_size*1.4), int(self.output_size*1.6))
        # self.fc13 = nn.Linear(int(self.output_size*1.6), int(self.output_size*1.8))
        # self.fc14 = nn.Linear(int(self.output_size*1.8), int(self.output_size*2))
        # self.fc15 = nn.Linear(int(self.output_size*2), int(self.output_size*2.2))
        # self.fc16 = nn.Linear(int(self.output_size*2.2), int(self.output_size*2.4))
        # self.fc17 = nn.Linear(int(self.output_size*2.4), int(self.output_size*2.6))
        # self.fc18 = nn.Linear(int(self.output_size*2.6), int(self.output_size*2.8))
        # self.fc19 = nn.Linear(int(self.output_size*2.8), int(self.output_size*3))
        # self.fc20 = nn.Linear(int(self.output_size*3), int(self.output_size))
        
    def forward(self, state):
        a = func.relu(self.fc1(state))
        a = func.relu(self.fc2(a))
        a = func.relu(self.fc3(a))
        a = func.relu(self.fc4(a))
        a = func.relu(self.fc5(a))
        a = func.relu(self.fc6(a))
        a = func.relu(self.fc7(a))
        a = func.relu(self.fc8(a))
        a = func.relu(self.fc9(a))
        # a = func.relu(self.fc10(a))
        # a = func.relu(self.fc11(a))
        # a = func.relu(self.fc12(a))
        # a = func.relu(self.fc13(a))
        # a = func.relu(self.fc14(a))
        # a = func.relu(self.fc15(a))
        # a = func.relu(self.fc16(a))
        # a = func.relu(self.fc17(a))
        # a = func.relu(self.fc18(a))
        # a = func.relu(self.fc19(a))
        # a = func.relu(self.fc20(a))
        bias = func.sigmoid(a)
        return bias 
            
class master_brain(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(master_brain, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, int(self.input_size*2))
        self.fc2 = nn.Linear(int(self.input_size*2), int(self.input_size*1.8))
        self.fc3 = nn.Linear(int(self.input_size*1.8), int(self.input_size*1.6))
        self.fc4 = nn.Linear(int(self.input_size*1.6), int(self.input_size*1.4))
        self.fc5 = nn.Linear(int(self.input_size*1.4), int(self.input_size*1.2))
        self.fc6 = nn.Linear(int(self.input_size*1.2), int(self.input_size*1))
        self.fc7 = nn.Linear(int(self.input_size*1), int(self.input_size*.8))
        self.fc8 = nn.Linear(int(self.input_size*.8), int(self.input_size*.6))
        self.fc9 = nn.Linear(int(self.input_size*.6), int(self.output_size))
        self.fc10 = nn.Linear(int(self.output_size), int(self.output_size*1.2))
        self.fc11 = nn.Linear(int(self.output_size*1.2), int(self.output_size*1.4))
        self.fc12 = nn.Linear(int(self.output_size*1.4), int(self.output_size*1.6))
        self.fc13 = nn.Linear(int(self.output_size*1.6), int(self.output_size*1.8))
        self.fc14 = nn.Linear(int(self.output_size*1.8), int(self.output_size*2))
        self.fc15 = nn.Linear(int(self.output_size*2), int(self.output_size*2.2))
        self.fc16 = nn.Linear(int(self.output_size*2.2), int(self.output_size*2.4))
        self.fc17 = nn.Linear(int(self.output_size*2.4), int(self.output_size*2.6))
        self.fc18 = nn.Linear(int(self.output_size*2.6), int(self.output_size*2.8))
        self.fc19 = nn.Linear(int(self.output_size*2.8), int(self.output_size*3))
        self.fc20 = nn.Linear(int(self.output_size*3), int(self.output_size))
        
    def forward(self, state):
        a = func.relu(self.fc1(state))
        a = func.relu(self.fc2(a))
        a = func.relu(self.fc3(a))
        a = func.relu(self.fc4(a))
        a = func.relu(self.fc5(a))
        a = func.relu(self.fc6(a))
        a = func.relu(self.fc7(a))
        a = func.relu(self.fc8(a))
        a = func.relu(self.fc9(a))
        a = func.relu(self.fc10(a))
        a = func.relu(self.fc11(a))
        a = func.relu(self.fc12(a))
        a = func.relu(self.fc13(a))
        a = func.relu(self.fc14(a))
        a = func.relu(self.fc15(a))
        a = func.relu(self.fc16(a))
        a = func.relu(self.fc17(a))
        a = func.relu(self.fc18(a))
        a = func.relu(self.fc19(a))
        a = func.relu(self.fc20(a))
        bias = func.sigmoid(a)
        return bias   
              
class agent_brain(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(agent_brain, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, int(self.input_size*2))
        self.fc2 = nn.Linear(int(self.input_size*2), int(self.input_size*1.8))
        self.fc3 = nn.Linear(int(self.input_size*1.8), int(self.input_size*1.6))
        self.fc4 = nn.Linear(int(self.input_size*1.6), int(self.input_size*1.4))
        self.fc5 = nn.Linear(int(self.input_size*1.4), int(self.input_size*1.2))
        self.fc6 = nn.Linear(int(self.input_size*1.2), int(self.input_size*1))
        self.fc7 = nn.Linear(int(self.input_size*1), int(self.input_size*.8))
        self.fc8 = nn.Linear(int(self.input_size*.8), int(self.input_size*.6))
        self.fc9 = nn.Linear(int(self.input_size*.6), int(self.output_size))
        #self.fc10 = nn.Linear(int(self.output_size), int(self.output_size*1.2))
        #self.fc11 = nn.Linear(int(self.output_size*1.2), int(self.output_size*1.4))
        #self.fc12 = nn.Linear(int(self.output_size*1.4), int(self.output_size*1.6))
        #self.fc13 = nn.Linear(int(self.output_size*1.6), int(self.output_size*1.8))
        #self.fc14 = nn.Linear(int(self.output_size*1.8), int(self.output_size*2))
        #self.fc15 = nn.Linear(int(self.output_size*2), int(self.output_size*2.2))
        #self.fc16 = nn.Linear(int(self.output_size*2.2), int(self.output_size*2.4))
        #self.fc17 = nn.Linear(int(self.output_size*2.4), int(self.output_size*2.6))
        #self.fc18 = nn.Linear(int(self.output_size*2.6), int(self.output_size*2.8))
        #self.fc19 = nn.Linear(int(self.output_size*2.8), int(self.output_size*3))
        #self.fc20 = nn.Linear(int(self.output_size*3), int(self.output_size))
        
        
    def forward(self, state):
        a = func.relu(self.fc1(state))
        a = func.relu(self.fc2(a))
        a = func.relu(self.fc3(a))
        a = func.relu(self.fc4(a))
        a = func.relu(self.fc5(a))
        a = func.relu(self.fc6(a))
        a = func.relu(self.fc7(a))
        a = func.relu(self.fc8(a))
        a = func.relu(self.fc9(a))
        #a = func.relu(self.fc10(a))
        #a = func.relu(self.fc11(a))
        #a = func.relu(self.fc12(a))
        #a = func.relu(self.fc13(a))
        #a = func.relu(self.fc14(a))
        #a = func.relu(self.fc15(a))
        #a = func.relu(self.fc16(a))
        #a = func.relu(self.fc17(a))
        #a = func.relu(self.fc18(a))
        #a = func.relu(self.fc19(a))
        #a = func.relu(self.fc20(a))
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
    
    def __init__(self, gamma, input_vector, bias, current_position, loss, agent_number, brain):
        self.input_vector = []
        self.current_position = current_position
        self.input_vector = input_vector
        self.bias = bias
        self.elapsed_time = 0
        self.idle_time = 0
        self.gamma = gamma
        self.reward_window = []
        if brain == 'master':
            self.model = master_brain(len(self.input_vector), len(self.bias))
            self.loss = loss
        
        elif brain == 'selector':
            self.model = selector_brain(len(self.input_vector), agent_number)
            self.loss = loss
        
        else:
            self.model = agent_brain(len(self.input_vector), len(self.bias))
            self.loss = loss
            
        self.memory = ReplayMem(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = .001)
        self.last_state = torch.Tensor(len(self.input_vector)).unsqueeze(0)
        self.last_bias = self.bias
        self.last_reward = 0
        
    
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

    def update(self, reward, last_state):
        new_state = torch.Tensor(last_state).float().unsqueeze(0)
        checker = [1.] * len(self.last_bias)
        if self.last_bias == checker:
            self.memory.push((self.last_state, new_state, 
                              torch.Tensor(self.last_bias).unsqueeze(0), 
                              torch.Tensor([self.last_reward]).unsqueeze(0)))
        else:
            self.memory.push((self.last_state, new_state, 
                              torch.Tensor(self.last_bias), 
                              torch.Tensor([self.last_reward]).unsqueeze(0)))
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

    def save(self, type):
        torch.save({'state_dict' : self.model.state_dict(), 
                    'optimizer' : self.optimizer.state_dict()},
                   'last_' + type +'.pth')
    
    def load(self, type):
        if os.path.isfile('last_' + type +'.pth'):
            print('Loading...')
            checkpoint = torch.load('last_' + type +'.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('done')
        else:
            print('no save file found')

def check_brain(type):
    return os.path.isfile('last_' + type +'.pth')

def create_queue_ordered_unbiased(Task_dictionary, max_task_number, time_slot_duration, A = 9, B = 6.5):
    tasks = []
    scores = []
    task_times = []
    n_a = list({k: v for k, v in Task_dictionary.items() if v[-1] == 'A'}.keys())
    n_b = list({k: v for k, v in Task_dictionary.items() if v[-1] == 'B'}.keys())
    n_c = list({k: v for k, v in Task_dictionary.items() if v[-1] == 'C'}.keys())
    for i in range(max_task_number):
        aux = round(rd.random() * 10, 2)
        if aux >= A:
            rand_task = rd.choices(n_c, k=1)            
        elif A > aux >= B:
            rand_task = rd.choices(n_b, k=1)
        else:
            rand_task = rd.choices(n_a, k=1)
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

#Refazer pra lidar com a nova fila
def order_queue_biased(Task_dictionary, task_queue, bias, type):
    temp_scores = []
    task_times = []
    task_scores = [1] * len(task_queue)
    tasks = task_queue.copy()
    if type == 'agent':
        
        for i in range(len(task_queue)):
            task = tasks[i]
            task, task_type = task.split('.')
            if task == 'Dummy':
                t_time = 99
            
            else:
                task = task.lstrip('[')
                task = int(task.rstrip(']'))
                
            
                if task_type == 'L':
                    t_time = Task_dictionary[task][2] + Task_dictionary[task][3]
                else:
                    t_time = Task_dictionary[task][7] + Task_dictionary[task][6]
            
            task_times.append(t_time)
        
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
    
    else:
        
        for task in tasks:
            if isinstance(task, str):
                task, task_type = task.split('.')
                if task == 'Dummy':
                    t_time_l = 99
                    t_time_u = 99
                
                else:
                    task = task.lstrip('[')
                    task = int(task.rstrip(']'))
                    t_time_l = Task_dictionary[task][2] + Task_dictionary[task][3]
                    t_time_u = Task_dictionary[task][5] + Task_dictionary[task][4]
                    
            else:    
                task = task[0]
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

# def order_queue_biased_ga(task_queue, bias, mut_prob, n_gen, idle_time, start_pos, w, matrix, 
#                           Tasks, ela_time, trigger, pop_size, capacity, n_agents):
#     task_order = GeneticAlgorithm(pop_size).solve(mut_prob, n_gen, idle_time, task_queue, start_pos, w, matrix,
#                           Tasks, ela_time, trigger, capacity, n_agents)
#     task_queue = [x for _,x in sorted(zip(task_order, task_queue))]
#     bias = [x for _,x in sorted(zip(task_order, bias))]
#     task_scores = []
#     for i in range(len(task_queue)):
#         task_scores.append(i+1)
#     for i in range(len(task_scores)):
#         task_scores[i] = task_scores[i]/max(task_scores)
#     task_scores.sort(reverse=True)
#     for i in range(len(task_scores)):
#         task_scores[i] += bias[i]
#     task_queue = [x for _,x in sorted(zip(task_scores, task_queue), reverse= True)]
#     bias = [x for _,x in sorted(zip(task_scores, bias), reverse=True)]
#     task_scores.sort(reverse=True)
#     return task_queue, bias, task_scores

# def order_queue_biased_pure(task_queue, bias):
#     task_queue = [x for _,x in sorted(zip(bias, task_queue), reverse= True)]
#     bias = sorted(bias, reverse=True)
#     return task_queue, bias

def add_random_task(task_queue, Task_dictionary, task_scores, bias, trigger, type, A = 9, B = 6.5):
    n_a = list({k: v for k, v in Task_dictionary.items() if v[-1] == 'A'}.keys())
    n_b = list({k: v for k, v in Task_dictionary.items() if v[-1] == 'B'}.keys())
    n_c = list({k: v for k, v in Task_dictionary.items() if v[-1] == 'C'}.keys())
    aux = round(rd.random() * 10, 2)
    
    if aux >= A:
        rand_task = rd.choices(n_c, k=1)            
    elif A > aux >= B:
        rand_task = rd.choices(n_b, k=1)
    else:
        rand_task = rd.choices(n_a, k=1)
    
    trigger.append(rd.randint(1, 10))
    task_queue.append(rand_task)
    # task_queue.append(str(rand_task) + '.L')
    # task_queue.append(str(rand_task) + '.U')
    bias.append(1)
    task_queue, bias, task_scores = order_queue_biased(Task_dictionary, task_queue, bias, type)
    
    return task_queue, bias, task_scores, trigger

# def add_random_task_ga(task_queue, Task_dictionary, task_scores, bias, trigger,
#                        mut_prob, n_gen, idle_time, start_pos, w, matrix, 
#                        ela_time, pop_size, A = 9, B = 6.5):
#     n_a = list({k: v for k, v in Task_dictionary.items() if v[-1] == 'A'}.keys())
#     n_b = list({k: v for k, v in Task_dictionary.items() if v[-1] == 'B'}.keys())
#     n_c = list({k: v for k, v in Task_dictionary.items() if v[-1] == 'C'}.keys())
#     aux = round(rd.random() * 10, 2)
#     if aux >= A:
#         rand_task = rd.choices(n_c, k=1)            
#     elif A > aux >= B:
#         rand_task = rd.choices(n_b, k=1)
#     else:
#         rand_task = rd.choices(n_a, k=1)
#     trigger.append(rd.randint(1, 10))
#     task_queue.append(rand_task)
#     bias = bias.tolist()
#     bias = bias[0]
#     bias.append(1)
#     task_queue, bias, task_scores = order_queue_biased_ga(task_queue, bias, 
#                                                           mut_prob, n_gen, 
#                                                           idle_time, start_pos, 
#                                                           w, matrix, Task_dictionary,
#                                                           ela_time, trigger,
#                                                           pop_size)
#     return task_queue, bias, task_scores, trigger

# def add_random_task_pure(task_queue, Task_dictionary, bias, trigger, A = 9, B = 6.5):
#     n_a = list({k: v for k, v in Task_dictionary.items() if v[-1] == 'A'}.keys())
#     n_b = list({k: v for k, v in Task_dictionary.items() if v[-1] == 'B'}.keys())
#     n_c = list({k: v for k, v in Task_dictionary.items() if v[-1] == 'C'}.keys())
#     aux = round(rd.random() * 10, 2)
#     if aux >= A:
#         rand_task = rd.choices(n_c, k=1)            
#     elif A > aux >= B:
#         rand_task = rd.choices(n_b, k=1)
#     else:
#         rand_task = rd.choices(n_a, k=1)
#     trigger.append(rd.randint(1, 10))
#     task_queue.append(rand_task)
#     bias = bias.tolist()
#     bias = bias[0]
#     bias.append(1)
#     task_queue = [x for _,x in sorted(zip(bias, task_queue), reverse= True)]
#     bias = sorted(bias, reverse=True)
#     return task_queue, bias, trigger

def check_time(elapsed_time, max_window):
    if elapsed_time >= max_window:
        return True

def route_to_neutral(current_position, adjacency_matrix, warehouse_number):
    
    mem = 999
    route_node = None
    
    for i in range(1, warehouse_number+1):
        aux = adjacency_matrix[current_position][i]
        if aux < mem:
            mem = aux
            route_node = i
    
    return route_node
      
def create_input(total_task_queue, task_dict, bias, current_position, elapsed_time, 
                 adjacency_matrix, warehouse_number, n_agents, type):   
    input_v = []
    split_size = int(len(total_task_queue) / n_agents)
    if type == 'agent':
        for agent in range(n_agents):
            task_action_time = []
            next_task_move_time = []
            task_window_start = []
            task_window_end = []
            task_obj_pos = []
            i = 0
            split_point = split_size * (agent + 1)
            
            if agent == 0:
                task_queue = total_task_queue[0:split_point]
                last_split = split_point
            else:
                task_queue = total_task_queue[last_split:split_point]
                last_split = split_point
                
            for i in range(len(task_queue)):
                curr_task, task_type = task_queue[i].split('.')
                curr_task = curr_task.lstrip('[')
                curr_task = curr_task.rstrip(']')
                if i+1 < len(task_queue):
                    next_task, next_type = task_queue[i+1].split('.')
                    next_task = next_task.lstrip('[')
                    next_task = next_task.rstrip(']')
                else:
                    next_task = 'Dummy'
                
                if curr_task == 'Dummy':
                    task_pos = current_position
                    action_time = 0
                    window_start = 0 
                    window_end = 99
                    if next_task == 'Dummy':
                        next_pos = current_position
                    elif next_type == 'L':
                        next_pos = task_dict[int(next_task)][0]
                    else:
                        next_pos = task_dict[int(next_task)][1]
                        
                    next_move_time = adjacency_matrix[task_pos][next_pos]
                
                else:            
                    curr_task = int(curr_task)
                    if next_task != 'Dummy':
                        next_task = int(next_task)
                    
                    if task_type == 'L':
                        task_pos = task_dict[curr_task][0]
                        action_time = task_dict[curr_task][4]
                        window_start = task_dict[curr_task][2]
                        window_end = task_dict[curr_task][3] + window_start
                        task_next_pos = task_dict[curr_task][1] + warehouse_number
                        next_move_time = adjacency_matrix[task_pos][task_next_pos]
                        if next_task == 'Dummy':
                            next_pos = task_dict[curr_task][0]
                        elif next_type == 'L':
                            next_pos = task_dict[next_task][0]
                        else:
                            next_pos = task_dict[next_task][1]
                        next_move_time = adjacency_matrix[task_pos][next_pos]
                    
                    else:
                        task_pos = task_dict[curr_task][1] + warehouse_number  
                        action_time = task_dict[curr_task][7]
                        window_start = task_dict[curr_task][5]
                        window_end = task_dict[curr_task][6] + window_start
                        if i == len(task_queue) - 1:
                            next_pos = route_to_neutral(task_pos, adjacency_matrix, warehouse_number)
                            next_move_time = adjacency_matrix[task_pos][next_pos] * 2
                        else:
                            if next_task == 'Dummy':
                                next_pos = task_dict[curr_task][0]
                            elif next_type == 'L':
                                next_pos = task_dict[next_task][0]
                            else:
                                next_pos = task_dict[next_task][1]
                            next_move_time = adjacency_matrix[task_pos][next_pos]
                
                task_obj_pos.append(task_pos)
                next_task_move_time.append(next_move_time)
                task_action_time.append(action_time)
                task_window_start.append(window_start)
                task_window_end.append(window_end)
            for i in range(len(task_queue)):
                input_v.append(task_obj_pos[i])
                input_v.append(next_task_move_time[i]/(max(next_task_move_time) + .000001))
                input_v.append(task_action_time[i]/(max(task_action_time) + .000001))
                input_v.append(task_window_start[i]/(max(task_window_start) + .000001))
                input_v.append(task_window_end[i]/(max(task_window_end) + .000001))
                
                if task_type == 'L':
                    input_v.append(1)
                else:
                    input_v.append(0)
                
                if curr_task == 'Dummy':
                    input_v.append(1)
                    input_v.append(0)
                    input_v.append(0)                
                
                elif task_dict[curr_task][-1] =='A':
                    input_v.append(0)
                    input_v.append(1)
                    input_v.append(0)
                elif task_dict[curr_task][-1] =='B':
                    input_v.append(0)
                    input_v.append(0)
                    input_v.append(1)
                else:
                    input_v.append(0)
                    input_v.append(0)
                    input_v.append(0)
                
                input_v.append(bias[i])
                
        input_v.append(current_position)
        return input_v
    else:
        
        #task_load_move = []
        task_load_start = []
        task_load_end = []
        #task_unload_move = []
        task_unload_start = []
        task_unload_end = []
        input_v = []
        
        for i in range(len(total_task_queue)):
            task = total_task_queue[i][0]
            # load_pos = task_dict[task][0]
            # unload_pos = task_dict[task][1] + warehouse_number
            #task_load_move.append(adjacency_matrix[current_position][load_pos])
            task_load_start.append(task_dict[task][2] + elapsed_time)
            task_load_end.append(task_dict[task][3] + task_load_start[i])
            #task_unload_move.append(adjacency_matrix[unload_pos][load_pos])
            task_unload_start.append(task_dict[task][5] + task_load_end[i])
            task_unload_end.append(task_dict[task][6] + task_unload_start[i])
        
        for i in range(len(total_task_queue)):
            curr_task = total_task_queue[i][0]
            #input_v.append(task_queue[i])
            #input_v.append(task_load_move[i]/max(task_load_move))
            input_v.append(task_load_start[i]/max(task_load_start))
            input_v.append(task_load_end[i]/max(task_load_end))
            #input_v.append(task_unload_move[i]/max(task_unload_move))
            input_v.append(task_unload_start[i]/max(task_unload_start))
            input_v.append(task_unload_end[i]/max(task_unload_end))
        
            if curr_task == 'Dummy':
                input_v.append(1)
                input_v.append(0)
                input_v.append(0)                
            
            elif task_dict[task][-1] =='A':
                input_v.append(0)
                input_v.append(1)
                input_v.append(0)
            elif task_dict[task][-1] =='B':
                input_v.append(0)
                input_v.append(0)
                input_v.append(1)
            else:
                input_v.append(0)
                input_v.append(0)
                input_v.append(0)
            
            input_v.append(bias[i])
        #input_v.append(current_position)
        return input_v

def Split_queue(Task_queue, agent_capacity):
    aux = Task_queue.copy()
    task_queue = []
    while len(aux) < agent_capacity:
        aux.append('Dummy')
        
    for task in aux:
        task_queue.append(str(task) + '.L')
        task_queue.append(str(task) + '.U')
        
    return task_queue
    
def Test(adjacency_matrix, task_queue, tasks_dictionary, elapsed_time, idle, 
         current_position, previous_position, warehouse_number, changed, trigger,
         time_start_task, agent_capacity):
    i = 0
    curr_pos = current_position
    prev_pos = previous_position
    curr_pos = current_position
    complete = []
    failed = []
    cargo = [0] * agent_capacity
    penalty = 0
#     for agent in range(n_agents):
#         i = 0
#         split_point = split_size * (agent + 1)
        
#         if agent == 0:
#             task_queue = total_task_queue[0:split_point]
#             last_split = split_point
#         else:
#             task_queue = total_task_queue[last_split:split_point]
#             last_split = split_point
            
    agent_is_full = False

    while i < len(task_queue): #keeps the function running while going through task queue
        if changed == False:
            time_start_task = elapsed_time
        curr_task, task_type = task_queue[i].split('.')
        curr_task = curr_task.lstrip('[')
        curr_task = curr_task.rstrip(']')             

        if curr_task == 'Dummy':
            load_pos = current_position
            unload_pos = 0
            loading_time = 0
            unloading_time = 0
            load_start = 0 + time_start_task
            load_end = 99 + load_start
            unload_start = 0 + time_start_task
            unload_end = 99 + unload_start
        else:
            curr_task = int(curr_task)
            load_pos = tasks_dictionary[curr_task][0]
            unload_pos = tasks_dictionary[curr_task][1] + warehouse_number  
            loading_time = tasks_dictionary[curr_task][4]
            unloading_time = tasks_dictionary[curr_task][7]
            load_start = tasks_dictionary[curr_task][2] + time_start_task
            load_end = tasks_dictionary[curr_task][3] + load_start
            unload_start = tasks_dictionary[curr_task][5] + time_start_task
            unload_end = tasks_dictionary[curr_task][6] + unload_start

        if changed == True:
            changed = False

        if changed == False:
            if trigger[floor(i/2)] >=9:
                time_start_task = elapsed_time
                changed = True

        if task_type == 'L':
            Load = True
        else:
            Load = False
        #checking if we should load the product on the AGV

        if Load == True:

            if load_pos == prev_pos and task_queue[i]!=task_queue[i-1]:
                elapsed_time += 5
                penalty += 15
            move_time = adjacency_matrix[curr_pos][load_pos] 
            elapsed_time += move_time

            #checking if the AGV arrived before the opening of the load window
            #if it did it stays there until the opening of the window



            if elapsed_time < load_start:

                idle += load_start - elapsed_time
                elapsed_time = load_start
                elapsed_time += loading_time
                idle += loading_time
                prev_pos = curr_pos
                curr_pos = load_pos

                if not agent_is_full: #If the agent is not full Load
                    for j in  range(len(cargo)):
                        if cargo[j] == 0:
                            cargo[j] = curr_task
                            break
                    complete.append(str(curr_task) + '.L')
                    penalty -= 5

                else: #If the agent is full apply heavy penalty
                    penalty += 10
                    failed.append(str(curr_task) + '.L')

                if changed == False:
                    if trigger[floor(i/2)] >=5:
                        time_start_task = elapsed_time
                        changed = True

                if 0 not in cargo:
                    agent_is_full = True

                i += 1
            #checking if the AGV arrived during it's intended loading window
            #if it did the task proceeds as usual

            elif elapsed_time >= load_start and elapsed_time <= load_end:

                elapsed_time += loading_time
                idle += loading_time
                prev_pos = curr_pos
                curr_pos = load_pos

                if not agent_is_full: #If the agent is not full Load
                    for j in  range(len(cargo)):
                        if cargo[j] == 0:
                            cargo[j] = curr_task
                            break
                    complete.append(str(curr_task) + '.L')
                    penalty -= 5

                else: #If the agent is full apply heavy penalty
                    penalty += 10
                    failed.append(str(curr_task) + '.L')

                if changed == False:
                    if trigger[floor(i/2)] >=5:
                        time_start_task = elapsed_time
                        changed = True

                if 0 not in cargo:
                    agent_is_full = True


                i += 1

            #checking if the AGV failed to arrive during it's time window
            #if it did it returns to the neutral point (node 0) while adding a 
            #penalty to it's score

            else:
                elapsed_time += 2 * move_time + loading_time
                if changed == False:
                    if trigger[floor(i/2)] >=5:
                        time_start_task = elapsed_time
                        changed = True
                penalty += 5
                i += 1
                failed.append(str(curr_task) + '.L')


        else:

            if unload_pos == prev_pos:
                elapsed_time += 5
                penalty += 15

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
                prev_pos = curr_pos
                curr_pos = unload_pos

                if curr_task not in cargo:
                    failed.append(str(curr_task) + '.U')
                    penalty += 10
                else:
                    for j in range(len(cargo)):
                        if cargo[j] == curr_task:
                            cargo[j] = 0
                            agent_is_full = False
                            break
                    complete.append(str(curr_task) + '.U')
                    penalty -= 5

                if changed == False:
                    if trigger[floor(i/2)] >=3:
                        time_start_task = elapsed_time
                        changed = True


                i += 1

            #checking if the AGV arrived during it's intended unloading window
            #if it did the task proceeds as usual and add the time taken to 
            #complete the task and the completed tasks to their respective lists

            elif elapsed_time >= unload_start and elapsed_time <= unload_end:

                elapsed_time += unloading_time
                idle += unloading_time
                prev_pos = curr_pos
                curr_pos = unload_pos


                if curr_task not in cargo:
                    failed.append(str(curr_task) + '.U')
                    penalty += 10
                else:
                    for j in range(len(cargo)):
                        if cargo[j] == curr_task:
                            cargo[j] = 0
                            agent_is_full = False
                            break
                    complete.append(str(curr_task) + '.U')
                    penalty -= 5

                if changed == False:
                    if trigger[floor(i/2)] >=3:
                        time_start_task = elapsed_time
                        changed = True


                i += 1

            #checking if the AGV failed to arrive during it's time window
            #if it did it returns to the neutral point (node 0) while adding a 
            #penalty to it's score

            else:
                elapsed_time += 2 * move_time + loading_time +\
                    adjacency_matrix[0][load_pos] + 5

                if changed == False:
                    if trigger[floor(i/2)] >=3:
                        time_start_task = elapsed_time
                        changed = True

                if curr_task in cargo:
                    for j in range(len(cargo)):
                        if cargo[j] == curr_task:
                            cargo[j] = 0
                            agent_is_full = False
                            break

                penalty += 10
                i += 1
                curr_pos = 0
                failed.append(str(curr_task) + '.U')
                    
    return task_queue, idle, elapsed_time, curr_pos, prev_pos, complete,\
           failed, changed, trigger, time_start_task, penalty

def Train_model(task_q, task_scores, adjacency_matrix, trainning_duration, Task_dictionary, 
                warehouse_number, gamma, trigger, agent_capacity, n_agents, master_loss, agent_loss, selector = False):
    scores = []
    total_penalty = 0
    elapsed_time = 0
    split_size = int(len(task_q)/n_agents)
    idling = 0
    task_queue = task_q.copy()  
    current_position = [0] * n_agents
    previous_position = [0] * n_agents
    test_trigger = trigger.copy()
    complete, failed = [], []
    master_bias = [1] * len(task_queue)
    agent_bias = [1] * max(split_size *2, agent_capacity * 2)
    aux_bias = [0] * n_agents
    selector_queue = [[] for _ in range(n_agents)]
    master_input_vector = create_input(task_queue, Task_dictionary, master_bias, 0, 
                                elapsed_time, adjacency_matrix, warehouse_number,
                                n_agents, 'master')
    selector_split = len(master_input_vector)/len(task_queue)
    agent_task_q = Split_queue(task_queue[:split_size], agent_capacity)    
    agent_input_vector = create_input(agent_task_q, Task_dictionary, agent_bias, 0, 
                                elapsed_time, adjacency_matrix, warehouse_number, n_agents, 'agent')
    
    
    
    last_reward = 0
    master_scores = []
    agent_scores = []
    if not selector:
        master = Dqn( gamma, master_input_vector, master_bias, current_position,
                    master_loss, n_agents, 'master')
    else:
        master = Dqn( gamma, master_input_vector[0:selector_split], master_bias, current_position,
                    master_loss, n_agents, 'selector')
    agent = Dqn( gamma, agent_input_vector, agent_bias, current_position, 
                agent_loss, n_agents, 'agent')
    if check_brain('master'):
        master.load('master')
    if check_brain('agent'):
        agent.load('agent')
    t_d = trainning_duration * 3600
    t_f = 0
    t_i = timer()
    changed = False
    time_start_task = 0
    while t_f - t_i <= t_d:
        aux_queue = [[]] * n_agents
        if not selector:
            master_last_state = master_input_vector
            master_bias = master.update(last_reward, master_last_state)
            master_bias = master_bias.tolist()[0]
            master_scores.append(master.score())
            master_task_q, master_bias, task_scores = order_queue_biased(Task_dictionary, task_queue, master_bias, 'master')
        else:
            for i in range(len(task_queue)):
                master_last_state = master_input_vector[selector_split * i: selector_split * (i+1)]
                master_last_state = master_input_vector
                master_bias = master.update(last_reward, master_last_state)
                master_bias = master_bias.tolist()[0]
                chosen_agent = master_bias.index(max(master_bias))
                selector_queue[chosen_agent].append(task_queue[i])
                master_scores.append(master.score())
            master_task_q = [queue for queues in selector_queue for queue in queues]
        for i in range(n_agents):
            agent_task_q = Split_queue(master_task_q[split_size * i : split_size * (i+1)],
                                       agent_capacity)
            agent_last_state = create_input(agent_task_q, Task_dictionary, agent_bias, current_position[i], 
                                elapsed_time, adjacency_matrix, warehouse_number, n_agents, 'agent')
            agent_bias = agent.update(last_reward, agent_last_state)
            agent_bias = agent_bias.tolist()[0]
            agent_scores.append(agent.score())
            last_idle = idling
            time_start = elapsed_time
            last_len_complete = len(complete)
            last_len_failed = len(failed)
            agent_task_q, final_bias, task_scores = order_queue_biased(Task_dictionary, agent_task_q, agent_bias, 'agent')
            agent_task_q, idling, elapsed_time, current_position[i], previous_position[i], complete,\
            failed, changed, test_trigger, time_start_task, penalty =\
            Test(adjacency_matrix, agent_task_q, Task_dictionary, elapsed_time, idling, 
                current_position[i], previous_position[i], warehouse_number,
                changed, test_trigger, time_start_task, agent_capacity)
            agent_bias = agent_bias[1:]
            aux_bias[i] = agent_bias
            total_penalty += penalty
            aux_queue[i] = agent_task_q
        final_bias = [bias for sublist in aux_bias for bias in sublist]
        for i in range(n_agents):
            del master_task_q[split_size * i]
            master_task_q, final_bias, task_scores, test_trigger = add_random_task(master_task_q, Task_dictionary,
                                                        task_scores, final_bias, test_trigger, 'master')
        master_task_q, final_bias, task_scores = order_queue_biased(Task_dictionary, master_task_q, final_bias, 'master')
        master_input_vector = create_input(master_task_q, Task_dictionary, final_bias, current_position, 
                                    elapsed_time, adjacency_matrix, warehouse_number, n_agents, 'master')
        
        complete = true_complete(complete)
        
        if len(complete) > last_len_complete:
            last_reward += 10
        
        if len(failed) > last_len_failed:
            last_reward -= 10
        
        last_reward -= round(((elapsed_time - time_start)/100) + 
                             ((idling - last_idle)/ 10) + 
                             total_penalty/n_agents, 0)
        t_f = timer()
    master.save('master')
    agent.save('agent')
    scores.append(master_scores)
    scores.append(agent_scores)
    master_loss.append(master.loss)
    agent_loss.append(agent.loss)
    return scores, last_reward, master_loss, agent_loss

def true_complete(fake_complete):
    L = []
    U = []
    complete = []
    for task in fake_complete:
        task_id, task_type = task.split('.')
        if task_type == 'L':
            L.append(task_id)
        else:
            U.append(task_id)
    
    for i in range(len(L)):
        for j in range(len(U)):
            if L[i] == U[j]:
                complete.append(L[i])
                U[j] = None
                break  
    
    return complete
 
def create_queue2(task_queue, Task_dictionary, current_position, elapsed_time,
                adjacency_matrix, warehouse_number, gamma, last_reward,
                scores, agent_capacity, n_agents, master_loss, agent_loss, selector = False):
    
    split_size = int(len(task_queue)/n_agents)
    master_loss = []
    agent_loss = []
    selector_queue = [[] for _ in range(n_agents)]
    master_bias = [1] * len(task_queue)
    final_queue = []
    master_input_vector = create_input(task_queue, Task_dictionary, master_bias, 0,
                                       elapsed_time, adjacency_matrix, warehouse_number,
                                       n_agents, 'master')
    selector_split = len(master_input_vector)/len(task_queue)
    master = Dqn( gamma, master_input_vector, master_bias, 0, master_loss, agent_loss, 'master')
    if not selector:
            master_last_state = master_input_vector
            master_bias = master.update(last_reward, master_last_state)
            master_bias = master_bias.tolist()[0]
            master_task_q, master_bias, _ = order_queue_biased(Task_dictionary, task_queue, master_bias, 'master')
    else:
        for i in range(len(task_queue)):
            master_last_state = master_input_vector[selector_split * i: selector_split * (i+1)]
            master_last_state = master_input_vector
            master_bias = master.update(last_reward, master_last_state)
            master_bias = master_bias.tolist()[0]
            chosen_agent = master_bias.index(max(master_bias))
            selector_queue[chosen_agent].append(task_queue[i])
        master_task_q = [queue for queues in selector_queue for queue in queues]    
    
    agent_task_q = Split_queue(master_task_q[0 : split_size], agent_capacity)
    agent_bias = [1] * max(split_size *2, agent_capacity * 2)
    agent_input_vector = create_input(agent_task_q, Task_dictionary, agent_bias, 0, 
                                elapsed_time, adjacency_matrix, warehouse_number, n_agents, 'agent')
    agent = Dqn( gamma, agent_input_vector, agent_bias, current_position, master_loss, agent_loss, 'agent')
    agent.load('agent')
    
    for i in range(n_agents):            
            agent_task_q = Split_queue(master_task_q[split_size * i : split_size * (i+1)],
                                       agent_capacity)
            agent_input = create_input(agent_task_q, Task_dictionary, agent_bias, current_position[i], 
                                elapsed_time, adjacency_matrix, warehouse_number, n_agents, 'agent')
            agent_bias = agent.update(last_reward, agent_input)
            agent_bias = agent_bias.tolist()[0]
            agent_task_q, agent_bias, _ = order_queue_biased(Task_dictionary, agent_task_q, agent_bias, 'agent')
            final_queue.extend(agent_task_q)
    scores.append(master.score()) 
    scores.append(agent.score())
    return final_queue, scores

# def Train_model_ga(adjacency_matrix, trainning_duration, Task_dictionary, 
#                    warehouse_number, gamma, n_gen, pop_size, mut_prob):
#     elapsed_time = 0
#     idling = 0
#     current_position = 0
#     test_trigger = []
#     complete, failed = [], []
#     task_q, task_scores = create_queue_ordered_unbiased(Task_dictionary, 20, 240)
#     bias = [1] * len(task_q)
#     for i in range(len(task_q)):
#         test_trigger.append(rd.randint(1, 10))
#     input_vector = create_input(task_q, Task_dictionary, bias, current_position, 
#                                 elapsed_time, adjacency_matrix, warehouse_number)
#     last_reward = 0
#     scores = []
#     brain = Dqn(task_q, Task_dictionary, adjacency_matrix, warehouse_number, 
#         gamma, input_vector, bias, current_position)
#     t_d = trainning_duration * 3600
#     t_f = 0
#     t_i = timer()
#     changed = False
#     time_start_task = 0
#     while t_f - t_i <= t_d:
#         last_state = input_vector
#         bias = brain.update(last_reward, last_state)
#         scores.append(brain.score())
#         last_idle = idling
#         time_start = elapsed_time
#         last_len_complete = len(complete)
#         last_len_failed = len(failed)
#         task_q, idling, elapsed_time, current_position, complete, failed, changed, test_trigger, time_start_task =\
#         Test(adjacency_matrix, task_q, Task_dictionary, elapsed_time, idling, 
#              current_position, warehouse_number, complete, failed, changed, test_trigger, time_start_task)
#         bias = bias[0][0:-1]
#         bias = bias.unsqueeze(0)
        
        
#         task_q, bias, task_scores, test_trigger = add_random_task_ga(task_q, Task_dictionary,
#                                                                      task_scores, bias, test_trigger,
#                                                                      mut_prob, n_gen, idling, 
#                                                                      current_position, warehouse_number, 
#                                                                      adjacency_matrix, elapsed_time, pop_size)
#         task_q, bias, task_scores = order_queue_biased_ga(task_q, bias, mut_prob,
#                                                           n_gen, idling, current_position,
#                                                           warehouse_number, adjacency_matrix,
#                                                           Task_dictionary, elapsed_time, test_trigger, pop_size)
                
#         input_vector = create_input(task_q, Task_dictionary, bias, current_position, 
#                                     elapsed_time, adjacency_matrix, warehouse_number)
#         if len(complete) > last_len_complete:
#             last_reward += 10
#         if len(failed) > last_len_failed:
#             last_reward -= 10
#         last_reward -= ((elapsed_time - time_start)/100) - ((idling - last_idle)/ 10)
#         t_f = timer()
#     #brain.save()
#     return scores, last_reward
    
# def Train_model_pure(adjacency_matrix, trainning_duration, Task_dictionary, warehouse_number, gamma, trigger):
#     elapsed_time = 0
#     idling = 0
#     current_position = 0
#     test_trigger = trigger
#     complete, failed = [], []
#     task_q, task_scores = create_queue_ordered_unbiased(Task_dictionary, 20, 240)
#     bias = [1] * len(task_q)
#     input_vector = create_input(task_q, Task_dictionary, bias, current_position, 
#                                 elapsed_time, adjacency_matrix, warehouse_number)
#     last_reward = 0
#     scores = []
#     brain = Dqn(task_q, Task_dictionary, adjacency_matrix, warehouse_number, 
#         gamma, input_vector, bias, current_position)
#     t_d = trainning_duration * 3600
#     t_f = 0
#     t_i = timer()
#     changed = False
#     time_start_task = 0
#     while t_f - t_i <= t_d:
#         last_state = input_vector
#         bias = brain.update(last_reward, last_state)
#         scores.append(brain.score())
#         last_idle = idling
#         time_start = elapsed_time
#         last_len_complete = len(complete)
#         last_len_failed = len(failed)
#         task_q, idling, elapsed_time, current_position, complete, failed, changed, test_trigger, time_start_task =\
#         Test(adjacency_matrix, task_q, Task_dictionary, elapsed_time, idling, 
#              current_position, warehouse_number, complete, failed, changed, test_trigger, time_start_task)
#         bias = bias[0][0:-1]
#         bias = bias.unsqueeze(0)

#         task_q, bias, test_trigger = add_random_task_pure(task_q, Task_dictionary,
#                                                           bias, test_trigger)
#         task_q, bias = order_queue_biased_pure(task_q, bias)
#         input_vector = create_input(task_q, Task_dictionary, bias, current_position, 
#                                     elapsed_time, adjacency_matrix, warehouse_number)
#         if len(complete) > last_len_complete:
#             last_reward += 10
#         if len(failed) > last_len_failed:
#             last_reward -= 10
#         last_reward -= ((elapsed_time - time_start)/100) - ((idling - last_idle)/ 10)
#         t_f = timer()
#     #brain.save()
#     return scores, last_reward

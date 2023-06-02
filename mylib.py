# -*- coding: utf-8 -*-
"""
Created on Fri May 26 09:57:49 2023

@author: cbsj0
"""
import random as rd
import numpy as np
import networkx as nx
import matplotlib as plt


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
    for i in range((warehouse_number*delivery_number)\
                   +abs(rd.randint(0, warehouse_number*delivery_number))):        
        '''filling the lists and the dictionary'''
        Load.append(Warehouses[rd.randint(0, len(Warehouses)-1)])
        Unload.append(Delivery_points[rd.randint(0, len(Delivery_points)-1)])
        Load_t0.append(abs(rd.randint(5, 30)))
        Load_tf.append(abs(rd.randint(5, 10)))
        Loading_t.append(abs(rd.randint(1, 5)))
        Unload_t0.append(abs(rd.randint(5, 10)))
        Unload_tf.append(abs(rd.randint(5, 10)))
        Unloading_t.append(abs(rd.randint(1, 5)))
        Task_dictionary[i+1] = [Load[i], Unload[i], Load_t0[i], Load_tf[i],
                                Loading_t[i], Unload_t0[i], Unload_tf[i], 
                                Unloading_t[i]]
    return Task_dictionary

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

def order_queue(task_q, elapsed_time, Task_dictionary):
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


def do_tasks(curr_pos, warehouse_number, task_queue, adjacency_matrix, tasks_dictionary,
             elapsed_time, idle_time):
    Load = True
    complete_tasks = []
    failed_tasks = []
    time_taken_per_task = []
    time_start = elapsed_time
    '''
    doing the tasks and verifying if they arrived during the intended time windows
    '''
    while task_queue != []: #keeps the function running while there are tasks to complete
        curr_task = task_queue[0]
        load_pos = tasks_dictionary[curr_task][0]
        unload_pos = tasks_dictionary[curr_task][1] + warehouse_number  
        loading_time = tasks_dictionary[curr_task][4]
        unloading_time = tasks_dictionary[curr_task][7]
        
        #checking if we should load the product on the AGV
        
        if Load == True:
            load_start = tasks_dictionary[curr_task][2] + elapsed_time
            load_end = tasks_dictionary[curr_task][3] + elapsed_time
            move_time = adjacency_matrix[curr_pos][load_pos]
            elapsed_time = time_start + move_time
        
            #checking if the AGV arrived before the opening of the load window
            #if it did it stays there until the opening of the window
            
            if elapsed_time < load_start:
                idle_time += load_start - elapsed_time
                elapsed_time = load_start
                elapsed_time += loading_time
                idle_time += loading_time
                curr_pos = load_pos
                Load = False
            
            #checking if the AGV arrived during it's intended loading window
            #if it did the task proceeds as usual
            
            elif elapsed_time >= load_start and elapsed_time <= load_end:
                elapsed_time += loading_time
                idle_time += loading_time
                curr_pos = load_pos
                Load = False
            
            #checking if the AGV failed to arrive during it's time window
            #if it did it returns to the neutral point (node 0) while adding a 
            #penalty to it's idling time
            
            else:
                idle_time += 99
                elapsed_time += 2 * move_time
                failed_tasks.append(task_queue[0])
                task_queue.pop(0)

        else:
            unload_start = tasks_dictionary[curr_task][5] + elapsed_time
            unload_end = tasks_dictionary[curr_task][6] + elapsed_time
            move_time = adjacency_matrix[unload_pos][curr_pos]
            elapsed_time = time_start + move_time
            
            #checking if the AGV arrived before the opening of the unloading window
            #if it did it stays there until the opening of the window and add
            #the time taken to complete the task and the completed tasks to their
            #respective lists
            
            if elapsed_time < unload_start:
                idle_time += unload_start - elapsed_time
                elapsed_time = unload_start
                elapsed_time += unloading_time
                idle_time += unloading_time
                curr_pos = unload_pos
                complete_tasks.append(task_queue[0])
                task_queue.pop(0)
                time_taken_per_task.append(elapsed_time-time_start)
                time_start += elapsed_time
                Load = True
                
                
            #checking if the AGV arrived during it's intended unloading window
            #if it did the task proceeds as usual and add the time taken to 
            #complete the task and the completed tasks to their respective lists
                
            elif elapsed_time >= unload_start and elapsed_time <= unload_end:
                elapsed_time += unloading_time
                idle_time += unloading_time
                curr_pos = unload_pos
                complete_tasks.append(task_queue[0])
                task_queue.pop(0)
                time_taken_per_task.append(elapsed_time-time_start)
                time_start += elapsed_time
                Load = True
                
            #checking if the AGV failed to arrive during it's time window
            #if it did it returns to the neutral point (node 0) while adding a 
            #penalty to it's idling time and adding the current taskt to the 
            #list of failed tasks
            
            else:
                idle_time += 99
                elapsed_time += 2 * move_time + loading_time +\
                    adjacency_matrix[0][load_pos]
                failed_tasks.append(task_queue[0])
                task_queue.pop(0)
                time_start += elapsed_time
                curr_pos = 0
                Load = True
                
    return elapsed_time, idle_time, complete_tasks, time_taken_per_task, task_queue, failed_tasks
        
def pareto(complete_tasks, time_taken_per_task):
   
    '''paretizing the data for ease of reading'''
    
    #sorting the tasks indexes and times, based on the times in decreasing order
    
    pareto_tasks, pareto_times = [x for _,x in sorted(zip(time_taken_per_task, complete_tasks),
                                                      reverse=True)],\
        [x for x,_ in sorted(zip(time_taken_per_task, complete_tasks), reverse=True)]
    time_sum = []
    i=0
    
    #changing the tasks indexes into strings for plotting
    
    for i in range(len(pareto_tasks)):
        pareto_tasks[i] = str(pareto_tasks[i])
    
    #making the sum of the times for plotting
    
    while i < len(pareto_times):
        j=0
        sumt = 0
        while j < len(pareto_times):            
            sumt += pareto_times[j]
            j+=1
            time_sum.append(sumt)
        i+=1
    
    #plotting
    
    fig, ax = plt.pyplot.subplots()
    ax.bar(range(len(pareto_tasks)), pareto_times, color="C0")
    ax.set_xticks(range(len(pareto_tasks)), labels = pareto_tasks)
    ax2 = ax.twinx()
    ax2.plot(range(len(pareto_tasks)), time_sum, color="C1", marker="D", ms=7)
    ax2.set_xticks(range(len(pareto_tasks)), labels = pareto_tasks)
    ax.tick_params(axis="y", colors="C0")
    ax2.tick_params(axis="y", colors="C1")
    plt.pyplot.show()
        
    



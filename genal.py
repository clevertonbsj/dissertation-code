# -*- coding: utf-8 -*-

import networkx as nx
import mylib as ml

w = int(input('Quantos armazéns gostaria? '))
d = int(input('Quantos pontos de entrega gostaria? '))
#n = int(input('Quantos AMRs gostaria? '))
Warehouses = []
Del_points = []
idle_time = 0
ela_time = 0
start_pos = 0

G, matrix = ml.create_graph(w, d)
'''plotting the graph'''
nx.draw_networkx(G, with_labels=True)


'''Filling the Warehouses and Delivery points lists as per the user input '''
for i in range(w):
    Warehouses.append(i+1)

for i in range(d):
    Del_points.append(i+1)

Tasks = ml.create_tasks(w, d, Warehouses, Del_points)

#task assignement
task_q = ml.create_queue(50, 240, Tasks)

'''

here will go the code for assigning tasks to multiple agents

'''

#task management
task_q = ml.order_queue(task_q, ela_time, Tasks)

#time management

ela_time, idle_time, complete_tasks, t_task, task_q = ml.do_tasks(start_pos, w,
                                                                  task_q, matrix, Tasks, ela_time, idle_time)

ml.pareto(complete_tasks, t_task)

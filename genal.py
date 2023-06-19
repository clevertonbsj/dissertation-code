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
task_q2 = task_q
task_q2 = ml.order_queue(task_q2, range(len(task_q)))
print(task_q2)
ela_time2, idle_time2, complete_tasks2, t_task2, task_q2, failed2 = \
    ml.do_tasks(start_pos, w, task_q2, matrix, Tasks, ela_time, idle_time)
if failed2 != []:
    if len(failed2) == 1:
        print('The task', failed2[0], 'has failed.')
    else:
        print('A total of', len(failed2), 'tasks failed, they are: ', failed2)

for i in failed2:
    idle_time2 -= 99
'''

here will go the code for assigning tasks to multiple agents

'''
#task management
pop_size, n_gen, mut_prob = 10, 25, 0.01
ga = ml.GeneticAlgorithm(pop_size)
task_order = ga.solve(mut_prob, n_gen, idle_time, task_q, start_pos, w, matrix,
                      Tasks, ela_time)
task_q = ml.order_queue(task_q, task_order)
print(task_q)
ela_time, idle_time, complete_tasks, t_task, task_q, failed = \
    ml.do_tasks(start_pos, w, task_q, matrix, Tasks, ela_time, idle_time)

for i in failed:
    idle_time -= 99

#ml.pareto(complete_tasks, t_task)
if failed != []:
    if len(failed) == 1:
        print('The task', failed[0], 'has failed.')
    else:
        print('A total of', len(failed), 'tasks failed, they are: ', failed)
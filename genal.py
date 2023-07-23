# -*- coding: utf-8 -*-

import networkx as nx
import mylib as ml
import random as rd

#w = int(input('Quantos armazéns gostaria? '))
#d = int(input('Quantos pontos de entrega gostaria? '))



#train = str(input('Gostaria de realizar o treinamento da rede neural? y/n '))
for z in range(2):
    if z == 0:
        w = 3
        d = 4
    else:
        w = 4
        d = 3

    train = 'y'
    if train == 'y' or train == 'Y':
        train = True
    else:
        train = False
        
    #n = int(input('Quantos AMRs gostaria? '))
    Warehouses = []
    Del_points = []
    idle_time = 0
    ela_time = 0
    start_pos = 0
    trigger = []
    G, matrix = ml.create_graph(w, d)
    '''plotting the graph'''
    nx.draw_networkx(G, with_labels=True)
    
    
    '''Filling the Warehouses and Delivery points lists as per the user input '''
    for i in range(w):
        Warehouses.append(i+1)
    
    for i in range(d):
        Del_points.append(i+1)
    
    Tasks = ml.create_tasks(w, d, Warehouses, Del_points)
    #ml.create_excel(Tasks, "file1.xlsx", "sheet1")
    #task assignement
    task_q = ml.create_queue(50, 240, Tasks)
    for i in range(len(task_q)):
        trigger.append(rd.randint(1, 10))    
    task_q2 = task_q
    task_q2 = ml.order_queue(task_q2, range(len(task_q)))
    print(task_q2)
    task_q2, idle_time2, ela_time2, curr_pos2, complete_tasks2, failed2, t_task2, start_t2 = \
        ml.Do(matrix, task_q, Tasks, ela_time, start_pos, w, trigger)
    if failed2 != []:
        if len(failed2) == 1:
            print('The task', failed2[0], 'has failed.')
        else:
            print('A total of', len(failed2), 'tasks failed, they are: ', failed2)
    
    
    
    ml.plot(complete_tasks2, t_task2, idle_time2, 'Earliest Due Date')
    
    '''
    
    here will go the code for assigning tasks to multiple agents
    
    '''
    #task management
    pop_size, n_gen, mut_prob = 10, 25, 0.01
    ga = ml.GeneticAlgorithm(pop_size)
    task_order = ga.solve(mut_prob, n_gen, idle_time, task_q, start_pos, w, matrix,
                          Tasks, ela_time, trigger)
    task_q = ml.order_queue(task_q, task_order)
    print(task_q)
    task_q, idle_time, ela_time, curr_pos, complete_tasks, failed, t_task, start_t = \
        ml.Do(matrix, task_q, Tasks, ela_time, start_pos, w, trigger)
    
    
    ml.plot(complete_tasks, t_task, idle_time, 'Algoritmo Genético')
    if failed != []:
        if len(failed) == 1:
            print('The task', failed[0], 'has failed.')
        else:
            print('A total of', len(failed), 'tasks failed, they are: ', failed)
    
    task_q3 = task_q
    bias = [1] * len(task_q)
    task_q3, bias, task_scores = ml.order_queue_biased(Tasks, task_q3, bias)
   
    while True:
        if train == True:
            #duration = float(input('Por quantas horas gostaria de treinar a rede neural? '))
            durations = [.0167, .1503, .334, .5, 3]
            for i in range(len(durations)):
                duration = durations[i]
                scores, reward = ml.Train_model(matrix, duration, Tasks, w, .9, trigger)
                bias, scores = ml.create_bias(task_q3, Tasks, start_pos, ela_time, matrix, w,
                                              task_scores, .9, reward, scores)
                bias = bias.squeeze(0).tolist()
                task_q3, bias, task_scores = ml.order_queue_biased(Tasks, task_q3, bias)
                task_q3, idle_time3, ela_time3, curr_pos3, complete_tasks3, failed3, t_task3, start_t3 = \
                    ml.Do(matrix, task_q3, Tasks, ela_time, start_pos, w, trigger)
                print(task_q3, duration)
                if failed3 != []:
                    if len(failed3) == 1:
                        print('The task', failed3[0], 'has failed.')
                    else:
                        print('A total of', len(failed3), 'tasks failed, they are: ', failed3)
    
                
    
                ml.plot(complete_tasks3, t_task3, idle_time3, 'RNA treinamento = ' + str(3600*durations[i]) + '(s)' + ' = ' + str(durations[i]) + '(h)')
        break

        #ml.save_reward_and_scores(scores, reward)
#        break
#        train = False
#        print('done trainning')

#    else:
        #scores, reward = ml.load_reward_and_scores('last scores and reward.txt')
#        bias, scores = ml.create_bias(task_q3, Tasks, start_pos, ela_time, matrix, w,
#                                      task_scores, .9, reward, scores)

#    task_q3, bias, task_scores = ml.order_queue_biased(Tasks, task_q3, bias)
#    break

#task_q3, idle_time3, ela_time3, curr_pos3, complete_tasks3, failed3, t_task3 = \
#    ml.Do(matrix, task_q3, Tasks, ela_time, idle_time, start_pos, w)
#print(task_q3)
#if failed3 != []:
#    if len(failed3) == 1:
#        print('The task', failed3[0], 'has failed.')
#    else:
#        print('A total of', len(failed3), 'tasks failed, they are: ', failed3)

#for i in failed3:
#    idle_time3 -= 99

#ml.pareto(complete_tasks3, t_task3)

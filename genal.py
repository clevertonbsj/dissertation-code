# -*- coding: utf-8 -*-

import mylib as ml
import random as rd

#w = int(input('Quantos armazéns gostaria? '))
#d = int(input('Quantos pontos de entrega gostaria? '))
mean_tt_edd = []
mean_it_edd = []
percent_complete_edd = []
mean_tt_ga = []
mean_it_ga = []
percent_complete_ga = []
mean_tt_rna_1min = []
mean_it_rna_1min = []
percent_complete_rna_1min = []
mean_tt_rna_10min = []
mean_it_rna_10min = []
percent_complete_rna_10min = []
mean_tt_rna_30min = []
mean_it_rna_30min = []
percent_complete_rna_30min = []
mean_tt_rna_1h = []
mean_it_rna_1h = []
percent_complete_rna_1h = []
mean_tt_rna_2h = []
mean_it_rna_2h = []
percent_complete_rna_2h = []

#train = str(input('Gostaria de realizar o treinamento da rede neural? y/n '))

for z in range(1):

    w = 3
    d = 4

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
    tasks_indexes = ['Load', 'Unload', 'Load_t0', 'Load_tf', 'Loading_t',
                     'Unload_t0', 'Unload_tf', 'Unloading_t']
    matrix_indexes = list(range(w+d+1))
    ml.create_excel(Tasks, 'tables/try' + str(z+1) +'.xlsx', 'dicionário', tasks_indexes)
    ml.create_excel(matrix, 'tables/matrix' + str(z+1) +'.xlsx', 'adjacencia', matrix_indexes)
    task_q2 = task_q
    task_q2 = ml.order_queue(task_q2, range(len(task_q)))
    
    task_q2, idle_time2, ela_time2, curr_pos2, complete_tasks2, failed2, t_task2, start_t2 = \
        ml.Do(matrix, task_q, Tasks, ela_time, start_pos, w, trigger)
    
    
    
    complete_percent_edd = round(len(complete_tasks2) / len(task_q)+.00000001, 2)
    t_task_edd_mean = round(sum(t_task2)/len(t_task2)+.00000001, 2)
    idle_edd_mean = round(sum(idle_time2)/len(idle_time2)+.00000001, 2)
    if z == 0 :
        ml.plot(complete_tasks2, t_task2, idle_time2, 'Earliest Due Date')
        print(task_q2)
        if failed2 != []:
            if len(failed2) == 1:
                print('The task', failed2[0], 'has failed.')
            else:
                print('A total of', len(failed2), 'tasks failed, they are: ', failed2)
    
    '''
    
    here will go the code for assigning tasks to multiple agents
    
    '''
    #task management
    task_q1 = task_q
    pop_size, n_gen, mut_prob = 10, 200, 0.01
    ga = ml.GeneticAlgorithm(pop_size)
    task_order = ga.solve(mut_prob, n_gen, idle_time, task_q1, start_pos, w, matrix,
                          Tasks, ela_time, trigger)
    task_q1 = ml.order_queue(task_q1, task_order)
    
    task_q1, idle_time, ela_time, curr_pos, complete_tasks, failed, t_task, start_t = \
        ml.Do(matrix, task_q1, Tasks, ela_time, start_pos, w, trigger)
    
    complete_percent_ga = round(len(complete_tasks) / len(task_q), 2)
    t_task_ga_mean = round(sum(t_task)/len(t_task), 2)
    idle_ga_mean = round(sum(idle_time)/len(idle_time), 2)
    if z == 0 :
        print(task_q)
        ml.plot(complete_tasks, t_task, idle_time, 'Algoritmo Genético')
        if failed != []:
            if len(failed) == 1:
                print('The task', failed[0], 'has failed.')
            else:
                print('A total of', len(failed), 'tasks failed, they are: ', failed)
    
    
    task_q3 = task_q1
    bias = [1] * len(task_q3)
    #task_q3, bias, task_scores = ml.order_queue_biased_ga(task_q3, bias, 
    #                                                      mut_prob, n_gen, 
    #                                                      idle_time, start_pos,
    #                                                      w, matrix, Tasks, ela_time, 
    #                                                      trigger, pop_size)
    
    while True:
        if train == True:
            #duration = float(input('Por quantas horas gostaria de treinar a rede neural? '))
            durations = [.00167]#.0167, .1503, .334, .5, 1]
            for i in range(len(durations)):
                duration = durations[i]
                scores, reward = ml.Train_model_pure(matrix, duration, Tasks, w, .9, trigger)
                bias, scores = ml.create_bias(task_q3, Tasks, start_pos, ela_time, matrix, w,
                                              .9, reward, scores)
                bias = bias.squeeze(0).tolist()
                task_q3, bias = ml.order_queue_biased_pure(task_q3, bias)
                task_q3, idle_time3, ela_time3, curr_pos3, complete_tasks3, failed3, t_task3, start_t3 = \
                    ml.Do(matrix, task_q3, Tasks, ela_time, start_pos, w, trigger)

                
                
                if z == 0 :
                    print(task_q3)
                    ml.plot(complete_tasks3, t_task3, idle_time3, 'RNA treinamento = '
                            + str(round(3600*sum(durations[0:i+1]), 2)) + '(s)' + ' = ' 
                            + str(round(sum(durations[0:i+1]), 2)) + '(h)')
                    if failed3 != []:
                        if len(failed3) == 1:
                            print('The task', failed3[0], 'has failed.')
                        else:
                            print('A total of', len(failed3), 'tasks failed, they are: ', failed3)
        
                complete_percent_rna = round(len(complete_tasks3) / len(task_q), 2)
                t_task_rna_mean = round(sum(t_task3)/len(t_task3), 2)
                idle_rna_mean = round(sum(idle_time3)/len(idle_time3), 2)
                if i == 0:
                    mean_tt_rna_1min.append(t_task_rna_mean)
                    mean_it_rna_1min.append(idle_rna_mean)
                    percent_complete_rna_1min.append(complete_percent_rna)
                if i == 1:
                    mean_tt_rna_10min.append(t_task_rna_mean)
                    mean_it_rna_10min.append(idle_rna_mean)
                    percent_complete_rna_10min.append(complete_percent_rna)
                if i == 2:
                    mean_tt_rna_30min.append(t_task_rna_mean)
                    mean_it_rna_30min.append(idle_rna_mean)
                    percent_complete_rna_30min.append(complete_percent_rna)
                if i == 3:
                    mean_tt_rna_1h.append(t_task_rna_mean)
                    mean_it_rna_1h.append(idle_rna_mean)
                    percent_complete_rna_1h.append(complete_percent_rna)
                if i == 4:
                    mean_tt_rna_2h.append(t_task_rna_mean)
                    mean_it_rna_2h.append(idle_rna_mean)
                    percent_complete_rna_2h.append(complete_percent_rna)
        
        break
    mean_tt_edd.append(t_task_edd_mean)
    mean_it_edd.append(idle_edd_mean)
    percent_complete_edd.append(complete_percent_edd)
    mean_tt_ga.append(t_task_ga_mean)
    mean_it_ga.append(idle_ga_mean)
    percent_complete_ga.append(complete_percent_ga)
    

Dicio = [mean_tt_edd, mean_tt_ga, mean_tt_rna_1min,  mean_tt_rna_10min, mean_tt_rna_30min,
         mean_tt_rna_1h, mean_tt_rna_2h, mean_it_edd, mean_it_ga, mean_it_rna_1min,
         mean_it_rna_10min, mean_it_rna_30min, mean_it_rna_1h, mean_it_rna_2h,
         percent_complete_edd, percent_complete_ga, percent_complete_rna_1min,
         percent_complete_rna_10min, percent_complete_rna_30min, percent_complete_rna_1h,
         percent_complete_rna_2h]
Dicio_indexes = ['Earliest Due Date', 'Algorítmo Genético', 'RNA treino de 1 min', 'RNA treino de 10 min',
                 'RNA treino de 30 min', 'RNA treino de 1 h', 'RNA treino de 2 h (TT)', 'Earliest Due Date',
                  'Algorítmo Genético', 'RNA treino de 1 min', 'RNA treino de 10 min', 'RNA treino de 30 min',
                  'RNA treino de 1 h', 'RNA treino de 2 h (IT)', 'RNA treino de 1 min', 'RNA treino de 10 min',
                  'RNA treino de 30 min', 'RNA treino de 1 h', 'RNA treino de 2 h (%)']
ml.create_excel(Dicio, 'tables/valores graficos.xlsx', 'vals', Dicio_indexes)

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

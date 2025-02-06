# -*- coding: utf-8 -*-

import mylib as ml
import random as rd
#import myORTools as mo
import os
import time

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

for z in range(100):

    w = 8
    d = 10
    cap = 5
    n_a = 2
    train = 'n'
    if train == 'y' or train == 'Y':
        train = True
    else:
        train = False
        
    #n = int(input('Quantos AMRs gostaria? '))
    Warehouses = []
    Del_points = []
    idle_time = 0
    ela_time = 0
    start_pos = [0] * n_a
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
    task_q_not_split = ml.create_queue(50, 240*n_a, Tasks)
    
    task_q = ml.Split_queue(task_q_not_split, n_a)
    for i in range(len(task_q)):
        trigger.append(rd.randint(1, 10))
    tasks_indexes = ['Load', 'Unload', 'Load_t0', 'Load_tf', 'Loading_t',
                     'Unload_t0', 'Unload_tf', 'Unloading_t', 'Type']
    matrix_indexes = list(range(w+d+1))
    ml.create_excel(Tasks, 'tables/Dicionário do treino' + str(z+1) +'.xlsx', 'dicionário', tasks_indexes)
    ml.create_excel(matrix, 'tables/matriz adjacência' + str(z+1) +'.xlsx', 'adjacencia', matrix_indexes)
    task_q2 = task_q_not_split.copy()
    task_q2_not_split = ml.order_queue_edd(task_q2, ela_time, Tasks, cap, n_a)
    task_q2 = ml.Split_queue(task_q2_not_split, n_a)
    task_q2, idle_time2, ela_time2, curr_pos2, complete_tasks2, failed2, t_task2, start_t2 = \
        ml.Do(matrix, task_q2, Tasks, ela_time, start_pos, w, trigger, n_a, cap)
    
    
    
    complete_percent_edd = round(len(complete_tasks2) / ((len(task_q)+.00000001)), 2)
    t_task_edd_mean = round(sum(t_task2)/len(t_task2)+.00000001, 2)
    idle_edd_mean = round(sum(idle_time2)/len(idle_time2)+.00000001, 2)
    if z == 2 :
        # ml.plot(complete_tasks2, t_task2, idle_time2, 'Earliest Due Date')
        print(task_q2)
        if failed2 != []:
            if len(failed2) == 1:
                print('The task', failed2[0], 'has failed.')
            else:
                print('A total of', len(failed2), 'tasks failed, they are: ', failed2)
    

     #task management
    task_q1 = task_q.copy()
    pop_size, n_gen, mut_prob = 500, 150, 0.01
    time1 = time.time()
    print('Starting GA')
    ga = ml.GeneticAlgorithm(pop_size)
    task_order = ga.solve(mut_prob, n_gen, idle_time, task_q1, 0, w, matrix,
                           Tasks, ela_time, trigger, cap, n_a)
    print('GA done', round(time.time() - time1), 's')
    task_q1 = ml.order_queue(task_q1, task_order)
    task_q1, idle_time, ela_time, curr_pos, complete_tasks, failed, t_task, start_t = \
         ml.Do(matrix, task_q1, Tasks, ela_time, [0,0], w, trigger, n_a, cap)
    
    
    complete_percent_ga = round(len(complete_tasks)/(len(task_q1)), 2)
    t_task_ga_mean = round(sum(t_task)/len(t_task), 2)
    idle_ga_mean = round(sum(idle_time)/len(idle_time), 2)
    if z == 2 :
        print(task_q)
        # ml.plot(complete_tasks, t_task, idle_time, 'Algoritmo Genético')
        if failed != []:
            if len(failed) == 1:
                print('The task', failed[0], 'has failed.')
            else:
                print('A total of', len(failed), 'tasks failed, they are: ', failed)
    
    task_q3 = task_q2_not_split.copy()
    bias = [1] * len(task_q3)
    task_q3, bias, task_scores = ml.order_queue_biased(Tasks, task_q3, bias, 'master')
    while True:
        if train == True:
            #duration = float(input('Por quantas horas gostaria de treinar a rede neural? '))
            durations = [.0167, .1503, .334, .5, 1]
            master_loss = []
            agent_loss = []
            prev_dur = 0
            for i in range(len(durations)):
                if i == 0:
                    if ml.check_brain('agent'):
                        os.remove('last_agent.pth')
                    if ml.check_brain('master'):
                        os.remove('last_master.pth')
                
                duration = durations[i]
                #scores, reward, master_loss, agent_loss = ml.Train_model(task_q3, task_scores, matrix, duration,
                #                                        Tasks, w, .9, trigger, cap, n_a, master_loss, agent_loss, prev_dur, z)
                scores = []
                reward = 0
                for _ in range(100):
                    new_queue = ml.create_queue(50, 240*n_a, Tasks)
                    aux = new_queue.copy()
                    new_queuef = ml.order_queue_edd(aux, ela_time, Tasks, cap, n_a)
                    while len(new_queuef) > 8:
                        del new_queuef[-1]
                    while len(new_queuef) < 8:
                        new_queuef, bias, scores, trigger = ml.add_random_task(new_queue, Tasks, scores, bias, trigger, 'master')
                    task_q3f, scores = ml.create_queue2(new_queuef, Tasks, start_pos, ela_time, matrix, w,
                                                    .9, reward, scores, cap, n_a, master_loss, agent_loss, duration * 3600, z)
                    
                    task_q3f, idle_time3, ela_time3, curr_pos3, complete_tasks3, failed3, t_task3, start_t3 = \
                        ml.Do(matrix, task_q3f, Tasks, ela_time, start_pos, w, trigger, n_a, cap)
                        
                    if len(complete_tasks3) == 0:
                        complete_percent_rna = 0
                    else:
                        complete_percent_rna = round(len(complete_tasks3)/len(task_q3f), 2)
                    if len(t_task3) == 0:
                        t_task_rna_mean = 0
                    else:
                        t_task_rna_mean = round(sum(t_task3)/len(t_task3), 2)
                    if len(idle_time3) == 0:
                        idle_rna_mean = 999999
                    else:
                        idle_rna_mean = round(sum(idle_time3)/len(idle_time3), 2)
                    if i == 0:
                        mean_tt_rna_1min.append(t_task_rna_mean)
                        mean_it_rna_1min.append(idle_rna_mean)
                        percent_complete_rna_1min.append(complete_percent_rna)
                        # ml.plot_losses(losses, 'loss treino 1 min')
                        
                    if i == 1:
                        mean_tt_rna_10min.append(t_task_rna_mean)
                        mean_it_rna_10min.append(idle_rna_mean)
                        percent_complete_rna_10min.append(complete_percent_rna)
                        # ml.plot_losses(losses, 'loss treino 10 min')
                    if i == 2:
                        mean_tt_rna_30min.append(t_task_rna_mean)
                        mean_it_rna_30min.append(idle_rna_mean)
                        percent_complete_rna_30min.append(complete_percent_rna)
                        # ml.plot_losses(losses, 'loss treino 30 min')
                    if i == 3:
                        mean_tt_rna_1h.append(t_task_rna_mean)
                        mean_it_rna_1h.append(idle_rna_mean)
                        percent_complete_rna_1h.append(complete_percent_rna)
                        # ml.plot_losses(losses, 'loss treino 1h')
                    if i == 4:
                        mean_tt_rna_2h.append(t_task_rna_mean)
                        mean_it_rna_2h.append(idle_rna_mean)
                        percent_complete_rna_2h.append(complete_percent_rna)
                        # ml.plot_losses(losses, 'loss treino 2 h')
                
                prev_dur = duration * 3600
                if z == 2 :
                    print(task_q3f)

# =============================================================================
#                     ml.plot(complete_tasks3, t_task3, idle_time3, 'RNA treinamento = '
#                             + str(round(3600*sum(durations[0:i+1]), 2)) + '(s)' + ' = ' 
#                             + str(round(sum(durations[0:i+1]), 2)) + '(h)')
# =============================================================================
                    if failed3 != []:
                        if len(failed3) == 1:
                            print('The task', failed3[0], 'has failed.')
                        else:
                            print('A total of', len(failed3), 'tasks failed, they are: ', failed3)
                '''
                if len(complete_tasks3) == 0:
                    complete_percent_rna = 0
                else:
                    complete_percent_rna = round(len(complete_tasks3)/len(task_q3f)/2, 2)
                if len(t_task3) == 0:
                    t_task_rna_mean = 0
                else:
                    t_task_rna_mean = round(sum(t_task3)/len(t_task3), 2)
                if len(idle_time3) == 0:
                    idle_rna_mean = 999999
                else:
                    idle_rna_mean = round(sum(idle_time3)/len(idle_time3), 2)
                if i == 0:
                    mean_tt_rna_1min.append(t_task_rna_mean)
                    mean_it_rna_1min.append(idle_rna_mean)
                    percent_complete_rna_1min.append(complete_percent_rna)
                    # ml.plot_losses(losses, 'loss treino 1 min')
                    
                if i == 1:
                    mean_tt_rna_10min.append(t_task_rna_mean)
                    mean_it_rna_10min.append(idle_rna_mean)
                    percent_complete_rna_10min.append(complete_percent_rna)
                    # ml.plot_losses(losses, 'loss treino 10 min')
                if i == 2:
                    mean_tt_rna_30min.append(t_task_rna_mean)
                    mean_it_rna_30min.append(idle_rna_mean)
                    percent_complete_rna_30min.append(complete_percent_rna)
                    # ml.plot_losses(losses, 'loss treino 30 min')
                if i == 3:
                    mean_tt_rna_1h.append(t_task_rna_mean)
                    mean_it_rna_1h.append(idle_rna_mean)
                    percent_complete_rna_1h.append(complete_percent_rna)
                    # ml.plot_losses(losses, 'loss treino 1h')
                if i == 4:
                    mean_tt_rna_2h.append(t_task_rna_mean)
                    mean_it_rna_2h.append(idle_rna_mean)
                    percent_complete_rna_2h.append(complete_percent_rna)
                    # ml.plot_losses(losses, 'loss treino 2 h')
                '''
                print('Rede nº:',z+1, '-', round(durations[i] * 60,0),  
                      'minutos de treino completos')
                
        
        break
    #mo.ORT_solve(matrix, Tasks, task_q_not_split, n_a, cap, 0, w, d)
# =============================================================================
    mean_tt_edd.append(t_task_edd_mean)
    mean_it_edd.append(idle_edd_mean)
    percent_complete_edd.append(complete_percent_edd)
    mean_tt_ga.append(t_task_ga_mean)
    mean_it_ga.append(idle_ga_mean)
    percent_complete_ga.append(complete_percent_ga)

Dicio = [[mean_tt_edd, mean_tt_ga], [mean_it_edd, mean_it_ga],
          [percent_complete_edd, percent_complete_ga,]]
Dicio_indexes = [['Earliest Due Date', 'Algorítmo Genético'],['Earliest Due Date',
                  'Algorítmo Genético'],['Earliest Due Date', 'Algorítmo Genético']]
ml.create_excel(Dicio[0], 'tables/Tempo levado para completar uma tarefa' + str(z) + '.xlsx', 
                    'Valores', Dicio_indexes[0])
ml.create_excel(Dicio[1], 'tables/Tempo em ócio' + str(z) + '.xlsx', 
                    'Valores', Dicio_indexes[1])
ml.create_excel(Dicio[2], 'tables/Percentual de tarefas concluídas' + str(z) + '.xlsx', 
                    'Valores', Dicio_indexes[2])
# =============================================================================
#    Dicio = [[mean_tt_rna_1min,  mean_tt_rna_10min, mean_tt_rna_30min,
#                  mean_tt_rna_1h, mean_tt_rna_2h], [mean_it_rna_1min,
#                  mean_it_rna_10min, mean_it_rna_30min, mean_it_rna_1h, 
#                  mean_it_rna_2h], [percent_complete_rna_1min,
#                  percent_complete_rna_10min, percent_complete_rna_30min,
#                  percent_complete_rna_1h, percent_complete_rna_2h]]
#    Dicio_indexes = [['RNA treino de 1 min', 'RNA treino de 10 min',
#                      'RNA treino de 30 min', 'RNA treino de 1 h', 
#                      'RNA treino de 2 h'],['RNA treino de 1 min', 
#                      'RNA treino de 10 min', 'RNA treino de 30 min',
#                      'RNA treino de 1 h', 'RNA treino de 2 h'],['RNA treino de 1 min', 
#                      'RNA treino de 10 min', 'RNA treino de 30 min', 
#                      'RNA treino de 1 h', 'RNA treino de 2 h (%)']]
#    ml.create_excel(Dicio[0], 'tables/Tempo levado para completar uma tarefa' + str(z) + '.xlsx', 
#                    'Valores', Dicio_indexes[0])
#    ml.create_excel(Dicio[1], 'tables/Tempo em ócio' + str(z) + '.xlsx', 
#                    'Valores', Dicio_indexes[1])
#    ml.create_excel(Dicio[2], 'tables/Percentual de tarefas concluídas' + str(z) + '.xlsx', 
#                    'Valores', Dicio_indexes[2])

import sys
import os
import time
import random
import copy
import operator
import numpy as np
import math
import matplotlib.pyplot as plt
import sched, time

# MIN_PHEROMONE = 1
# MAX_PHEROMONE = 100000
K_BEST = 20
EVAPORATE_RATIO = 0.05


class Graph():
    def __init__(self, file):
        with open(file, 'r') as f:
            lines = f.read().splitlines()
            splited_lines = [line.split() for line in lines]
            self._buildGraph(splited_lines)
        
    def _buildGraph(self, lines):
        self.graph = {}
        self.INIT = '1'
        self.END = self._getEndNode(lines) 
        # print("Init:", self.INIT, " End:", self.END)
        for vertex in lines:
            self.addEdge(vertex[0], vertex[1], vertex[2])

    def _getEndNode(self, lines):
        max = 0
        for line in lines:
            if int(line[0]) > max:
                max = int(line[0])
        return str(max)

    def addEdge(self, ori, dest, weight):
        if(ori in self.graph):
            self.graph[ori][dest] = {}
            self._setWeight(ori, dest, weight)
            self._setPheromone(ori, dest, 1)
        else:
            self.graph[ori] = {}
            self.graph[ori][dest] = {}
            self._setWeight(ori, dest, weight)
            self._setPheromone(ori, dest, 1)

    def _setWeight(self, ori, dest, weight):
        self.graph[ori][dest]['weight'] = int(weight)

    def _getWeight(self, ori, dest):
        return self.graph[ori][dest]['weight']

    def _setPheromone(self, ori, dest, pheronome):
        # if pheronome <= MIN_PHEROMONE:
        #     self.graph[ori][dest]['pheromone'] = MIN_PHEROMONE
        # elif pheronome >= MAX_PHEROMONE:
        #     self.graph[ori][dest]['pheromone'] = MAX_PHEROMONE
        # else:
        self.graph[ori][dest]['pheromone'] = pheronome

    def _getPheromone(self, ori, dest):
        return self.graph[ori][dest]['pheromone']

    def _adjacentNeighbors(self, node):
        return [key for key in self.graph[node].keys()]
# print()


def update_pheromone(graph, path, path_fitness, best_path):
    origin = path[0]
    aux_path = path[1:]
    max_weight = 0
    for dest in aux_path:
        weight = graph._getWeight(origin, dest)
        max_weight = max(weight, max_weight)
        origin = dest
    
    pheromone_update_value = path_fitness/best_path
    
    origin = path[0]

    # print("PATH: ", aux_path)
    # time.sleep(20)

    # print("PATH FITNESS: ", path_fitness)

    for dest in aux_path:
        # Update pheromone
        pheromone = graph._getPheromone(origin, dest)
        pheromone += pheromone_update_value
        # pheromone += 3
        if origin == '1':
            # print(pheromone)
            pass
        graph._setPheromone(origin, dest, pheromone)
        origin = dest


def evaporate_pheromone(graph):
    """
    origin=path[0]
    aux_path=path[1:]
    for dest in aux_path:
        pheromone=graph._getPheromone(origin, dest)
        pheromone*=(1-EVAPORATE_RATIO)
        graph._setPheromone(origin, dest, pheromone)
        origin=dest
    """
    graph_path = graph.graph
    for origin in graph_path:
        # print("{}:".format(origin))
        for dest in graph_path[origin]:
            # print(dest, end=" ")
            curr_pheromone = graph._getPheromone(origin, dest)
            graph._setPheromone(origin, 
                                dest, 
                                curr_pheromone*(1-EVAPORATE_RATIO)
                                )
        # print()
        # time.sleep(2) 


def aco_iteration(graph, num_ants, best_ind=(None, None), elitism=False):
    paths = []
    fitness_list = []
    fitness_value_list = []
    average_fitness = None
    worst_fitness = None

    for _ in range(num_ants):
        path_i = generate_path(graph)
        if(path_i is not None):
            paths.append(path_i)

    # print(paths)
    max_fit = 0
    best_path = None

    for path in paths:
        fitness = calculate_fitness(graph, path)
        fitness_list.append((path, fitness))
        fitness_value_list.append(fitness)
        if(max_fit < fitness):
            max_fit = fitness
            best_path = path
        if worst_fitness is None or worst_fitness > fitness:
            worst_fitness = fitness    


        # break

    average_fitness = np.array(fitness_value_list).mean()

    """print("Best: {} Worst: {} Average: {}".format(
                                                  max_fit, 
                                                  worst_fitness, 
                                                  average_fitness
                                                  )) 

    print("Max fitness: ", max_fit)
    # print("Path: ", path)

    # print("ALL FITNESS:", fitness_value_list)
    """
    best_fitness_values = []
    best_fitness = []
    for _ in range(K_BEST):
        if len(fitness_value_list) > 0:
            best_fitness_values.append(max(fitness_value_list))
            fitness_value_list.remove(max(fitness_value_list))
        # pass
    # print(fitness_list)    


    # print("Best Individual", best_ind)

    for path in fitness_list:
        # print("Path: ", path)
        if(path[1] in best_fitness_values):
            if elitism is True:
                if best_ind[1] is None or path[1] > best_ind[1]:
                    best_fitness.append(path)
            else:
                best_fitness.append(path)

    for path in fitness_list:
        if(path[1] in best_fitness_values):
            if best_ind[1] is None:
                best_ind = path
            elif best_ind[1] < path[1]:
                best_ind = path

    # for path in fitness_list:
        # pass
    """
    print("Updatable paths")
    for path in best_fitness:
        print(path[1], end=" ")
    print("\n")
    """
    # print("Best: ", best_fitness)
    for path in best_fitness:
        index = best_fitness_values.index(path[1])
        index = index if index > 0 else 1
        update_pheromone(graph, path[0], path[1], best_fitness_values[0])
    # print()

    evaporate_pheromone(graph)

    return best_ind, max_fit, average_fitness, worst_fitness


def calculate_fitness(graph, path):
    # Get first node and iretate over the others
    origin = path[0]
    path = path[1:]

    # set initial fitness
    fitness = 0

    # Fitness = Sum of all edges ( path(u,v) ) of path
    for dest in path:
        fitness += graph._getWeight(origin, dest)
        origin = dest
    
    return fitness

 
def generate_path(graph):
    start_node = graph.INIT
    end_node = graph.END

    path = [start_node]
    actual_node = start_node

    while(actual_node != end_node):
        probabilities = neighbor_probabilities(graph, actual_node)

        actual_node = getNextNode(probabilities)

        validation = 0
        while(actual_node in path):
            validation += 1
            actual_node = getNextNode(probabilities)
            if(validation > 50):
                return None

        path.append(actual_node)

    #print(path)    
    return path


def getNextNode(probabilities):
    # rand_val = random.random()
    # r = random.uniform(0, sum(probabilities.values()))
    keys = list(probabilities.keys())
    values = list(probabilities.values())
    # print(probabilities)
    # print("Keys: ", keys)
    # print("Values: ", values)
    # escolha = np.random.choice(keys, 1, replace=False, p=values)[0]
    # print("Escolhi: ", escolha)
    # print("Prob: ", probabilities[escolha])
    # time.sleep(5)
    return np.random.choice(keys, 1, p=values)[0]

    # print("R: ", r)
    # print("Rand:", rand_val)
    total = 0
    for k, v in probabilities.items():
        total += v
        if r < total:
            # print("Escolhi: ", k, " com probabilidade: ", v)
            # time.sleep(10)
            return k
    return k


def neighbor_probabilities(graph, actual_node):
    neighbor_list = graph._adjacentNeighbors(actual_node)

    probabilities = {}
    total_probabilties = 0
    # print("ACTUAL NODE: ", actual_node)
    for neighbor in neighbor_list:
        # print(neighbor)
        neighbor_weight = graph._getWeight(actual_node, neighbor)
        neightbor_pheromone = graph._getPheromone(actual_node, neighbor)
        
        # if(actual_node == "1"):
        #     # print("Neighbor:{}:W-{}:P-{:.3f}".format(
        #     #                                          neighbor, 
        #     #                                          neighbor_weight,
        #     #                                          neightbor_pheromone)
        #     #                                          )
        #     pass

        probabilities[neighbor] = neighbor_weight*neightbor_pheromone

        total_probabilties += probabilities[neighbor]
    # total_probabilties = sum(float(probabilities[d]) for d in probabilities.keys())

    for prob in probabilities:
        probabilities[prob] /= total_probabilties

    # if(actual_node == "1"):
    #     # print()
    #     # print("Probs of {}".format(actual_node))
    #     for prob in probabilities:
    #         # print("{}:{:.3f}".format(prob, probabilities[prob]), end="  | ")
    #         pass
    #     # print("\n\n")
    #     # time.sleep(3)
    # # sort probabilities
    # # sorted_probs = sorted(probabilities.items(), key=operator.itemgetter(1))

    return probabilities


def ACO(graph, num_iterations):
    cont = 0
    max_fit_list = []
    avg_fit_list = []
    worst_fit_list = []
    cont_list = []
    best_ind = (None, None)

    for _ in range(num_iterations):
        # print("CONTADOR: {}".format(cont))
        best_ind, max_fit, avg_fit, worst_fit = aco_iteration(graph, 600, best_ind, elitism=False)

        # Values of each iteration stored
        max_fit_list.append(max_fit)
        avg_fit_list.append(avg_fit)
        worst_fit_list.append(worst_fit)

        # Iteration number
        cont_list.append(cont)
        cont += 1

        print("{},{:.3f},{},{}".format(max_fit, avg_fit, worst_fit, cont))
        # print(max_fit, avg_fit, worst_fit, cont)

        # Graph plot
        # plt.plot(cont_list, max_fit_list, 'r--', cont_list, avg_fit_list, 'b--', cont_list, worst_fit_list, 'g--')
        # plt.pause(1)
        # plt.gcf().clear()
        # plt.close()

graph = Graph('./datasets/graph1.txt')
# ACO(graph, 1)
# print(graph.graph)
# aco_iteration(graph, 300)
ACO(graph, 5000)
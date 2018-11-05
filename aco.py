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

MIN_PHEROMONE = 0.1
MAX_PHEROMONE = 1000
K_BEST = 100
EVAPORATE_RATIO = 0.05
NUM_ANTS = 600
NUM_ITERATIONS = 1000


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
        for vertex in lines:
            self._newNode(vertex[0], vertex[1], vertex[2])

    def _getEndNode(self, lines):
        max = 0
        for line in lines:
            if int(line[0]) > max:
                max = int(line[0])
        return str(max)

    def _newNode(self, ori, dest, weight):
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
        if pheronome <= MIN_PHEROMONE:
            self.graph[ori][dest]['pheromone'] = MIN_PHEROMONE
        elif pheronome >= MAX_PHEROMONE:
            self.graph[ori][dest]['pheromone'] = MAX_PHEROMONE
        else:
            self.graph[ori][dest]['pheromone'] = pheronome

    def _getPheromone(self, ori, dest):
        return self.graph[ori][dest]['pheromone']

    def _adjacentNeighbors(self, node):
        neighbors = []
        for keys in self.graph[node].keys():
            neighbors.append(keys)
        return neighbors


def update_pheromone(graph, path, path_fitness, best_path):
    """
        Update phermone based on path's fitness
    """
    origin = path[0]
    aux_path = path[1:]
    max_weight = 0
    for dest in aux_path:
        weight = graph._getWeight(origin, dest)
        max_weight = max(weight, max_weight)
        origin = dest

    pheromone_update_value = path_fitness/best_path

    origin = path[0]

    for dest in aux_path:
        # Update pheromone
        pheromone = graph._getPheromone(origin, dest)
        pheromone += pheromone_update_value
        graph._setPheromone(origin, dest, pheromone)
        origin = dest


def evaporate_pheromone(graph):
    """
        Evaporate pheromone from the ACO's graph
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


def aco_iteration(graph, num_ants, best_ind=(None, None), elitism=False):
    """
        Function responsible for generating a new iteration of the ACO
        algorithm.

        graph: ACO graph
        num_ants: Number of ants
        best_ind: Best ant so far
        elitism: Run on elistim mode (Didnt work pretty well)
    """
    paths = []
    fitness_list = []
    fitness_value_list = []
    average_fitness = None
    worst_fitness = None

    # Generate a path for every ant
    for _ in range(num_ants):
        path_i = generate_path(graph)
        if(path_i is not None):
            paths.append(path_i)

    max_fit = 0
    best_path = None

    # Calculate all path's fitness and stores the worst and the best fitnesses
    for path in paths:
        fitness = calculate_fitness(graph, path)
        fitness_list.append((path, fitness))
        fitness_value_list.append(fitness)
        if(max_fit < fitness):
            max_fit = fitness
            best_path = path
        if worst_fitness is None or worst_fitness > fitness:
            worst_fitness = fitness

    average_fitness = np.array(fitness_value_list).mean()

    best_fitness_values = []
    best_fitness = []
    for _ in range(K_BEST):
        if len(fitness_value_list) > 0:
            best_fitness_values.append(max(fitness_value_list))
            fitness_value_list.remove(max(fitness_value_list))

    # Get all K-best fitness
    for path in fitness_list:
        if(path[1] in best_fitness_values):
            if elitism is True:
                if best_ind[1] is None or path[1] > best_ind[1]:
                    best_fitness.append(path)
            else:
                best_fitness.append(path)

    # Get all K-best paths
    for path in fitness_list:
        if(path[1] in best_fitness_values):
            if best_ind[1] is None:
                best_ind = path
            elif best_ind[1] < path[1]:
                best_ind = path

    # Update pheromone from all K-best paths, accordingly to the path fitness
    for path in best_fitness:
        index = best_fitness_values.index(path[1])
        index = index if index > 0 else 1
        update_pheromone(graph, path[0], path[1], best_fitness_values[0])

    # Evaporate pheromone from all graph
    evaporate_pheromone(graph)

    return best_ind, max_fit, average_fitness, worst_fitness


def calculate_fitness(graph, path):
    """
        Calculates path's fitness
    """
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
    """
        Generates a path from the Start Node to the End Node
        
        The path generated is simple. In other words, the path
        doesn't contain duplicates
    """
    # Sets the start and end nodes
    start_node = graph.INIT
    end_node = graph.END

    path = [start_node]
    actual_node = start_node

    # Generate a random neighbor until didn't reach the end
    while(actual_node != end_node):
        probabilities = neighbor_probabilities(graph, actual_node)

        actual_node = getNextNode(probabilities)
        
        # Prevents generating a duplicate node
        validation = 0
        while(actual_node in path):
            validation += 1
            actual_node = getNextNode(probabilities)
            if(validation > 50):
                return None

        path.append(actual_node)

    return path


def getNextNode(probabilities):
    """
        Generates a random node based on probabilities previously calculated
    """
    keys = list(probabilities.keys())
    values = list(probabilities.values())
    return np.random.choice(keys, 1, p=values)[0]


def neighbor_probabilities(graph, actual_node):
    """
        Calculate the probabilities of choosing a neighbor
        from the actual_node
    """
    # Get all adjacent neighbors
    neighbor_list = graph._adjacentNeighbors(actual_node)

    probabilities = {}
    total_probabilties = 0

    # Calculates the probabilities of each neighbor
    for neighbor in neighbor_list:
        neighbor_weight = graph._getWeight(actual_node, neighbor)
        neightbor_pheromone = graph._getPheromone(actual_node, neighbor)
        
        probabilities[neighbor] = math.pow((
                                            neighbor_weight*neightbor_pheromone
                                            ),
                                           1  # Exponent number
                                           )

        total_probabilties += probabilities[neighbor]

    for prob in probabilities:
        probabilities[prob] /= total_probabilties

    return probabilities


def ACO(graph, num_iterations):
    cont = 0
    max_fit_list = []
    avg_fit_list = []
    worst_fit_list = []
    cont_list = []
    best_ind = (None, None)

    for _ in range(num_iterations):
        best_ind, max_fit, avg_fit, worst_fit = aco_iteration(
                                                              graph,
                                                              NUM_ANTS,
                                                              best_ind,
                                                              elitism=False
                                                              )

        # Values of each iteration stored
        max_fit_list.append(max_fit)
        avg_fit_list.append(avg_fit)
        worst_fit_list.append(worst_fit)

        # Iteration number
        cont_list.append(cont)
        cont += 1

        print("{},{:.3f},{},{}".format(max_fit, avg_fit, worst_fit, cont))

graph = Graph('./datasets/graph1.txt')
ACO(graph, NUM_ITERATIONS)

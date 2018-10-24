import sys
import os
import time
import random
import copy
import operator

MIN_PHEROMONE=1
MAX_PHEROMONE=50
K_BEST=10
EVAPORATE_RATIO=0.1

class Graph():
    def __init__(self, file):
        
    	with open(file, 'r') as f:
            lines = f.read().splitlines()
            splited_lines = [line.split() for line in lines]
            self._buildGraph(splited_lines)
        
    def _buildGraph(self, lines):
        self.graph={}
        self.INIT=lines[0][0]
        self.END=lines[-1][0]
        print("Init:", self.INIT, " End:",self.END)
        for vertex in lines:
            self.addEdge(vertex[0], vertex[1], vertex[2])

    def addEdge(self, ori, dest, weight):
        if(ori in self.graph):
            self.graph[ori][dest]={}
            self._setWeight(ori, dest, weight)
            self._setPheromone(ori, dest, 1)
        else:
            self.graph[ori]={}
            self.graph[ori][dest]={}
            self._setWeight(ori, dest, weight)
            self._setPheromone(ori, dest, 1)
        
    def _setWeight(self, ori, dest, weight):
        self.graph[ori][dest]['weight']=int(weight)
    
    def _getWeight(self, ori, dest):
        return self.graph[ori][dest]['weight']
    
    def _setPheromone(self, ori, dest, pheronome):
        if pheronome <= MIN_PHEROMONE:
            self.graph[ori][dest]['pheromone']=MIN_PHEROMONE
        elif pheronome >= MAX_PHEROMONE:
            self.graph[ori][dest]['pheromone']=MAX_PHEROMONE
        else:
            self.graph[ori][dest]['pheromone']=pheronome
    
    def _getPheromone(self, ori, dest):
        return self.graph[ori][dest]['pheromone']
    
    def _adjacentNeighbors(self, node):
        return [key for key in self.graph[node].keys()]
#print()

def update_pheromone(graph, path, path_fitness):
    origin=path[0]
    aux_path=path[1:]
    
    max_weight=0

    for dest in aux_path:
        weight=graph._getWeight(origin, dest)
        max_weight=max(weight, max_weight)
        origin=dest
    
    pheromone_update_value = float(max_weight)/path_fitness

    origin=path[0]
    for dest in aux_path:
        #Update pheromone
        pheromone=graph._getPheromone(origin, dest)
        pheromone*=(1+pheromone_update_value)
        graph._setPheromone(origin, dest, pheromone)

        origin=dest

def evaporate_pheromone(graph, path):
    origin=path[0]
    aux_path=path[1:]
    for dest in aux_path:
        pheromone=graph._getPheromone(origin, dest)
        pheromone*=(1-EVAPORATE_RATIO)
        graph._setPheromone(origin, dest, pheromone)
        origin=dest



def aco_iteration(graph, num_ants):
    paths=[]
    fitness_list=[]
    fitness_value_list=[]

    for _ in range(num_ants):
        path_i = generate_path(graph)
        if(path_i != None):
            paths.append(path_i)

    #print(paths)
    max_fit=0
    best_path=None

    for path in paths:
        fitness=calculate_fitness(graph, path)
        fitness_list.append((path, fitness))
        fitness_value_list.append(fitness)
        if(max_fit < fitness):
            max_fit=fitness
            best_path=path
        #break

    print("Max fitness: ", max_fit)
    #print("Path: ", path)

    #print("ALL FITNESS:", fitness_value_list)

    best_fitness_values=[]
    for _ in range(K_BEST):
        best_fitness_values.append(max(fitness_value_list))
        fitness_value_list.remove(max(fitness_value_list))
        #pass
    #print(fitness_list)    
    
    #print("BEST K FITNESS:", best_fitness_values)

    for path in fitness_list:
        if(path[1] not in best_fitness_values):
            fitness_list.remove(path)

    for path in fitness_list:
        update_pheromone(graph, path[0], path[1])
        #break

    for path in fitness_list:
        evaporate_pheromone(graph, path[0])

    return "Iteration sucessfull"

def calculate_fitness(graph, path):
    #Get first node and iretate over the others
    origin = path[0]
    path = path[1:]

    #set initial fitness
    fitness=0

    #Fitness = Sum of all edges ( path(u,v) ) of path
    for dest in path:
        fitness+=graph._getWeight(origin, dest)
        origin=dest
    
    return fitness

def generate_path(graph):
    start_node=graph.INIT
    end_node=graph.END

    path=[start_node]
    actual_node=start_node

    while(actual_node != end_node):
        probabilities=neighbor_probabilities(graph, actual_node)

        actual_node = getNextNode(probabilities)

        validation=0
        while(actual_node in path):
            validation+=1
            actual_node = getNextNode(probabilities)
            if(validation > 50):
                return None

        path.append(actual_node)

    #print(path)    
    return path

def getNextNode(probabilities):
    rand_val = random.random()
    #print("Rand:", rand_val)
    total = 0
    for k, v in probabilities.items():
        total += v
        if rand_val <= total:
            return k
    assert False, 'unreachable'

def neighbor_probabilities(graph, actual_node):
    neighbor_list=graph._adjacentNeighbors(actual_node)

    probabilities={}
    total_probabilties=0

    for neighbor in neighbor_list:
        neighbor_weight=graph._getWeight(actual_node, neighbor)
        neightbor_pheromone=graph._getPheromone(actual_node, neighbor)
        #print("Weight:", neighbor_weight, " Pheromone:", neightbor_pheromone)

        probabilities[neighbor] = neighbor_weight*neightbor_pheromone

        total_probabilties+=probabilities[neighbor]
    
    for prob in probabilities:
        probabilities[prob] /= total_probabilties
    
    #sort probabilities
    #sorted_probs = sorted(probabilities.items(), key=operator.itemgetter(1))
    
    return probabilities

def ACO(graph, num_iterations):
    for _ in range(num_iterations):
        aco_iteration(graph, 300)


graph = Graph('./datasets/graph1.txt')
#ACO(graph, 1)
#print(graph.graph)
#aco_iteration(graph, 300)
ACO(graph, 500)
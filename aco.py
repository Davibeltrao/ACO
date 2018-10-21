import sys
import os
import time
import random
import operator

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
        self.graph[ori][dest]['pheromone']=pheronome
    
    def _getPheromone(self, ori, dest):
        return self.graph[ori][dest]['pheromone']
    
    def _adjacentNeighbors(self, node):
        return [key for key in self.graph[node].keys()]
#print()

def ACO(graph, num_ants):
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

    print(path)    
    return path

def getNextNode(probabilities):
    rand_val = random.random()
    print("Rand:", rand_val)
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
    
    sorted_probs = sorted(probabilities.items(), key=operator.itemgetter(1))
    #print(sorted_probs)

    return probabilities


graph = Graph('./datasets/graph1.txt')
ACO(graph, 1)
#print(graph.graph)
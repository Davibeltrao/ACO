import sys
import os
import time
import random

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
        pass    
    
    def _setWeight(self, ori, dest, weight):
        self.graph[ori][dest]['weight']=weight
    
    def _setPheromone(self, ori, dest, pheronome):
        self.graph[ori][dest]['pheromone']=pheronome
#print()

def ACO(graph, num_ants):
    start_node=graph.INIT
    end_node=graph.END

    path=[start]
    actual_node=start

    while(actual_node != end_node):
        


graph = Graph('./datasets/graph1.txt')
#print(graph.graph)
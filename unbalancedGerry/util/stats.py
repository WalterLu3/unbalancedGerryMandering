'''
Methods related to getting map value including
1. population value
2. compactness scores
    a. connected componenet score
    b. disperse score ("the score related to the object in Hess model (not exactly the same)")
    c. Polsby-Popper score  # to be added
'''

import pickle as pk
import random
import geopandas as gpd
import networkx as nx
from ..file.fileIO import *
from ..generation.generateMap import EuclideanDistance, findAdj

def calculatePop(assign,vote):
    """
    Calculate the population for each district

    Parameter
    ---------
    assign : dict, 
        the assigned map

    vote : dict,
        dictionary of a given vote year

    Return
    ------
    population : dict,
        dictionary that contains the population of each district
    """

    # load map
    with open(assign,'rb') as f:
        data = pk.load(f)['final']

    # aggregate the votes of each district
    districtVote = {}
    assign = {}
    for n in data.nodes:
        d = data.nodes[n]['assign']
        for n1 in n.split('&'):
            if d not in districtVote:
                districtVote[d] = 0
            districtVote[d] += vote[n1]['DEM'] + vote[n1]['REP']
            
    return districtVote

def disperseScore(inFile, position):
    """
    Calculate the compactness in terms of disperse score

    Parameter
    ---------
    inFile : dict,
        Assignment map.

    position: dict,
        coordinates of every unit

    Return
    ------
    disperseScore : float,
        a value that evaluates the compactness in terms of disperse score
    """
    G = inFile['final']
    
    coord = {} ## record the coordinates for each district
    avgCoord = {} ## record the centroid
    compactScore = {} ## record the compactness score
        
    for n in G.nodes:
        d = G.nodes[n]['assign']
        if d not in coord:
            coord[d] = []
            avgCoord[d] = None
            compactScore[d] = 0
        coord[d].append(position[n])
        #break

    avgPos = inFile['posDistrict']
    for d in avgPos:
        X = avgPos[d][0]
        Y = avgPos[d][1]
        avgCoord[d] = (X,Y)

    for d in avgPos:
        for e in coord[d]:
            compactScore[d] += EuclideanDistance(avgCoord[d],e)
    
    disperseScore = sum(compactScore.values())
    return disperseScore

def disconnectionScore(inFile):

    """
    Calculate the compactness in terms of disconnnection score

    Parameter
    ---------
    inFile : dict,
         a dictonary where the 'final' dictionary contains the information for assignment.

    Return
    ------
    disconnectionScore : float,
        a value that evaluates the compactness in terms of disconnection score
    """
    # get the pair of nodes on the border
    a,b = findAdj(inFile['final']) # adjacency is already encoded in the newwork graph

    max_discontiguity = 0
    sum_discontiguity = 0

    dNum = len(inFile['posDistrict'].keys())

    # identify the nodes in every district (H(d,f))
    sourceNodes = {}
    for i in range(1,dNum + 1):
        sourceNodes[i] = [
                node
                for node, data
                in inFile['final'].nodes(data=True)
                if data.get("assign") == i
            ]

    used = set()

    # only need to remove the nodes on the border
    for e in b :
        for i in range(2):
            if e[0][i] in used:
                continue
            else:
                used.add(e[0][i])
            # remove node
            subNodes = set(sourceNodes[e[1][i]]) - set([e[0][i]])
            tempG = inFile['final'].subgraph(subNodes)
            connected_comp = list(nx.connected_components(tempG))

            # check if it is still connectd
            if len(connected_comp) == 1:
                continue
            # if not, then calculate the disconnection score
            else:
                tempMax = 0
                connected_comp.sort(key = len)
                for cc in connected_comp[:-1]:
                    tempMax += len(cc)

                max_discontiguity = max(tempMax,max_discontiguity)
                sum_discontiguity += tempMax

    # sum of all branches is also calculated
    return max_discontiguity


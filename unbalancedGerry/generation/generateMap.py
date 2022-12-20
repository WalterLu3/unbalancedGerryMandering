"""
Inbalanced Map Generation Model.
"""

# imbalanced Gerrymandering
# reassign the connected components when the graph is not connected 
# Gerrymandering with different population ratio
import pickle as pk
import numpy as np
import networkx as nx
import matplotlib.patches as mpatches
from matplotlib import colors
import random
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import time
import os,sys


def merge(G, nodeA, nodeB):
    """
    Merge nodeA and nodeB in the networkX graph G. The change is done in-place for G.

    Parameters
    ----------
    G : networkx.classes.graph.Graph,
        initial graph created with networkx package.

    nodeA: string,
        name of the first node for merging in G.

    nodeB:
        name of the second node for merging in G.

    Returns
    -------
    name : str,
        the new name for the merged nodes.

    Notes
    -----
    The merge is done in-place in the input graph G.
    """
    
     # Get all adjacent nodes of two nodes
    assert (nodeA in G.nodes) and (nodeB in G.nodes), "Input node(s) does not exist!"
    new_adj = []
    new_adj += list(G[nodeA])
    # check if node A and node B are adjacent
    assert (nodeB in new_adj) , "Input nodes are not adjacent!"

    new_adj += G[nodeB]
    # Create the new node with combined name
    name = str(nodeA) + '/' + str(nodeB)
    # Add new edges
    G.add_edges_from([(p, name) for p in new_adj])
    # Remove old nodes and the edges related to old nodes will also be removed
    G.remove_nodes_from([nodeA, nodeB])
    
    return name

def EuclideanDistance(A,B):
    """
    Calculate the Euclidean distance of two coordinates.

    Parameters
    ----------
    A : list (or tuple),
        coordinate of the first point.

    B : list (or tuple),
        coordinate of the second point.

    Returns
    -------
    result : float,
        Euclidean Distance calculated by using A and B.
    """
    # return Euclidean distance
    result = ((A[0]-B[0])**2 + (A[1]-B[1])**2)**(1/2)
    return result

def generateInit(adjList, position, districtNum):
    """
    Generate initial graph by keeping merging a random node and its closest neighbor.

    Parameters
    ----------
    adjList : list,
        a list of tuples where each tuple represents the adjacency of a pair of node. For examaple, tuple (a,b)
        means node a and node b are adjacent.)

    position : dictionary,
        a dictionary that contains the coordinate for every node that appears in adjList.

    districtNum : int,
        district number we want to have for a given state.

    Returns
    -------
    G : networkx.classes.graph.Graph,
        initial graph that has the district number satisfying districtNum.

    district : dictionary,
        district assignment stored in dictionary form.

    """
    # create a graph for according to the input adjacency
    G = nx.Graph()
    G.add_edges_from(adjList)

    # in order to make d districts, we need to run merge for n-d times
    originalNodeNum = len(G.nodes)
    for i in range(originalNodeNum-districtNum):
        # randomly choose one node
        currentNodeNum = originalNodeNum - i
        idx = random.randrange(0, currentNodeNum)
        nodeA = list(G.nodes)[idx]

        # find the nearest node
        centroidA = [0,0,0] #first two are coordinate, and the last one is number of merged nodes
        
        for j in nodeA.split("/"):
            centroidA[0] += position[j][0]
            centroidA[1] += position[j][1]
            centroidA[2] += 1
            
        centroidA[0] = centroidA[0]/centroidA[2]
        centroidA[1] = centroidA[1]/centroidA[2]
        
        nearest = None
        distance = np.inf
        for i in G[nodeA]:
            temp_distance = EuclideanDistance(centroidA,position[i])
            if temp_distance < distance:
                distance = temp_distance
                nearest = i
        
        # start merge
        newNode = merge(G, nodeA, nearest)
        # update information
        position[newNode] = sum([np.array(position[n]) for n in newNode.split('/')])/ \
                            len(newNode.split('/'))

    # store the merge result in alternative form
    district = {}
    dNum = 1
    for n in G.nodes:
        for i in n.split("/"):
            district[i] = dNum
        dNum += 1

    return G, district

def findAdj(GPop):
    """
    Detect the set of adjacent districts in the input map. (The assignment information is stored in the ['assign'] attributes
    of the network nodes)

    Parameters
    ----------
    GPop : networkx.classes.graph.Graph,
        the network graph where each node's 'assign' attribute contains the district assginment information.

    Returns
    -------
    listDistrict : list,
        list of tuples where each tuple contains a pair of adjacent districts

    adjNodes : list,
        list of tuples where each tuple is in the form (A, (B,C)). A is a unit on the border and A is assigned
        to district B. A is also adjacent to units that belongs to district nodeBdistrict 
    """
    adjDistricts = []
    adjNodes = []
    for i in GPop.edges:
        nodeADistrict = GPop.nodes[i[0]]["assign"]
        nodeBDistrict = GPop.nodes[i[1]]["assign"]
        if nodeADistrict != nodeBDistrict:
            adjDistricts.append((nodeADistrict,nodeBDistrict))
            adjNodes.append((i,(nodeADistrict,nodeBDistrict)))

    listDistrict = list(set(adjDistricts)) 
    return listDistrict, adjNodes

def findPopulation(GPop,districtNum):
    """
    Given a network graph, and the number of districts, calculates the population of each district.

    Parameters
    ----------
    GPop : networkx.classes.graph.Graph.
        the network graph where each node's 'assign' attribute contains the district assginment information.

    districtNum : int,
        district number we want to have for a given state.

    Returns
    -------
    popDistricts : dict,
        a dictionary that contains the population for each district
    """
    popDistrict = {}
    for i in range(1,districtNum+1):
        total = 0
        for n in GPop.nodes:
            if GPop.nodes[n]["assign"] == i:
                total += GPop.nodes[n]["population"]
        popDistrict[i] = total
    return popDistrict

def findPosition(GPop,districtNum,position):
    """
    Given a network graph, and the number of districts, calculates the population of each district.

    Parameters
    ----------
    GPop : networkx.classes.graph.Graph.
        the network graph where each node's 'assign' attribute contains the district assginment information.

    districtNum : int,
        district number we want to have for a given state.

    position : dict,
        the coordinate of every unit

    Returns
    -------
    positionDistricts : dict,
        a dictionary that contains the coordinate for each district. Each district has a list [A,B,C]. A is the 
        x coordinate, B is the y coordinate, and C is the number of units in that district. (Storing C allows
        us to speed up the calculation process later on.
    """
    positionDistrict = {}
    for i in range(1,districtNum+1):
        positionDistrict[i] = [0,0,0]
    for i in GPop.nodes:
        positionDistrict[GPop.nodes[i]["assign"]][0] +=  position[i][0]
        positionDistrict[GPop.nodes[i]["assign"]][1] +=  position[i][1]
        positionDistrict[GPop.nodes[i]["assign"]][2] +=  1
    
    for i in range(1,districtNum+1):
        positionDistrict[i] = [positionDistrict[i][0]/positionDistrict[i][2], \
                               positionDistrict[i][1]/positionDistrict[i][2], \
                               positionDistrict[i][2]]
    return positionDistrict

# def returnSecond(x):
    
#     return x[1]

def changPopAndPos(popDistrict,posDistrict,sourceDistrict,destinationDistrict,unit, position,population):

    """
    Recalculate population and position of each district after moving a unit from sourceDistrict to destinationDistrict.

    Parameters
    ----------

    popDistricts : dict,
        a dictionary that contains the population for each district

    posDistricts : dict,
        a dictionary that contains the coordinate for each district. Each district has a list [A,B,C]. A is the 
        x coordinate, B is the y coordinate, and C is the number of units in that district.

    sourceDistrict : int,
        the district that will give one of its border unit to destinationDistrict

    destinationDistrict : int,
        the district that will receive one of sourceDistrict's border unit


    Returns
    -------
    None

    (The change will be done in-place for popDistrict and posDistrict)
    """
    # adjust population
    popDistrict[sourceDistrict] -= population[unit]
    popDistrict[destinationDistrict] += population[unit]

    #adjust position
    sourceX = posDistrict[sourceDistrict][0] *  posDistrict[sourceDistrict][2]
    sourceY = posDistrict[sourceDistrict][1] *  posDistrict[sourceDistrict][2]

    destX = posDistrict[destinationDistrict][0] *  posDistrict[destinationDistrict][2]
    destY = posDistrict[destinationDistrict][1] *  posDistrict[destinationDistrict][2]

    posDistrict[sourceDistrict][0] = sourceX - position[unit][0]    
    posDistrict[sourceDistrict][1] = sourceY - position[unit][1]   
    posDistrict[sourceDistrict][2] = posDistrict[sourceDistrict][2] - 1

    posDistrict[destinationDistrict][0] = destX + position[unit][0]    
    posDistrict[destinationDistrict][1] = destY + position[unit][1]   
    posDistrict[destinationDistrict][2] = posDistrict[destinationDistrict][2] + 1

    posDistrict[sourceDistrict][0] = posDistrict[sourceDistrict][0]/posDistrict[sourceDistrict][2]
    posDistrict[sourceDistrict][1] = posDistrict[sourceDistrict][1]/posDistrict[sourceDistrict][2]

    posDistrict[destinationDistrict][0] = posDistrict[destinationDistrict][0]/posDistrict[destinationDistrict][2]
    posDistrict[destinationDistrict][1] = posDistrict[destinationDistrict][1]/posDistrict[destinationDistrict][2]    


def adjustingProcess(GPop, districtNum, position, population, weights, interval, iterations = 1000):
    """
    Adjusting the current map so that it is feasible in terms of population and contiguity.

    Parameters
    ----------
    GPop : networkx.classes.graph.Graph.
        the current network graph where each node's 'assign' attribute contains the district assginment information.

    districtNum : int,
        district number we want to have for a given state.

    position : dict,
        the coordinate of every unit
    
    population : dict,
        the population of every unit
    weights : dictionary,
        a dictionary of int that contains the population weight for each district

    interval : float,
        difference bound that determines the stopping condition

    iterations : int,
        max iteration of the adjusting phase


    Returns
    -------
    GPop : networkx.classes.graph.Graph,
        the final assignment graph. (Changes are done in-place in this function)
    
    popDistrict : dict,
        final population of every district
    
    posDistrict : dict,
        final position of every district

    converge : bool,
        if converges True, else false
    
    """
    max_disparity = []

    # start population adjustment
    lastInfeasible = None
    last_change = None
    last_discontiguous = []
    for i in range(iterations):
        # find the adjacency information in terms of districts
        adjDistricts, adjNodes= findAdj(GPop)

        # calculate the population and position information at the start once
        if i == 0 :
            popDistrict = findPopulation(GPop,districtNum)
            posDistrict = findPosition(GPop,districtNum,position)

        
        # identify the district pair with maximum population disparity
        largestDiff = 0
        largestAdj = None
        tempRecord = []
        for j in adjDistricts:
            absDiff = abs(popDistrict[j[0]]/weights[j[0]] - popDistrict[j[1]]/weights[j[1]]) #
            tempRecord.append(absDiff)
            if absDiff > largestDiff:
                largestDiff = absDiff
                largestAdj = j
        max_disparity.append(largestDiff)

        # print the disparity value every 50 iterations
        if i%50 == 0:
            print(i,largestDiff)

        # if the disparity value is less than the given bound, we can end
        if largestDiff < interval:
            break


        # identify the overpopulated and the underpopulated districts of the maximum disparity pair
        sourceDistrict = None
        destinationDistrict = None
        
        if popDistrict[largestAdj[0]]/weights[largestAdj[0]] > popDistrict[largestAdj[1]]/weights[largestAdj[1]]:
            sourceDistrict = largestAdj[0]
            destinationDistrict = largestAdj[1]
        else:
            sourceDistrict = largestAdj[1]
            destinationDistrict = largestAdj[0]
            
        # get all the inner border points
        sourceNodes = [
            node
            for node, data
            in GPop.nodes(data=True)
            if data.get("assign") == sourceDistrict
        ]
        
        borderPoints = []
        for a in adjNodes:
            if sourceDistrict in a[1] and destinationDistrict in a[1]:
                for n in a[0]:
                    if n in sourceNodes:
                        borderPoints.append(n)

        # identify the border unit that maximizes the compactness                
        tempDist = []
        for p in borderPoints:     
            tempDist.append((p,
                            EuclideanDistance(position[p],posDistrict[destinationDistrict]) - \
                            EuclideanDistance(position[p],posDistrict[sourceDistrict])))
            
        tempDist = sorted(tempDist,key = lambda x: x[1], reverse=False)

        removeNode = tempDist[0][0]
        subNodes = set(sourceNodes) - set([removeNode])
        tempG = GPop.subgraph(subNodes)

        # assign that chosen point
        GPop.nodes[removeNode]["assign"] = destinationDistrict
        changPopAndPos(popDistrict,
                        posDistrict,
                        sourceDistrict,
                        destinationDistrict,
                        unit = removeNode,
                        position = position, 
                        population = population)

        # check if connected, if not then assign the connected the smallest few components also
        if not nx.is_connected(tempG):
            connected_comp = list(nx.connected_components(tempG))
            connected_comp.sort(key = len)
            for cc in connected_comp[:-1]:
                for n in cc:
                    GPop.nodes[n]["assign"] = destinationDistrict
                    changPopAndPos(popDistrict,
                            posDistrict,
                            sourceDistrict,
                            destinationDistrict,
                            unit =n,
                            position = position, 
                            population = population)

        
    ## see if converges
    converge = True
    if i == iterations:
        converge = False
    # align with the assignment 
    assignDict = nx.get_node_attributes(GPop, "assign")
    return GPop, popDistrict, posDistrict, converge

def generateMap(adjList,districtNum,population,position,popWeights,popBound,iterations=10000):

    # input check
    assert (type(adjList) == list) , 'TypeError : adjList'
    assert (type(districtNum) == int) , 'TypeError : districtNum'
    assert (type(population) == dict) , 'TypeError : population'
    assert (type(position) == dict) , 'TypeError : position'
    assert (type(popWeights) == list) , 'TypeError : popWeights'
    assert (type(popBound) == float) , 'TypeError : popBound'
    assert (type(iterations) == int) , 'TypeError : iterations'
    assert (len(popWeights) == districtNum) , 'ValueError : population weights are not aligned with the number of districts'
    
    position2 = copy.deepcopy(position)
    # generate initiakl graph
    initG, initAssign = generateInit(adjList, position2, districtNum) 

    # reorder district according to the population
    popWeights.sort()

    beforePop = {}
    for key in initAssign:
        dis = initAssign[key]
        beforePop[dis] = beforePop.get(dis,0) + population[key]

    biggestDist = sorted(beforePop.keys(),key = lambda x : beforePop[x])

    weights = {}
    for i in range(len(biggestDist)):
        weights[biggestDist[i]] = popWeights[i]


    totalPop = sum(beforePop.values())
    interval = (totalPop/sum(popWeights)) * popBound # calculate interval

    # start generating maps based the initial map

    # create a new graph that contains the information of the initial graph
    GPop = nx.Graph()
    GPop.add_edges_from(adjList)
    nx.set_node_attributes(GPop,initAssign,name = "assign")
    nx.set_node_attributes(GPop,population,name = "population") # some extra step, can be optimized in the future

    start = time.time() 
    GPop, popDistrict, posDistrict, converge = adjustingProcess(GPop = GPop,
                                                                districtNum = districtNum, 
                                                                position = position,
                                                                population = population,
                                                                weights = weights,
                                                                interval = interval,
                                                                iterations = 10000)
    end = time.time() 
    outputInfo = {}
    outputInfo['init'] = initAssign
    outputInfo['final'] = GPop
    outputInfo['posDistrict'] = posDistrict
    outputInfo['popDistrict'] = popDistrict
    outputInfo['time'] = end - start
    outputInfo['converge'] = converge

    return outputInfo


if __name__ == '__main__' :

    ######## Input model ########
    # total congressional vote
    CONGRESS_VOTE = 8
    # district number
    DISTRICT_NUM = 5

    ############################

    # open wisconsin data
    with open("../../../Gerrymander_Scenario_Generate/unintentional_gerrymandering/updated_nodeWisc.pk", "rb") as f:
        nodes = pk.load(f)

    # open adjacency data
    adjList = []

    with open("../../../Gerrymander_Scenario_Generate/unintentional_gerrymandering/updated_adjWisc.pk", "rb") as f:
        adjList = pk.load(f)

    # open node position
    position = {}

    with open("../../../Gerrymander_Scenario_Generate/unintentional_gerrymandering/updated_coordWisc.pk", "rb") as f:
        position = pk.load(f)

    population = {}

    with open("../../../Gerrymander_Scenario_Generate/unintentional_gerrymandering/updated_popWisc.pk", "rb") as f:
        population = pk.load(f)

    totalPop = 0
    unitNumber = {} #used to count the units in it
    for n in nodes:
        unitNumber[n] = 1
        totalPop += population[n]

    result = generateMap(adjList = adjList,
                        districtNum = DISTRICT_NUM,
                        population = population,
                        position = position,
                        popWeights = [2,2,2,1,1],
                        popBound = 0.05,
                        iterations=10000)

    with open("../../tempRepo/test.pk","wb") as f:
        pk.dump(result,f)

    # # initial merge
    # position2 = copy.deepcopy(position)
    # G, initial_assign = mergeProcess(adjList, position2,DISTRICT_NUM)

    # #print(initial_assign)

    # #with open('experiment_result.pk','wb') as f:
    # #    pk.dump(initial_assign,f)

    # # with open('experiment_result.pk','rb') as f:
    # #    initial_assign = pk.load(f)

    # # with open('experiment20221004/imbalance_result5_d6.pk','rb') as f:
    # #     result = pk.load(f)
    # # initial_assign = result['init']
    # # with open('experiment20221004/imbalance_result3_d8_cc.pk','rb') as f:
    # #     result = pk.load(f)
    # # initial_assign = result['init']

    # # Align weighted graph

    # POP_WEIGHTS.sort()
    # beforePop = {}
    # for key in initial_assign:
    #     dis = initial_assign[key]
    #     beforePop[dis] = beforePop.get(dis,0) + population[key]
    # print(beforePop)

    # biggestDist = sorted(beforePop.keys(),key = lambda x : beforePop[x])

    # weights = {}
    # for i in range(len(biggestDist)):
    #     weights[biggestDist[i]] = POP_WEIGHTS[i]

    # interval = (totalPop/sum(POP_WEIGHTS)) * popBound

    # print('bound : {}'.format(interval))
    # #adjusting process
    # GPop = nx.Graph()
    # GPop.add_edges_from(adjList)
    # nx.set_node_attributes(GPop,initial_assign,name = "assign")
    # nx.set_node_attributes(GPop,population,name = "population")
    # start = time.time()
    # GPop, popResult, posDistrict, converge = adjustingProcess(GPop, weights, interval = interval, iterations = 10000, districtNum = DISTRICT_NUM, poistion = position, population = population)
    # end = time.time()
    # print(popResult)



    # extraInfo = {}
    # extraInfo['init'] = initial_assign
    # extraInfo['final'] = GPop
    # extraInfo['adjList'] = adjList
    # extraInfo['population'] = population
    # extraInfo['position'] = position
    # extraInfo['posDistrict'] = posDistrict
    # extraInfo['time'] = end - start
    # extraInfo['coverge'] = converge

    # # pathDir = 'WisconsinExperimentsLarge2/{}districts/'.format(DISTRICT_NUM)
    # # newFile = 'WisconsinExperimentsLarge2/{}districts/{}.pk'.format(DISTRICT_NUM,len(os.listdir(pathDir))+1)

    # with open(newFile,'wb') as f:
    #     pk.dump(extraInfo,f)
        
        

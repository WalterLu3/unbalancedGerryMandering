'''
Methods related to getting map value including
1. population value
2. compactness scores
'''

import pickle as pk
import random
import geopandas as gpd
from  ..file.fileIO import *

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

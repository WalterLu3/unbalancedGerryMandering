'''
Methods related to map drawing.
'''

import pickle as pk
import random
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from  ..file.fileIO import *

def rgb_to_hex(rgb):
    """
    Convert rgb in decimal into hex form

    Parameter
    ---------
    rgb : tuple,
        tuple that contains rgb color code (ex. (120,255,1))

    Return
    ------
    A string that converts rgb into hex form.
    """

    return '#%02x%02x%02x' % rgb


def colorChoice(districtSet):
    """
    Choose a random color for each district. (The district should be named in numerical order (ex. 1-8)

    Parameter
    ---------
    districtSet : list,
        list of district name. (The district should be named in numerical order (ex. 1-8)

    Return
    ------
    customColor : list,
        list of colors in the form hex color code. The length of customColor is the same as the length of districtSet.
    """

    customColor = []
    district_idx = {}
    count = 1
    for i in districtSet:
        # randomly generate color
        col = (random.randrange(1,255),random.randrange(1,255),random.randrange(1,255))

        # convert rgb code to hexadecimal 
        customColor.append(rgb_to_hex(col))
        count += 1
    return customColor

def saveGraph(inFile,colors,shapeDF,pathFile):
    """
    Save a graph given a file, colors, shape dataframe, and the file path to be stored.

    Parameters
    ----------
    inFile : dictionary,
        a dictonary where the 'final' dictionary contains the information for assignment.

    colors : list,
        list of colors in the form hex color code.
    
    shapeDF : ....,
        a dataframe that contains the information of the plotting (coordinate/multi-polygon)

    pathFile : string,
        the path for the file to be stored

    Return
    ------
        a success message.
    """
    G = inFile['final']

    # split some already merged units and recover the district assigment on the smallest level
    district = {}
    for n in G.nodes:
        for u in n.split("&"):
            district[u] = G.nodes[n]['assign']

    reader = district

    # create a column in the shapeDF for storeing district information
    shapeDF['d'] = 0 
    for k in reader:
        units = k.split("&")
        for u in units:
            shapeDF.loc[int(u[1:])-1,'color'] =  colors[reader[k]-1]
            shapeDF.loc[int(u[1:])-1,'d'] =  reader[k]


    # plot the figure
    fig = shapeDF.plot(figsize=(30,30),color=shapeDF['color'],legend=True).get_figure()

    patches = []
    for c in range(len(colors)):
        patches.append(mpatches.Patch(color=colors[c], label='d{}'.format(c+1)))
    districtNum = len(set(district.values()))
    plt.legend(handles=patches[:districtNum],fontsize='xx-large')

    # save the figure to the figure file
    fig.savefig(pathFile)

    return 'succeed.'

def saveSingle(inputPath,shapePath,outputPath):
    """
    Draw a graph given assignment information.

    Parameters
    ----------
    inputPath : string,
        path where assignment data is located

    shapePath : string,
        path where the shape file is located

    outputParh : string,
        path where the output graph is set to be stored.

    Return
    ------
    a success message.

    """

    # read in the assignment file
    inFile = loadPK(inputPath)
    districtSet = sorted(list(inFile['posDistrict'].keys()))
    
    colors = colorChoice(inFile)

    # read in the shapeFile

    shapeDF = gpd.read_file(shapePath)
    shapeDF['color'] = 0

    # create graph
    saveGraph(inFile = inFile,colors = colors, shapeDF = shapeDF, pathFile = outputPath)
    return 'succeed.'

def saveMultiple(inputPathList,shapePath,outputPathList):
    """
    Draw multiple graphs given assignment information in the same color.

    Parameters
    ----------
    inputPathList : list,
        paths where assignment data is located

    shapePath : string,
        path where the shape file is located

    outputPathList : list,
        paths where the output graph are set to be stored.

    Return
    ------
    a success message.

    """

    # create graph
    for i in range(len(inputPathList)):
        # read in the assignment file
        inFile = loadPK(inputPathList[i])
        districtSet = sorted(list(inFile['posDistrict'].keys()))
        
        if i == 0:
            colors = colorChoice(inFile)
            # read in the shapeFile
            shapeDF = gpd.read_file(shapePath)
            shapeDF['color'] = 0

        saveGraph(inFile = inFile,colors = colors, shapeDF = shapeDF, pathFile = outputPathList[i])
    return 'succeed.'







    









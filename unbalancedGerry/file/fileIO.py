import pickle as pk

def loadPK(fileName):
    """
    Load a pickle file.

    Parameter
    ---------
    fileName : string,
        the path of a pickle file

    Return
    ------
    data : type depends on the file type stored in the pk fileName specified in file parameter
    """
    with open(fileName, 'rb') as f:
        data = pk.load(f)

    return data

def writePK(data,fileName):
    """
    Write a pickle file.

    Parameter
    ---------
    data : allows all type,
        data user wants to store as an external file

    fileName : string,
        the path of the file

    Return
    ------
    a string saying success.

    """
    with open(fileName, 'wb') as f:
        pk.dump(data,f)

    return 'Succeed.'



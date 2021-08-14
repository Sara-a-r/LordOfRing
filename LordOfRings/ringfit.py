"""
This module fit multiple circle in sparse matrix
"""

import numpy as np


def get_coord(datafile):
    """
    get_coord gives the X, Y coordinates of ones from a txt file containing a 
    sparse matrix of 0 and 1. 
    The coordinates start from 1 instead of 0.

    Parameters
    ----------
    datafile : .txt file [str]
        The name of the datafile contained in the folder ./data/. You don't
        need to pass './data/'.

    Returns
    -------
    ( 1d numpy-array, 1d numpy-array ) [float, float]
        The coordinates in a tuple  (X, Y).

    """
    circle = np.loadtxt('data/'+datafile)
    coord = np.argwhere(circle!=0)
    Xcoord = coord[:, 1] + 1
    Ycoord = coord[:, 0] + 1
    return Xcoord, Ycoord


def init_triplets(list_events):
    """
    init_triplets gives three array: the first contains the index of border hits
    in the sparse matrix of each event, the second contains the x coordinates of
    hits for all events in a single array, the third contains the y coordinates
    as before.

    Parameters
    ----------
    list_events : list of str
        List of .txt files that contain the sparse matrix of each events. These
        files are contained in the folder ./data/.

    Returns
    -------
    ( 1d numpy-array, 1d numpy-array, 1d numpy-array ) [float, float, float]
        The triplet and the coordinates in a tuple  (triplet, X, Y).

    """
    nevents = len(list_events)  # number of events
    maxhits = 84 # maximum num of hits for event

    # Triplet initialized as empty matrix
    #            (event1)
    #           | XMAX1[3] , XMIN1[3], YMAX1[3], YMIN1[3] |
    # Triplet = |(event2)                                 |
    #           | XMAX2[3] , XMIN2[3], YMAX2[3], YMIN2[3] |
    # Later we will flat this array in 1D
    triplet = np.zeros((nevents, 3*4))

    # Data collected in two matrix (same struct of triplet but different
    # number of columns)
    X = np.zeros((nevents, maxhits))
    Y = np.zeros((nevents, maxhits))

    # Fill one event at time in the triplet matrix and X, Y matrix
    for i, circle in enumerate(list_events):
        xi, yi = get_coord(circle) # Get the sorted coord of event

        X[i, :len(xi)] = xi
        Y[i, :len(yi)] = yi

        idx_sx = np.argsort(xi)
        idx_sy = np.argsort(yi)
        [xi[idx_sx[0]], xi[idx_sx[0]], xi[idx_sx[0]]]

        # Fill the triplet with maximum and minumum relative to the event
        triplet[i,  :3]  = idx_sx[-3:] #XMAX
        triplet[i, 3:6]  = idx_sx[ :3] #XMIN
        triplet[i, 6:9]  = idx_sy[-3:] #YMAX
        triplet[i, 9: ]  = idx_sy[ :3] #YMIN

    triplet = triplet.flatten()
    X = X.flatten()
    Y = Y.flatten()
    return triplet, X, Y

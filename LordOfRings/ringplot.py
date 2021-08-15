"""
This module manage the creation of plot .
"""

import numpy as np
import matplotlib.pyplot as plt
import LordOfRings.ringfit as rf

maxhits = 64 # closer multiple of two to 84

def data_show(filename):
    """data_show show the sparse matrix contained in the file txt.

    Parameters
    ----------
    filename : str
        The name of the txt file (including the extension).

    Returns
    -------

    """
    data = np.loadtxt('data/'+filename)
    plt.imshow(data.T, origin='lower')
    plt.show()


def triplet_plot(idx_event, dict_events):
    """
    triplet_plot shows in a heat map the rings in a event and in a different 
    color the triplets, i.e the border points.
    
    Parameters
    ----------
    dict_events : dictionary
        Dictionary whose keys are the name of .txt files and whose values are  
        the x, y coordinates of the relative event in a list.
        This format is the output of the function LordOfRing.ringfit.load_data.

    idx_event : int
        The index of the event in the list that we want to plot.

    Returns
    -------

    """
    nevents = len(dict_events)
    triplet, X, Y = rf.init_triplets(dict_events)

    # reshape arrays for keep code readable
    Xmat = X.reshape((nevents,maxhits))
    Ymat = Y.reshape((nevents,maxhits))
    tripletmat = triplet.reshape((nevents,12))

    # extract the index of the coordinates of the event idx_event
    coloridx = tripletmat[idx_event].astype(int)

    nonzeromask = Xmat[idx_event, :]!=0 # remove zeros from X, Y

    # extract coordinates from X, Y
    Xvec1 = Xmat[idx_event, :][nonzeromask].astype(int)
    Yvec1 = Ymat[idx_event, :][nonzeromask].astype(int)

    # Go back to coordinates from 0 to max-1
    Xvec1 = Xvec1 - 1
    Yvec1 = Yvec1 - 1

    # Stack coordinates in array of two cols
    coord = np.stack((Xvec1, Yvec1), axis=1)

    # Load the matrix
    circle = np.loadtxt('data/'+list(dict_events.keys())[idx_event])

    # initialize color to 2 (the plot is an heat map)
    c = 2
    for n, i in enumerate(coloridx):
        # if n is multiple of three change color
        if n%3 == 0:
            c+=1
        # change the values of circle (remember to revert coordinates in matrix)
        circle[coord[i, 1], coord[i, 0]] = c

    # show result
    plt.imshow(circle.T, origin='lower')
    plt.show()

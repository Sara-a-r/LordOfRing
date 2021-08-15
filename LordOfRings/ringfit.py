"""
This module fit multiple circle in sparse matrix
"""

import numpy as np

maxhits = 64 # closer multiple of two to 84

def load_data(list_events):
    """
    Parameters
    ----------
    list_events : list of str
        List of .txt files that contain the sparse matrix of each events. These
        files are contained in the folder ./data/.
    
    Returns
    -------
    dictionary
        Dictionary with keys the name of .txt files and with values given by the
         x/y coordinates of the relative event.

    """
    dict_data = {}
    for circle_name in list_events:
        circle = np.loadtxt('data/'+circle_name)
        coord = np.argwhere(circle!=0)
        X = coord[:, 1] + 1
        Y = coord[:, 0] + 1
        dict_data[circle_name] = [X, Y]
    return dict_data


def init_triplets(dict_events, t=1):
    """
    init_triplets gives three array: the first contains the index of border hits
    in the sparse matrix of each event, the second contains the x coordinates of
    hits for all events in a single array, the third contains the y coordinates
    as before.

    Parameters
    ----------
    dict_events : dictionary
        Dictionary whose keys are the name of .txt files and whose values are  
        the x, y coordinates of the relative event in a list.
         This format is the output of the function LordOfRing.ringfit.load_data.

    t : float
        Treshold value for the selection of the triplets, it defines the minimum 
        reciprocal distance of the three points in the same triplet.

    Returns
    -------
    ( 1d numpy-array, 1d numpy-array, 1d numpy-array ) [float, float, float]
        The triplet and the coordinates in a tuple  (triplet, X, Y).

    """
    nevents = len(dict_events)  # number of events

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

    def d3(x, y, z, t):
        # this function return boolean array of len = 3:
        # - True if the d[i, j] > t
        # - False otherwise. 
        dxy = np.linalg.norm(x-y)
        dyz = np.linalg.norm(y-z)
        dxz = np.linalg.norm(x-z)
        return np.array([(dxy>t), (dyz>t), (dxz>t)])

    # Fill one event at time in the triplet matrix and X, Y matrix
    for i, (xi, yi) in enumerate(dict_events.values()):
        #xi, yi = rf.get_coord(circle) # Get the sorted coord of event

        X[i, :len(xi)] = xi
        Y[i, :len(yi)] = yi

        # index sort based on x and y severaly (individualmente)
        idx_sx = np.argsort(xi)
        idx_sy = np.argsort(yi)

        # coordinates sorted based on x and y severaly (individualmente)
        xsortv = np.stack(( xi[idx_sx], yi[idx_sx]), axis=1)
        ysortv = np.stack(( xi[idx_sy], yi[idx_sy]), axis=1)

        # boolean check to stop the while loop
        boolcheck = True
        while boolcheck:
            # boolean array of len equal to len(idx)
            boolarrayx = np.ones(len(idx_sx)).astype(bool)
            boolarrayy = np.ones(len(idx_sy)).astype(bool)

            # Evaluate the reciprocal distance of the points in the same triplet
            bminx = d3(*xsortv[:3], t)
            bmaxx = d3(*xsortv[-3:], t)
            bminy = d3(*ysortv[:3], t)
            bmaxy = d3(*ysortv[-3:], t)
            
            # if all the values in all the 4 bool array are true
            if bminx.all() and bmaxx.all() and bminy.all() and bmaxy.all():
                # we've the right triplets and we exit
                boolcheck = False
            else:
                # sobstitute in boolarray the first/last three bool elements
                boolarrayx[:3] = bminx
                boolarrayx[-3:] = bmaxx
                
                # delete (with the mask boolarrayx) the elements that not 
                # satisfy the contition
                idx_sx = idx_sx[boolarrayx]
                xsortv = xsortv[boolarrayx]

                # same for the y index ordered array
                boolarrayy[:3] = bminy
                boolarrayy[-3:] = bmaxy
                idx_sy = idx_sy[boolarrayy]
                ysortv = ysortv[boolarrayy]
        
        # Fill the triplet with maximum and minumum relative to the event
        triplet[i,  :3]  = idx_sx[-3:] #XMAX
        triplet[i, 3:6]  = idx_sx[ :3] #XMIN
        triplet[i, 6:9]  = idx_sy[-3:] #YMAX
        triplet[i, 9: ]  = idx_sy[ :3] #YMIN

    triplet = triplet.flatten()
    X = X.flatten()
    Y = Y.flatten()
    return triplet, X, Y

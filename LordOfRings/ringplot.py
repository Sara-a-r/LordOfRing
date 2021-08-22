"""
This module manage the creation of plot.
"""

import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
import LordOfRings.ringfit as rf


def data_show(filename):
    """data_show show the sparse matrix contained in the file txt.

    Parameters
    ----------
    filename : str
        The name of the txt file (including the extension).

    Returns
    -------

    """
    fig, ax = plt.subplots()
    data = np.loadtxt('data/'+filename)
    ax.imshow(data.T, origin='lower')
    return ax


def ptolemy_contourf(idx_event, dict_events, maxhits = 64, t=1, annotate = False, thr = 0.2):
    """
    ptolemy_contourf makes 4 contourf plot (one for each triplet) having as z 
    the value p associated to the Ptolemy theorem : 
    p = AB * BC + AD * CD - AC * BD. Triplets values are choosen using the
    function 'LordOfRings.ringfit.init_triplet'.
    The function plots also the hits of the sparse matrix highlighting the 
    triplet's points.
    
    Parameters
    ----------
    idx_event : int
        The index of the event in the list that we want to plot.

    dict_events : dictionary
        Dictionary whose keys are the name of .txt files and whose values are  
        the x, y coordinates of the relative event in a list.
        This format is the output of the function LordOfRing.ringfit.load_data.

    maxhits : int
        The maximum number of points after the second pruning. If a circle has 
        more than maxhits points after the first pruning remove them randomly
        until they reach maxhits.

    t : float
        Treshold value for the selection of the triplets, it defines the minimum 
        reciprocal distance of the three points in the same triplet.
    
    annotate : boolean
        If True write on each point of triplets the letter that correspond to 
        the sequential order of the 3 points forming the triplet itself.

    thr : float
        Threshold on the p value: if a hit has p < thr it is highlighted in the 
        plot.

    Returns
    -------

    """
    nevents = len(dict_events)
    triplet, X, Y = rf.init_triplets(dict_events, t = t)
    fig, axs = plt.subplots(1,4,figsize = (19, 5))

    for idx_trip, ax in zip(range(4), axs.flat):

        def pypto(x, y, trip):
            def d(xa, ya, xb, yb):
                dist = np.sqrt( ( xa - xb )**2 + ( ya - yb )**2)
                return dist
            A = [trip[0], trip[1]]
            B = [trip[2], trip[3]]
            C = [trip[4], trip[5]]
            D = [x, y]
            return d(*A, *B) * d(*C, *D) + d(*A, *D) * d(*C, *B) - d(*A, *C) * d(*B, *D)
       
        xx = np.array([X[i] for i in triplet])
        yy = np.array([Y[i] for i in triplet])
        real_trip = []
        idx_tripl1 = 4*3*idx_event + idx_trip*3
        idx_tripl2 = 4*3*idx_event + (idx_trip+1)*3
     
        for x, y in zip(xx[idx_tripl1:idx_tripl2], yy[idx_tripl1:idx_tripl2]):
            real_trip.append(x)
            real_trip.append(y)
     
     
        real_trip = np.array(real_trip)
     
        ypto = []
        for i, (x, y) in enumerate(zip(X, Y)):
            pto = pypto(x, y, real_trip)
            ypto.append(pto)
        
        ypto = np.array(ypto)[maxhits*idx_event : maxhits*(idx_event+1) ]
        dd = np.array(dd)[maxhits*idx_event : maxhits*(idx_event+1) ]
     
        Xinterval = X[maxhits*idx_event : maxhits*(idx_event+1) ]
          
        Xmatca = X[maxhits*idx_event : maxhits*(idx_event+1)][ypto<thr] 
        Ymatca = Y[maxhits*idx_event : maxhits*(idx_event+1)][ypto<thr] 
     
        # reshape arrays for keep code readable
        tripletmat = triplet.reshape((nevents,12))
     
        Xmat = X.reshape((nevents, maxhits))
        Ymat = Y.reshape((nevents, maxhits))
     
     
        # extract the index of the coordinates of the event idx_event
        coloridx = tripletmat[idx_event].astype(int) - idx_event*maxhits
     
        coloridx = coloridx[3*idx_trip:3*(idx_trip+1)]
     
     
        nonzeromask = Xmat[idx_event, :]!=0 # remove zeros from X, Y
        nonzeromaskca = Xmatca!=0 # remove zeros from X, Y
     
     
        # extract coordinates from X, Y
        Xvec1 = Xmat[idx_event, :][nonzeromask].astype(int)
        Yvec1 = Ymat[idx_event, :][nonzeromask].astype(int)
     
        Xvec1ca = Xmatca[nonzeromaskca].astype(int)
        Yvec1ca = Ymatca[nonzeromaskca].astype(int)
     
        # Go back to coordinates from 0 to max-1
        Xvec1 = Xvec1 - 1
        Yvec1 = Yvec1 - 1
     
        Xvec1ca = Xvec1ca - 1
        Yvec1ca = Yvec1ca - 1    
     
        # Stack coordinates in array of two cols
        coord = np.stack((Xvec1, Yvec1), axis=1)
        coordca = np.stack((Xvec1ca, Yvec1ca), axis=1)
     
     
        # Load the matrix
        circle = np.loadtxt('data/'+list(dict_events.keys())[idx_event])
     
        letters = ['A', 'B', 'C']
        # initialize color to 2 (the plot is an heat map)
        for x, y in coordca:
            circle[x, y] = 2
        
        c = 2
        for n, (i, l) in enumerate(zip(coloridx, letters)):
            # if n is multiple of three change color
            if n%3 == 0:
                c+=1
            # change the values of circle
            circle[coord[i, 0], coord[i, 1]] = c
            if annotate:
                ax.annotate(l, (coord[i, 0] + 0.5, coord[i, 1] + 0.5), color='yellow', fontsize=18)
        j = idx_trip + 4*idx_event
        # show result
        ar = np.arange(128)
        xmesh, ymesh = np.meshgrid(ar, ar)
        P, _ = pypto(xmesh, ymesh, real_trip)
        P = ma.masked_where(P <= 0, P)
        level = np.logspace(-7, 4, 12)
        cont = ax.contourf(ar, ar, P, level, locator=plt.ticker.LogLocator(), 
                           cmap=plt.get_cmap('Blues'), alpha=0.8, antialiased=True, 
                           zorder=1)

        alphafilter = np.copy(circle)
        am = ma.masked_where(alphafilter==0, alphafilter)

        ax.imshow(am.T, origin='lower', zorder=2, cmap=plt.cm.get_cmap('Reds_r') , interpolation='none')
        ax.set_title(f'Catch {np.sum(ypto<dd)} points')
    plt.tight_layout()
    plt.suptitle(f'evt {idx_event}')
    plt.show()
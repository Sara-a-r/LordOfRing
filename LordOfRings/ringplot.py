"""
This module manage the creation of plot.
"""

import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
from matplotlib import ticker
import LordOfRings.ringfit as rf
import math


def data_show(filename, ax = None):
    """
    data_show show the sparse matrix contained in the file txt.

    Parameters
    ----------
    filename : str
        The name of the txt file (including the extension).
        
    ax : matplotlib axes (default None)
        The axes in wich plot the sparse matrix, if None define a new ax.
    
    Returns
    -------

    """
    if ax == None:
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
    font = {'size'   : 12} # stack
    plt.rc('font', **font) # overflow
    triplet, X, Y = rf.init_triplets(dict_events, t = t)
    fig, axs = plt.subplots(1,4,figsize = (21, 5), sharex = True)

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
        P = pypto(xmesh, ymesh, real_trip)
        P = ma.masked_where(P <= 0, P)
        level = np.logspace(-7, 4, 12)
        cont = ax.contourf(ar, ar, P, level, locator=ticker.LogLocator(), 
                           cmap=plt.get_cmap('Blues'), alpha=0.8, antialiased=True, 
                           zorder=1)

        alphafilter = np.copy(circle)
        am = ma.masked_where(alphafilter==0, alphafilter)

        ax.imshow(am.T, origin='lower', zorder=2, cmap=plt.cm.get_cmap('Reds_r') , interpolation='none')
        ax.set_title(f'Catch {np.sum(ypto<thr)} points')
        
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.91, 0.2, 0.01, 0.6])
    fig.colorbar(cont, cax=cbar_ax)
    plt.suptitle(f'evt {idx_event}')
    plt.show()
    
    

def show_circle_fit(list_events, rr, xc, yc, ncircline):
    """
    show_circle_fit makes a plot of sparse matrices contained in list_events. In
    each plot is also shown the fit results corresponding to that event.
    The subplots layout is optimized by the function.

    Parameters
    ----------
    list_events : list of str
        List of .txt files that contain the sparse matrix of each events. These
        files are contained in the folder ./data/.

    rr : 2d numpy-array [float]
        The radii predicted by the multiring alghoritm in the format described
        in the return of the function 'LordOfRings.ringfit.multi_ring_fit'.

    xc : 2d numpy-array [float]
        The x coordinates of the center predicted by the multiring alghoritm in 
        the format described in the return of the function 
        'LordOfRings.ringfit.multi_ring_fit'.

    yc : 2d numpy-array [float]
        The y coordinates of the center predicted by the multiring alghoritm in 
        the format described in the return of the function 
        'LordOfRings.ringfit.multi_ring_fit'.

    ncircleline : int
        It correspondes to the number of sparse matrix in each row. This number 
        has to be lower or equal to the number of events.

    Returns
    -------

    """
    # If there are more event than ncircle
    if len(list_events)>ncircline:
        # if I need to plot 50 event in columns of 5 event I need 10 rows: 50/5,
        # if the division doesn't give an integer we need to round up (ceil) it, at the last iteration
        # we break the line of plot with empty plot.
        for ii, s in enumerate(range(math.ceil(len(list_events)/ncircline))): # division with ceil for divide rows
            k = ncircline*ii # parameter that jump of ncircle 
            fig, axs = plt.subplots(1, ncircline, figsize = (23, 8))
            for i, ax in zip(range(ncircline), axs.flat): # start plot the circle 
                if k + i >= len(list_events): # if we are in the last (incomplete) line
                    fig.delaxes(ax) # remove the ax
                else:
                    data_show(list_events[k + i], ax = ax) # plot circle points from sparse matrix
                    for j in range(4): # Plot the four fit curve from the Taubin algoritm
                        ax.add_artist( plt.Circle( ( xc[k + i, j]-1, yc[k + i, j]-1 ), rr[k + i, j], fill=False, color = 'y', alpha=0.5) )
                    ax.set_title(f'{list_events[k+i]}') # setting title
            plt.show()
    else: # else plot all in one line
        fig, axs = plt.subplots(1, len(list_events), figsize = (int(20*len(list_events)/ncircline), 5))
        for i, ax in zip(range(len(list_events)), axs.flat):
            data_show(list_events[i], ax = ax)
            for j in range(len(rr)):
                ax.add_artist( plt.Circle( ( xc[i, j]-1, yc[i, j]-1 ), rr[i, j], fill=False, color = 'y', alpha=0.5) )
            ax.set_title(f'{list_events[i]}')
        plt.show()
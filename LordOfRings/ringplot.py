"""
This module manage the creation of plot.
"""

import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.patches as mpatches
import LordOfRings.ringfit as rf
import LordOfRings.core as core
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
    matplotlib axes
        The axes in wich plot the sparse matrix, if None returne a new ax.
    
    """
    if ax == None:
        fig, ax = plt.subplots()
    data = np.loadtxt('data/'+filename)
    circ = ax.imshow(data.T, origin='lower', cmap = 'Blues_r')
    return ax


def ptolemy_contourf(idx_event, dict_events, maxhits = 64, t=1, annotate = False, thr = 0.2, GPU = False):
    """
    ptolemy_contourf makes 4 contourf plot (one for each triplet) having as z 
    the value p associated to the Ptolemy theorem : 
    p = AB * BC + AD * CD - AC * BD. Triplets values are choosen using the
    function 'LordOfRings.core.init_triplet'.
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
    
    GPU : boolean
        If you want to extract triplet from GPU set this value on True, 
        else choose False.

    Returns
    -------

    """
    
    font = {'size'   : 12} # resize all the fonts
    plt.rc('font', **font) #
    fig, axs = plt.subplots(1,4,figsize = (21, 5), sharex = True)

    nevents = len(dict_events) # the len of the dictionary is equal to the number of events
    # initialize the triplets and the joined array for all the events in dict_events
    triplet, X, Y = core.init_triplets(dict_events, t = t, maxhits=maxhits, GPU = GPU)
    
    listval = [] # this is used for generate the legend of an imshow (plt.legend wouldn't work)
    for idx_trip, ax in zip(range(4), axs): # we make a plot for every triplet
        Xca, Yca, my_triplet, trip_coord = core.ptolemy_candidates(triplet, X, Y, idx_event, idx_trip, thr, nevents, maxhits)
        
        Xinterval = X[maxhits*idx_event : maxhits*(idx_event+1) ]
        Yinterval = Y[maxhits*idx_event : maxhits*(idx_event+1) ] 
        
        nonzeromask = Xinterval!=0 # remove zeros from X, Y
        nonzeromaskca = Xca!=0 # remove zeros from Xca, Yca
        # apply the masks
        Xinterval = Xinterval[nonzeromask].astype(int)
        Yinterval = Yinterval[nonzeromask].astype(int)
        Xca = Xca[nonzeromaskca].astype(int)
        Yca = Yca[nonzeromaskca].astype(int)
        
        # The coordinates start from 1 for ours choise: we need to make them come back (-1) in order to 
        # compare to the point of the original sparse matrix.
        Xinterval -= 1
        Yinterval -= 1
        Xca -= 1
        Yca -= 1    
     
        # Stack coordinates in array of two cols
        coord = np.stack((Xinterval, Yinterval), axis=1)
        coordca = np.stack((Xca, Yca), axis=1)

        # Load the matrix
        circle = np.loadtxt('data/'+list(dict_events.keys())[idx_event])
     
        # one letter for each point of the triplet (usefull for checking the cyclicity)
        letters = ['A', 'B', 'C']
        # we put to 2 the point that are recognized by ptolemy in the matrix
        for x, y in coordca:
            circle[x, y] = 2
            
        # we put to 3 the point that belong to the triplet
        for (i, l) in zip(my_triplet, letters):
            circle[coord[i, 0], coord[i, 1]] = 3
            if annotate: # we add lecters over the triplet points
                ax.annotate(l, (coord[i, 0] + 0.5, coord[i, 1] + 0.5), color='red', fontsize=18)

        # plot the conturf
        ar = np.arange(len(circle)) # inizialize a vector of lenght equal to the lateral size of the circle matrix
        xmesh, ymesh = np.meshgrid(ar, ar) # create a grid with that vector describing all the possible coordinate 
        P = core.pypto(xmesh, ymesh, trip_coord) # evaluate the ptolemy value for all the points in the 'Ptolemy matrix'
        P = ma.masked_where(P <= 0, P) # mask the ptolemy matrix where is lower than zero (for create a logaritmic conturf color scale)
        level = np.logspace(-7, 4, 12) # scale level for logaritmic heatmap 
        cont = ax.contourf(ar, ar, P, level, # level of colormap 
                           locator=ticker.LogLocator(), # for logaritmic colormap 
                           cmap=plt.get_cmap('Blues'), alpha=0.8, antialiased=True, 
                           zorder=1) # zorder=1 mean that this plot is under the plots with zorder = 2

        # copy of original matrix for avoid unwanted modificaitons
        alphafilter = np.copy(circle)
        # create a mask where alphafilter is equal to 0 (the value masked will bee consider with transparent color by imshow!)
        am = ma.masked_where(alphafilter==0, alphafilter)
        # plot the masked array 
        circ_plot = ax.imshow(am.T, origin='lower', zorder=2, cmap=plt.cm.get_cmap('Reds_r') , interpolation='none')
        ax.set_title(f'Catch {len(Xca[Xca!=0])} points')
        circ_plot.set_clim(1, 4)
        # get the colors of the values, according to the 
        # colormap used by imshow
    # STUFF FOR LEGEND IN IMSHOW... # stack overflow....
        values = np.unique(circle.ravel())[1:]
        for value in values:
            listval.append(value) 
    listval = np.unique(np.array(listval))
    colors = [ circ_plot.cmap(circ_plot.norm(value)) for value in listval]
    
    # create a patch (proxy artist) for every color 
    textlabel = {'1':'Not catched by Ptolemy', '2':'Catched by Ptolemy', '3':'Triplet point'}
    patches = [ mpatches.Patch(color=colors[i], label="{tt}".format(tt=textlabel[str(num)]) ) for i, num in enumerate(listval.astype(int))]
    # put those patched as legend-handles into the legend
    axs[0].legend(handles=patches, loc='upper left', bbox_to_anchor=(0., 1.2), ncol=len(listval), fontsize=10)
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.91, 0.2, 0.01, 0.6])
    fig.colorbar(cont, cax=cbar_ax)
    # dict_events.key() extract the 'keys' (in our case the name of the files of the events in dict_events),
    # with list() we turn that names into a list, in the end we use the index 'idx_event' in order to extract the event name
    # that we have analized here.
    evt_name = list(dict_events.keys())[idx_event]
    plt.suptitle(f'evt {evt_name}')
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
                        ax.add_artist( plt.Circle( ( xc[k + i, j]-1, yc[k + i, j]-1 ), rr[k + i, j], fill=False, color = 'grey', alpha=0.7) )
                    ax.set_title(f'{list_events[k+i]}') # setting title
            plt.show()
    else: # else plot all in one line
        fig, axs = plt.subplots(1, len(list_events), figsize = (int(20*len(list_events)/ncircline), 5))
        if len(list_events)>1:
            for i, ax in zip(range(len(list_events)), axs):
                data_show(list_events[i], ax = ax)
                for j in range(4):
                    ax.add_artist( plt.Circle( ( xc[i, j]-1, yc[i, j]-1 ), rr[i, j], fill=False, color = 'grey', alpha=0.7) )
                ax.set_title(f'{list_events[i]}')
        else:
            data_show(list_events[0], ax = axs)
            for j in range(4):
                axs.add_artist( plt.Circle( ( xc[0, j]-1, yc[0, j]-1 ), rr[0, j], fill=False, color = 'grey', alpha=0.7) )
            axs.set_title(f'{list_events[0]}')
        plt.show()
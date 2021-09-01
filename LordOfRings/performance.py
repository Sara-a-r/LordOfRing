'''
This module evaluates the performance of the algorithim and print the results
in a fashionable way.
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import LordOfRings.ringfit as rf
import time

def get_true_value(list_event):
    """ get_true_value return radius and center with the index of the 
    membership of that circle to some event.

    Parameters
    ----------
    list_events : list of str
        List of .txt files that contain the sparse matrix of each events. These
        files are contained in the folder ./data/.

    Returns
    -------
    (1d numpy-array, 2d numpy-array)
        The first array contains the events names repeated as many time as the
        number of circle in the relative event.
        The second array contains the center position (xc, yc) and the radius
        of the events whose names is stored in the first output of this 
        function. 
    """
    work_dir = os.getcwd() # get the working directory
    data_dir = os.path.join(work_dir,"data/") # define the data directory
    assert os.path.exists(data_dir) #if the directory does not exist stop code

    event_circle_idx = []
    for j, evt in enumerate(list_event):  
        # Open the file
        file_data = open(data_dir + evt, 'r').readlines()
        #### GET NUMBER OF CIRCLES ########
        line1 = file_data[1] # extract line one (the first line with data)
        # the number of circle is the last number in the line one
        ncircle = int(line1.split()[-1]) # get the last number (is the number of circle)
        # With the number of circle we know how many data we will save for this evt:
        data = np.empty((ncircle, 3)) # a row for each circle and on column:
                                      #  col1   col2   col3
                                      #   xc     yc      r 
        # Loop only on commented line: we have ncircle circle so we have to read 
        # only from 1 to ncircle + 1 (in the first line there are the name xc, etc..)
        raw_data = np.array(file_data)[1: ncircle + 1]
        for i, line in enumerate(raw_data):
            # extract the data in list
            list_data = [int(s) for s in line.split() if s.isdigit()]
            # [:-1] delete the number of circle from list data (now is only xc, yc, r)
            data[i] = np.array(list_data)[:-1] # write this information in data[i] (the empty arary created before)
            # get the event index in the array (in order to save to wich event belong this circle)
            event_circle_idx.append(evt)
        if j==0:# if we are in the first event initialize the array true_values
            true_values = data
        else: # else do a vstack in order to mantain the structure created
            true_values = np.vstack((true_values, data))
    event_circle_idx = np.array(event_circle_idx)
    return event_circle_idx, true_values


def hist_error(list_event, rr, xc, yc, bins = None,  ax = None, label = None, error_threshold = 50, **kwargs):
    """
    hist_error plot the histogram of the percentage error of the fits against
    the true values given in the toy dataset created by LordOfRings.generator.

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
    
    bins : int or None
        Number of bins in the histogram, if None it is computed with the 
        sturges rule.
    
    ax : matplotlib.axes or None
        axes contain the histogram, if None a new ax is created.

    error_threshold : float (lower than 100)
        maximum percentage error for which a it is considered to match a real
        radius.
        
    **kwargs : dictionary
        matplotlib.axes arguments.
      
    Returns
    -------
    (matplotlib.axes, float)
        The axes that contain the histogram of the logarithm of the errors and 
        the number of circle predicted (i.e. "matched" circles) divided by the 
        total number of circle in all the events of list_event. 
    """
    idx, true_val = get_true_value(list_event)
    # empty array of the same lenght of the total number of radius in all the events.
    tot_score = np.empty(len(idx))
    # number of events
    nevents = len(list_event)
    # CREATION OF A SCORE #
    tot_err = []
    tot_circle = 0
    tot_pred = 0
    for i, evt in enumerate(list_event): # 'evt' contain the txt name of the events
        # Extract the true values for each event, we transopose in order to have all the radius in the same row, 
        # all the xc in the same row, all the yc in the same row.
        #
        #  - true_val[idx==evt] structure:
        #    xc    yc    r
        #
        #  - transpose true_val[idx==evt] structure:
        #    xc
        #    yc
        #    r
        # writing the name separated by ',' python separate each row of true_val[idx==evt].T
        xc_true, yc_true, r_true = true_val[idx==evt].T # the traspose is for create
                                                        # an array of radius, of xc ...
        # number of circle in this event ('evt'), is extracted from r_true that contains all the radii for the circles 
        # of this evt
        number_of_circle = len(r_true)
        # this will contain the total number of circle in all the events
        tot_circle += number_of_circle
        # boolean array containing 'number_of_circle' times False, it will be 
        # filled with True at the index arg_min_r (see below) every time that 
        # a circle is 'correctly' predicted.
        boolcheck = np.zeros(number_of_circle).astype(bool)
        for j in range(4): # loop over the four radius of results
            r_pred  = rr[i, j]
            xc_pred = xc[i, j]
            yc_pred = yc[i, j]
            # ERROR FORMULA:
            #         |r_true - r_pred|
            #  err% = ----------------- * 100  
            #               r_true
            if r_pred != 0: # if the fits has predicted something for this j, 
                            # zeros correspond to "missed" prediction by the algorithm.
                # array of errors % (same lenght of r_pred)
                err_array_r = np.abs(r_true - r_pred)/r_true*100
                # the index where the mimimum error % is occured (referred to the array of real radius) 
                arg_min_r = np.argmin(err_array_r)
                # the minimum error % (if error greather than 100 just puts 100 here)
                err_r  = min(100, np.min(np.abs(r_true - r_pred)/r_true*100))

                # same for xc
                arg_min_xc = np.argmin(np.abs(xc_true - xc_pred)/xc_true*100)
                err_xc = min(100, np.min(np.abs(xc_true - xc_pred)/xc_true*100))
                # same for yc
                arg_min_yc = np.argmin(np.abs(yc_true - yc_pred)/yc_true*100)
                err_yc = min(100, np.min(np.abs(yc_true - yc_pred)/yc_true*100))

                # evaluated the max error in the three previus error (r, xc, yc)
                err_max = np.max(np.array([err_r, err_xc, err_yc]))
                # if the max error respects the threshold asked by user
                if err_max < error_threshold:                    
                    tot_err.append(err_max) # append the max error to the list defined at the beginning
                    # if the index of the predicted is the same for all the three radius parameters (r, xc, yc)
                    # (i.e. the mimimum error is referred to the same circle)
                    if (arg_min_r == arg_min_xc) and (arg_min_xc == arg_min_yc):
                        boolcheck[arg_min_r] = True # put a true value in the boolcheck in the position arg_min_r 
                                                    # (the position of the predicted radius)
        # add to tot_pred the number of predicted circles for this event (is the number of ones in boolcheck)
        tot_pred += np.sum(boolcheck)
    # Make the logarithm of the tot_err 'cause is too much pressed into zero.
    tot_err = np.log10(np.array(tot_err))
    if ax == None: # if ax is none then create it.
        fig, ax = plt.subplots()
    if bins == None:
        bins = 1 + int(np.log2(len(tot_err)))
    if label == None:
        label = f'{list_event[0][0]} circ, pred/tot = {tot_pred/tot_circle:.2f}'
    ###### ax belluire #######
    ax.grid(linewidth=0.5)
    ax.set_axisbelow(True)
    ax.hist(tot_err, bins=bins, label=label, **kwargs)
    ax.set_ylabel('count')
    ax.set_xlabel(r'$\log_{10}(err\%)$')
    ##########################
    return ax, tot_pred/tot_circle
    
def speed_test(list_events, maxhits = 64, step = 8, ax = None):
    """
    This function compare the speed of the GPU code against the non-GPU code
    as the number of events varies. The result are written in a matplotlib 
    axes.

    Parameters
    ----------
    list_events : list of str
        List of .txt files that contain the sparse matrix of each events. These
        files are contained in the folder ./data/.
    
    maxhits : int
        Maximum number of points per event (i.e. the maximum number of ones in
        each sparse matrix).
    
    step : int
        Number of events added each iteration.
    
    ax : matplotlib axes
        The final plot will be added in this axes. If None a new ax is created.

    Returns
    -------
    matplotlib axes
        The ax with the plot.

    """
    # Create the axes ------------------------------------
    if ax == None: # Create the ax if the usr pass None
        fig, ax = plt.subplots(figsize=(8,6))
        plt.rcParams.update({'font.size': 15}) # Increase the fontsize
        
    # Create the variables -------------------------------
    gpu_time = [] # Array of the time of fit without gpu
    nogpu_time = [] # array of fit's time with gpu
    dict_event = {} # Initialize empty events dictionary
    sup = int(step * np.floor(len(list_events)/step)) # floor is the 'difect round' in italian
    # Loop over the events with stepsize = step ----------
    for i in np.arange(0, sup, step): # loop that increase the number of processed event
        add_to_dict = rf.load_data(list_events[i:i+step]) # add to dictionary the events from i to i + step
        dict_event.update(add_to_dict) # Update the dictionary
        # GPU fit ________________________________________
        start = time.time() # initialize the time for the GPU code
        rr, xc, yc, evt_search  = rf.multi_ring_fit(dict_event, triplet_threshold = 10, maxhits=maxhits, means=False, GPU = True)
        gpu_time.append(time.time()-start) # Save the GPU timi in the array
        # non GPU fit ____________________________________
        start = time.time() # initialize the time for the non-GPU code
        rr, xc, yc, evt_search  = rf.multi_ring_fit(dict_event, triplet_threshold = 10, maxhits=maxhits, means=False, GPU = False)
        nogpu_time.append(time.time()-start) # save the time for the non-GPU code
        
    # Plot the results -----------------------------------    
    # Plot the time vs the number of events processede at same time
    ax.plot(np.arange(0, sup, step), nogpu_time, label='No GPU')
    ax.plot(np.arange(0, sup, step), gpu_time, label = 'GPU')
    ax.legend()
    # adding auxiliary ticks and grid
    maxtime = round(6/5 * np.max(np.array(nogpu_time)), 1) # round 6/5 of the max time in the non gpu code (at 0.x) (6/5 is arbitrary)
    major_ticks = np.arange(0, maxtime, 0.5) # Set the major ticks with a step of 0.5 
    minor_ticks = np.arange(0, maxtime, 0.1) # set the minor ticks with a step of 0.1
    ax.set_xlabel('Number of events')
    ax.set_ylabel('execution time (s)')
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    return ax


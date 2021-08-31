"""
This module fit multiple circle in sparse matrix
"""

import numpy as np
import os
import traceback # for print the full error in try except
import LordOfRings.core as core
import sys

def load_data(list_events, maxhits = 64):
    """
    Parameters
    ----------
    list_events : list of str
        List of .txt files that contain the sparse matrix of each events. These
        files are contained in the folder ./data/.
    
     maxhits : int
        Maximum number of points per event (i.e. the maximum number of ones in
        each sparse matrix).
    
    Returns
    -------
    dictionary
        Dictionary with keys the name of .txt files and with values given by the
        x,y coordinates of the relative event in a list (i.e. if X, Y are the 
         coordinates arrays so in the dictionary we obtain [X, Y]).

    """
    # Create empty dictionary
    dict_data = {}
    # for each event extract the data in the X, Y array, then put the arrays in the dictionary.
    for circle_name in list_events:
        # initialize empty vector of zeros 
        X = np.zeros(maxhits)
        Y = np.zeros(maxhits)
        # Load the matrix
        circle = np.loadtxt('data/'+circle_name)
        # Find the ones positions in the matrix
        coord = np.argwhere(circle!=0)
        # add + 1 in order to start the coordinates from 1 instead of zero.
        x_nonzero = coord[:, 0] + 1
        y_nonzero = coord[:, 1] + 1
        # fill the first elemento of X, Y (of lenght maxhits) with the nonzero coordinates extracted before
        X[:len(x_nonzero)] = x_nonzero
        Y[:len(y_nonzero)] = y_nonzero
        # Write the coordinates in the dictionary at position described by the string 'circle_name'.
        dict_data[circle_name] = [X, Y]
    return dict_data


def multi_ring_fit(dict_events, maxhits=64, 
                   ptolemy_threshold = 0.2, triplet_threshold = 10,
                   means= False, meanthr = 2, 
                   rsearch = 0 , drsearch = 0, 
                   GPU = True):
    """
    multi_ring_fit executes multiple circle fits on arbitrary number of events
    giving the values predicted by the alghoritm. 

    Parameters
    ----------
    dict_events : dictionary
        Dictionary whose keys are the name of .txt files and whose values are  
        the x, y coordinates of the relative event in a list.
        This format is the output of the function LordOfRing.ringfit.load_data.
        
    maxhits : int
        Maximum number of points per event (i.e. the maximum number of ones in
        each sparse matrix). Is a good practice to keep this value as a multiple
        of 2 if GPU is True (it will define the number of thread per block).
        Default is 64.
    
    triplet_threshold : float 
        Minimum distance between two points of the same triplet. Default is 10.
        
    ptolemy_threshold : float
        Maximum value of the difference defined by the Ptolemy theorem: the points
        for which this difference is greater than ptolemy_threshold are excluded
        by the fit algorithm. Default is 0.2.
    
    means : boolean
        If True find similar circle from each event and mean them. This remove
        duplicate circles. Default is False.
        
    meanthr : float
        Two circle are similar if their radius and center's coordinates 
        difference ( evaluated separately ) are less then meanthr. 
        Default is 2.
    
    rsearch : float
        The radius that will be searched in the data after the fit, if zero 
        the fourth argument returned by the function will be an empty 
        numpy array. If zero no radius will be searched. 
        Default is 0.
    
    drsearch : float
        The error on the radius rsearch, the function will return all the 
        events having a fitted radius in 
        (rsearch - drsearch, rsearch + drsearch). 
        Default is 0.
        
    GPU : boolean
        If True execute the fit on the GPU, if False run a python iterative 
        routine for the fit task. Default is True.

    Returns
    -------
    ( 2d numpy-array, 2d numpy-array, 2d numpy-array, 1d numpy-array ) 
    [float, float, float, string]
        The radius, X center, Ycenter of all the events in three array. These
        values are distributed in different row, one for each event. 
        Last array contain the names of the events with the radius in the 
        range searched with rsearch, drsearch founded by the algoritm.

    """
    # Extract the events name from the dictionary with the data
    list_events = np.array(list(dict_events.keys())) # dict_events.keys() gives the names of the events in a dict format, 
                                                     # with list() we turn that format into list and in the end 
                                                     # we create the numpy array with that list
    
    nevents = len(dict_events) # the number of events is just the lenght of the dictionary
    
    # If the user wants the GPU and the GPU is ok with that
    if core.GPU_enabled() and GPU: ############ GPU code ####################################
        try: # This try is for the file 'file_name': this file is created in the user folder, if some 
             # error occour it has to be removed otherwise the user will find this new file in his folder 
             # without a good reason..
            
            ############## Create the file with the right MAXHITS ############################
            # Modify a constant (MAXHINTS) directly in the .cu file starting from MultiRing.cu
            file_name = f"{maxhits}MultiRing.cu" 
            triplet_file_name = f"{maxhits}triplet.cu"# create file instead of overwrite the original 
            # os.path.dirname() extract the directoty of the argument, in this case the argument is 
            # __file__, this special variable (of python) contains the directory of the module and, 
            # the module name, in this case the name is 'ringfit' and the __file__ variable will be 
            # something like:
            # '/content/drive/MyDrive/Progetto CMEPDA/LordOfRings/LordOfRings/ringfit.py'.
            module_path = os.path.dirname(__file__)
            # CODE FOR FILE_NAME
            if os.path.exists(module_path + '/cuda/' + file_name)==False: # if doesn't exist in the precaricated files
                # With this code we create a new file in the user folder with the new value of maxhits.
                lines = open(module_path+'/cuda/MultiRing.cu', 'r').readlines() # with readlines() we 
                                                                                # write the lines file 
                                                                                # in a list: every line
                                                                                # in an element of the list.
                line_num = 4 # I know that maxhits is at line 4
                lines[line_num] = f'#define MAXHITS (int) {maxhits}\n' # rewrite the line 4 with this new string
                out = open(file_name, 'w') # open the output file (in the user folder)
                out.writelines(lines) # rewrite the full file in this one
                out.close() # close the file
                # The file that will be execute is the new file
                execute_name = file_name
            else: # if the file 'file_name' is found
                # the file that will be executed is the precaricated 
                execute_name = module_path + '/cuda/' + file_name 
            # CODE FOR TRIPLET_FILE_NAME (same of code for file_name)
            if os.path.exists(module_path + '/cuda/' + triplet_file_name)==False: # if doesn't exist in the precaricated files
                # With this code we create a new file in the user folder with the new value of maxhits.
                lines = open(module_path+'/cuda/triplet.cu', 'r').readlines() # with readlines() we 
                                                                                # write the lines file 
                                                                                # in a list: every line
                                                                                # in an element of the list.
                line_num = 4 # I know that maxhits is at line 4
                lines[line_num] = f'#define MAXHITS (int) {maxhits}\n' # rewrite the line 4 with this new string
                out = open(triplet_file_name, 'w') # open the output file (in the user folder)
                out.writelines(lines) # rewrite the full file in this one
                out.close() # close the file
                # The file that will be execute is the new file
                triplet_execute_name = triplet_file_name
            else: # if the file 'file_name' is found
                # the file that will be executed is the precaricated 
                triplet_execute_name = module_path + '/cuda/' + triplet_file_name 

# TO DELETE # Why precaricate files? to speed up the code (generate cuda file is slow)
            # call the function that manage cuda ################# FIT RUTINE #######################
            rr, xc, yc = core.CUDA_fit(execute_name, triplet_execute_name, dict_events, 
                                       maxhits, triplet_threshold, ptolemy_threshold, rsearch, drsearch)
            #########################################################################################
            if os.path.exists(file_name): # If the file was created in the user folder
                os.remove(file_name)      # remove the file once the fit is over
            if os.path.exists(triplet_file_name):
                os.remove(triplet_file_name)
        except:
            traceback.print_exc() # print the entire error occurred
            if os.path.exists(file_name): # If the file still exist
                os.remove(file_name) # remove the file even if some error occours
            if os.path.exists(triplet_file_name): # If the file still exist
                os.remove(triplet_file_name)
            # Like in C if some error occour the program will exit with a "return 1", for this 
            # conventional reason we exit with 1 here in the exception.
            sys.exit(1) # stop the code here if some error occour 

    else: ################################## non-GPU code ############################################
        if GPU and not core.GPU_enabled(): # If the user wants the gpu but the Device doesn't work
            print('Device (GPU) not available, proceding on Host (CPU)...')
        ############################# FIT RUTINE #####################################
        # extract triplets and coordinates (merged in singles array)
        triplet, X, Y = core.init_triplets(dict_events, maxhits = maxhits, t=triplet_threshold)
        rr, xc, yc = core.py_fit(X, Y, triplet, maxhits, nevents, 
                                 thr = ptolemy_threshold, min_pts = triplet_threshold, rsearch = rsearch, drsearch = drsearch)
        ##############################################################################
     
    # Post-processing of the output from the fit rutines
    # Reshape the result divided by events 
    # example of array rr after the reshape:
    #  
    # evt0    r00     r01    r02    r03        
    # evt1    r10     r11    r12    r13
    # ...              ........ 
    rr = np.reshape(rr, (nevents, 4))
    xc = np.reshape(xc, (nevents, 4))
    yc = np.reshape(yc, (nevents, 4))

    ############### Search of the radius rsearch ############################################
    # The fit returns negative values of radius only if it found radius in rsearh +- drsearch
    if rsearch != 0 : # if the user ask for a radius
        ma_where = np.any(rr < 0, axis = 1) # with axis = 1 we search in each row, with any 
                                            # we got a mask array of lenght nevents with
                                            # true if in a row (in an event) there was a radius 
                                            # in the right range
        evt_pos = list_events[ma_where] # name of events with radius in (r-dr,r+dr)
    else : evt_pos = np.asarray([]) # if the user didn't ask for a radius (rsearch = 0) evt_pos is an empty array
    
    # Make all radius values positive
    rr = np.abs(rr)
    ##########################################################################################
    
    ###### Mean of the similar radius ########################################################
    # If the user want to prune the output radius we can just mean them event per event, this 
    # operatin will slow down the GPU program and is not recomended.
    if means:
        # We will comment just for rad, for xc and yc is analogue.
        #define array of zeros of final results
        rad_mean = np.zeros(rr.shape)
        xc_mean  = np.zeros(xc.shape)
        yc_mean  = np.zeros(yc.shape)
        # shape gives in a tuplet (a non-mutable list) the number of rows and columns of rr, in
        # the position 0 of this tuple we have the number of rows of rr, in the position 1 the 
        # number of columns of rr (4 in our case)
        for i in range(rr.shape[0]):  # i runs on events (rows)
            rad = rr[i] # fit results for each event:
            xci = xc[i] #  this three are numpy arrays of one dimension 
            yci = yc[i] #  containing the 4 values obtained for the single event 'i'.
            for j in range(rr.shape[1]):  #j runs on results for event i (columns)
                # We check the difference from the first element of rad (this first element will change in the 
                # next loop's rounds) to all the other elements of rad, mask where this difference is lower than 
                # the mimimum threshold to mean the values 
                maskr = np.abs( rad - rad[0] ) < meanthr   # verify the difference (boolean array)
                maskx = np.abs( xci - xci[0] ) < meanthr
                masky = np.abs( yci - yci[0] ) < meanthr
                # Logical and of masks for radius, xc and yc, only in the positions in wich this three are True
                # we need to mean the values (because if radius, xc and yc are all similar so the circles are similar)
                mask = maskr & maskx & masky  #boolean array (true where r, xc, yc are similar).
                # Extract the element that will be mean
                r = rad[mask] # delete element where mask is false (contain only similar element)
                x = xci[mask]
                y = yci[mask]
                if rad[0] != 0: # If the checked radius is different from zero (a zero radius means a failed fit...)
                    rad_mean[i, j] = np.mean(r)  # mean value of similar element
                    xc_mean[ i, j] = np.mean(x)
                    yc_mean[ i, j] = np.mean(y)
                # Update rad with the only element non mediated before
                rad = rad[~mask]  #delete similar element for the next iteration (now they are all different from rad[0])
                xci = xci[~mask]
                yci = yci[~mask]
                # in this way rad[0] will be different in the next iteration...
                if len(rad) <= 1: #if remain <= 1 element stop cycle on j (pass to the next event)
                    if len(rad) == 1: #if remain only 1 element copy it in the results array
                        rad_mean[i,j+1] = rad[0] # |
                        xc_mean[i, j+1] = xci[0] # |
                        yc_mean[i, j+1] = yci[0] # | --> (this stuff before break if len(rad) == 1)
                    break # if len(rad) == 0 just go out because the last loop have detected the remaining circle as similar 
        # this is only in the 'if mean', otherwise nothing change...
        rr = rad_mean
        xc = xc_mean
        yc = yc_mean
    return rr, xc, yc, evt_pos

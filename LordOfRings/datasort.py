"""
This module imports and sorts the data names from the folder data.
"""

import numpy as np
import os

def datasort():
    """datasort extracts data names from the folder data and sorts them using 
    two criteria: first the number m of circle in a single matrix (mcircle),
    second the index n of the matrix in the dataset (n.txt).

    Returns
    -------
    1d numpy-array
        the numpy array contains numpy arrays with the sorted names of data,
        one array for each number of circles. 

    """

    # get the working directory
    work_dir = os.getcwd()
    # define the data directory
    data_dir = os.path.join(work_dir,"data/")
    #if the directory does not exist, you have to create it
    if not os.path.exists(data_dir):
        raise FileNotFoundError('No such file or directory: ' + data_dir)

    # extract all dataset
    all_circs = np.array([x for x in os.listdir(data_dir) if x.endswith(".txt")])
    
    # REMINDER of method split:
    #--------------------------
    #| if s is a string containing the separator string (in this case the
    #| separator is the string 'circle_') the method s.split(separator) just 
    #| split the string in two part: what come before the separator (is the 
    #| number of circle in that event) and what comes next the separator 
    #| (in that case what comes next is the number of the event 
    #| relative of the events containing the same number of circles)
    #---------------------------
    # extracting number of circle in the single data in a list
    num_circ_list = np.array([s.split('circle_')[0] for s in all_circs]).astype(int)

    # extracting indexes of data (what come first of .txt)
    num_data_list = np.array([s.split('circle_')[1][:-4] for s in all_circs]).astype(int)

    # argsorting the number of circle (what come first of circle_)
    idx_sorted_circle = np.argsort(num_circ_list)

    ## First reorder of txt based on number of circles
    sorted_circle = all_circs[idx_sorted_circle]

    ## Second reorder based on data
    # create a matrix: 
    # - on rows all the data with same number of circle (what comes before of 'circle_', 
    # - on cols the different data index (what come next of 'circle_', for example 1.txt  2.txt ...).
    nrows = len(np.unique(num_circ_list)) # here we don't need + 1 because the number of circle start from 1
    ncols = np.max(num_data_list) + 1 # here we add + 1 because the number of data start from 0.
    # Reshape the events array (sorted by the number of circle in each event)
    reshaped_sorted_circle = np.reshape(sorted_circle, (nrows, ncols))
    
    # sort every row with argsort
    idx_sorted_data = np.argsort(reshaped_sorted_circle) # argsort launched in this way sort every row separately
    
    # reordering every row in a loop with the index obtained before
    list_sorted = np.array([row[idxrow] for row, idxrow in zip(reshaped_sorted_circle, idx_sorted_data)])
    # (row[idxrow] is an array reordered with an array of index (idx_row))
    return list_sorted.squeeze()

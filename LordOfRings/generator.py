"""
This module manage the creation of random circle in sparse matrix using numpy package.
"""

import numpy as np
import matplotlib.pyplot as plt
import LordOfRings.inputHandling as ih
import os
import sys

def center_radius_generator(mu, sigma, rmin, rmax, n):
    """center_radius_generator generate three random numbers, two from
    a normal distribution and one from an uniform distribuion (with randint).
    The number extracted from the normal distribution are turned to integer
    values because we are working with matrix of integers.

    Parameters
    ----------
    mu : float, greater than or equal to zero, lower than the grid linear size
        The mean for the normal distribution.
    sigma : float, positive value
        The standard deviation for the normal distribution.
    rmax : float, positive value, greater than rmin
        The maximum radius for the uniform extraction.
    rmin : float, greater than or equal to zero, lower than rmax
        The minimum radius for the uniform extraction.
    n : int
        The linear size of the grid.

    Returns
    -------
    (1d numpy-array [int], int)
        the first element of the returned tuple contain the center position,
        the second element contain the radius.

    """
    # Testing input values
    ih.raise_value_error('mu', mu, min = 0, max = n)
    ih.raise_value_error('sigma', sigma, min = 0)
    ih.raise_value_error('rmax', rmax, min = rmin)
    ih.raise_value_error('rmin', rmin, max = rmax)
    ih.raise_value_error('n', n, min = 0)
    
    # we make absolute value in order to remove negative data, + 1 in order to remove the possibility of have a 0 coordinate.
    n_rand = 1 + np.abs(np.random.normal(loc = mu, scale = sigma, size = 2)).astype(int) # lower than zero removed
    # If obtain values greater then the grid linear size change that values
    if n_rand[0] >= n: 
        n_rand[0] = (2 * n - n_rand[0]) - 1 # reflected on axis for come back in (0, n)
    if n_rand[1] >= n:
        n_rand[1] = (2 * n - n_rand[1]) - 1  # reflected on axis for come back in (0, n)
    c = n_rand
    r = np.random.randint(rmin, rmax) # One number from uniform distribution
    return c, r


def circle_generator(n, c, r):
    """circle_generator create a sparse matrix nxn with values 0 or 1.
    The only elements of the matrix that are 1 belong to a circle of radius r
    and center c.

    Parameters
    ----------
    n : int
        The linear size of the matrix
    c : 1d numpy array [float]
        The center's coordinates.
    r : float
        The radius of the circle.

    Returns
    -------
    2d numpy-array [int]
        The matrix of zeros containing the circle as 1 values.

    """
    # Testing input values
    ih.raise_value_error('n', n, min = 0)
    ih.raise_value_error('r', r, min = 0)
    for ci in c: # Testing the center here (not in raise_value_error 'cause there is only <=, here is just <)
        if ci < 0:
            raise ValueError(f'x should be greater than or equal to {0}. The given value was {ci}')
        if ci >= n:
            raise ValueError(f'y should be lower than or equal to {n}. The given value was {ci}')
    #
    v = np.arange(1, n+1) # starting with uniform vector from 1 to n + 1
    Y, X = np.meshgrid(v, v) # create a meshgrid ( https://stackoverflow.com/questions/36013063/what-is-the-purpose-of-meshgrid-in-python-numpy )
    deltaX = np.abs(X-c[0]) # x-xc for all coordinates
    deltaY = np.abs(Y-c[1]) # y-yc for all coordinates
    distance = np.sqrt(deltaX**2 + deltaY**2) # distance from the center for all coordinates
    circle = np.empty(distance.shape) # create an empty matrix (filled with ones in the next lines)
    circle[np.abs(distance - r) > 0.3 ] = 0 # if a point of the distance matrix is more than 0.3 from r write on matrix 0  
    circle[np.abs(distance - r) <= 0.3 ] = 1 # if a point of the distance matrix is 0.3 than 0.3 from r write on matrix 1
    # in this way we have a precision of the points that belong to the circle of 0.3
    return circle # return the sparse matrix containing the event. 


def rnd_circle_pruning(circle, threshold = 0.3, maxhits = 64):
    """rnd_circle_pruning modifies the input sparse matrix randomly removing
    some ones that belongs to the ring. The random selection uses a uniform
    distribution.

    Parameters
    ----------
    circle : 2d numpy-array [int]
        The input sparse matrix that contains the ring as ones.
    threshold : float
        Parameter that controll the extraction, i.e. lower values correspond to
        higer number of ones in the returned matrix and vice versa.
    maxhits : int
        The maximum number of points after the second pruning. If a circle has 
        more than maxhits points after the first pruning remove them randomly
        until they reach maxhits.

    Returns
    -------
    2d numpy-array [int]
        The matrix obtained from the random pruning.

    """

    # Testing input values
    if np.logical_and(circle != 0, circle != 1).any():
        raise ValueError('The input circle must be a sparse matrix of ones.')
    ih.raise_value_error('threshold', threshold, min = 0, max = 1)

    #first pruning : random delete some points of the circle
    randvector = np.random.rand( len(circle[circle!=0]) ) # Initialize a random vector of the same lenght of 
                                                          # the number of ones in the sparse matrix (uniform distrib.)
    randvector[randvector > threshold] = 1   # the points of the random vector that are more than threshold are turned to 1 
    randvector[randvector <= threshold] = 0  # the point of the random vector that are less than threshold are turned to 0
    circle[circle!=0] = randvector # we replace the ones in the matrix with the ones, zeros written in randvector.
    
    #second pruning : remove points to reach a number of ones equal to maxhits.
    # the GPU algoritm can't deal with an arbitrary number of hits, we need to reduce this number in order to fit it to the 
    # GPU architecture.
    ones_arr = circle[circle == 1] # exctract the array of ones in the matrix 
    len_ones = len(ones_arr) # lenght of the array of ones 
    idx_ones = np.arange(len_ones) # equispatial integer array used for indexes the array of ones extracted from circle.
    if len_ones >= maxhits : # if there are more than maxhits ones in the event
        # we choose len_ones - maxhits element from the array idx_ones
        idx_to_zero = np.random.choice(idx_ones, size = len_ones - maxhits, replace = False, p = None)
        # this elements are used as index for turning to zeros some ones of ones_arr untill it reach only maxhits ones
        ones_arr[idx_to_zero] = 0
        # we replace the ones in the matrix with the pruned array
        circle[circle == 1] = ones_arr
    return circle


def data_gen(n, ndata, ncircle, mu = None, sigma = None, rmin = None, rmax = None, threshold = 0.3, seed = None, maxhits = 64):
    """data_gen create set of data containing sparse matrixes of ones in txt
    format. Data are created in a folder named 'data' in the current directory.

    Parameters
    ----------
    n : int
        The linear size of the grid.
    ndata : int
        The number of data set to generate.
    ncircle : int
        The number of circle in each sparse matrix.
    mu : float, greater than or equal to zero, lower than the grid linear size
        The mean for the normal distribution.
    sigma : float, positive value
        The standard deviation for the normal distribution.
    rmin : float, greater than or equal to zero, lower than rmax
        The minimum radius for the uniform extraction.
    rmax : float, positive value, greater than rmin
        The maximum radius for the uniform extraction.
    threshold : float
        Parameter that controll the extraction, i.e. lower values correspond to
        higer number of ones in the returned matrix and vice versa.
    seed : int, default = None
        The seed for the random numbers extraction for reproducibility.
    maxhits : int
        The maximum number of ones in the sparse matrix. It's recommended a 
        value lower then n.

    Returns
    -------
    
    """
    np.random.seed(seed) # if you pass the seed you obtain reproducible result.
    # check the input
    if mu == None: mu = n/2
    if sigma == None: sigma = mu/2
    if rmin == None: rmin = mu/4
    if rmax == None: rmax = mu
    # inizialize matrix of zeros
    circle = np.zeros((n, n), dtype=int)
    
    # Initialize two lenght to write the correct file names
    # len(str) = number of words in that string (es. len('10') = 2)
    # this fix the maximal number of zeros in the data name: 
    #   - es. if ndata = 9  --> we need at max one 0 (for the first file only)
    #   - es. if ndata = 20 --> we need at max two 0 (the first file is '..00.txt')
    zerosdata = len(str(ndata - 1)) 
    zeroscirc = len(str(ncircle - 1)) # the same for the number of circle 
    
    work_dir = os.getcwd() # get the working directory
    data_dir = os.path.join(work_dir,"data") # define the data directory
    if not os.path.exists(data_dir): #if the directory does not exist, you can create it
        os.mkdir(data_dir)
    
    # save auxiliary mean (we change this mean at each iteration in order to separate the centers of the various circles)
    mu0 = mu
    # loop over the number of data
    for i in range(ndata): 
        list_info = '    cx    cy    r    ncircle    \n' # first line of the string with the information about the data i
        # loop over the number of circle for data
        for j in range(ncircle): 
            # we modify the mean in order to separate the circle in the same matrix 
            mu = mu0 + (-1)**(j+i)*n/6 # we elevate ^(j + i ) in this way even if j is just zero (as in case of one single circle) 
                                       # we will obtain the circles in different position at every iteration (over ndata) because
                                       # the number i change.
            # call the function to generate random center and radius
            c, r = center_radius_generator(mu, sigma, rmin, rmax, n)
            # overwrite the matrix circle with the new matrix, at the first iteration this just fill the matrix of zeros, 
            # in the next iteration (thanks to the 'logical or') we will obtain a matrix with more than one circle ( if ncircle > 1)
            #___________________ 
            #| a  b  |  a or b | 
            #| 0  0  |    0    |
            #| 0  1  |    1    | Reminder of 'or' logical table.
            #| 1  0  |    1    |
            #| 1  1  |    1    |
            #|------------------
            circle = np.logical_or(circle_generator(n, c, r), circle)
            # update the info string with the radius and center of current circle
            list_info += f'    {c[0]}    {c[1]}    {r}    {ncircle}    \n'
        # Prune all the circles to reach at max 'maxhits' points
        circle = rnd_circle_pruning(circle, threshold = threshold, maxhits = maxhits)
        # The file name: zfill is used to add the rigth number of zeros in the filename (es. if i = 1 and ndata = 10 then zfill write '01')
        file_name_path = data_dir + f'/{str(ncircle).zfill(zeroscirc)}circle_{str(i).zfill(zerosdata)}.txt'
        # save the fil, fmt is used to write integer (instead of 0.0000 and 1.0000), header is used in order to add 
        # the specification of that event
        np.savetxt(file_name_path, circle, fmt='%0.f', header=list_info)
        # Reset the circle matrix when you pass to the next event
        circle = np.zeros((n, n), dtype=int)

"""
This module manage the creation of random circle in sparse matrix using numpy package.
"""

import numpy as np

def center_radius_generator(mu, sigma, rmin, rmax):
    """center_radius_generator generate three random numbers, two from 
    a normal distribution and one from an uniform distribuion (with randint).
    The number extracted from the normal distribution are turned to integer 
    values because we are working with matrix of integers.

    Parameters
    ----------
    mu : float
        The mean for the normal distribution.
    sigma : float
        The standard deviation for the normal distribution.
    rmax : float
        The maximum radius for the uniform extraction.
    rmax : float
        The minimum radius for the uniform extraction.

    Returns
    -------
    (1d numpy-array [int], int)
        the first element of the returned tuple contain the center position, 
        the second element contain the radius.

    """
    
    # METTERE GLI ASSERT PER TESTARE GLI INPUT 
    
    n_rand = np.random.normal(loc = mu, scale = sigma, size = 2).astype(int)
    # If obtain negative values repeat the extraction
    while np.sum(n_rand < 0) != 0:
        n_rand = np.random.normal(loc = mu, scale = sigma, size = 2).astype(int)
    c = n_rand
    r = np.random.randint(rmin, rmax)
    return c, r


def circle_generator(n, c, r):
    """circle_generator create a sparse matrix nxn with values 0 or 1. 
    The only elements of the matrix that are 1 belong to a circle of radius r
    and center c.

    Parameters
    ----------
    n : int
        The linear size of the matrix
    c : 1d numpy array [int]
        The center's coordinates.
    r : int
        The radius of the circle.

    Returns
    -------
    2d numpy-array [int]
        The matrix of zeros containing the circle as 1 values.

    """
    
    # METTERE GLI ASSERT PER TESTARE GLI INPUT 
    
    v = np.arange(1, n+1)
    X, Y = np.meshgrid(v, v)
    deltaX = np.abs(X-c[0])
    deltaY = np.abs(Y-c[1])
    distance = np.sqrt(deltaX**2 + deltaY**2)
    circle = np.empty(distance.shape)
    circle[np.abs(distance - r) > 0.5 ] = 0
    circle[np.abs(distance - r) <= 0.5 ] = 1
    return circle


def rnd_circle_pruning(circle, threshold = 0.5):
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

    Returns
    -------
    2d numpy-array [int]
        The matrix obtained from the random pruning.

    """
    
    # METTERE GLI ASSERT PER TESTARE GLI INPUT 


    randvector = np.random.rand( len(circle[circle!=0]) )
    randvector[randvector > threshold] = 1
    randvector[randvector <= threshold] = 0
    circle[circle!=0] = randvector
    return circle

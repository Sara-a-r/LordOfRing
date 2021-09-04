""" Internal functions called by the fit function in module LordOfRings.ringfit.multi_ring_fit"""

import numpy as np
import os

try: # why try?
     # Because the modules (when they are imported) starts from the first line of code:
     # when you do 'import LordOfRings.core' you start importing numpy, then pycuda...
     # if pycuda is not installed (i.e. no GPU) so this module gives error and then even the 
     # module that has import core give error and then nothing will works (even in python without gpu...)
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda import gpuarray
    from pycuda.compiler import SourceModule
    from pycuda.tools import DeviceData
except:
    pass


def GPU_enabled():
    """
    This function say to the user if the GPU is able to compute or if
    the installation of pycuda went wrong with a boolean variable.
    

    Returns
    -------
    GPU : Boolean
        If GPU is True the the GPU is able to compute, else there as been an 
        error in the installation of pycuda.
    """

    # Initalize a boolean variable 
    GPU = True
    try: # Try import pycuda and do some random stuff with that
        from pycuda.tools import DeviceData
        specs = DeviceData() # if pycuda is not installed this will trown an error so |
    except:# If the import fails or pycuda can't show Device specs                    |                   
        # Turn the Boolean variable to false <---------------------------------------- 
        GPU = False
    return GPU


def CudaFindTriplet(dict_events, maxhits = 64, threshold = 10):
    """
    CudaFindTriplet uses the GPU to find the triplets that will be used in 
    Ptolemy's theorem to fit the circle. It gives also the x and y coordinates 
    of hits for all events.

    Parameters
    ----------
    dict_events : dictionary
        Dictionary whose keys are the name of .txt files and whose values are  
        the x, y coordinates of the relative event in a list.
        This format is the output of the function LordOfRing.ringfit.load_data.
    
    maxhits : int
        Maximum number of points per event (i.e. the maximum number of ones in
        each sparse matrix). Default is 64.

    threshold : float
        Treshold value for the selection of the triplets, it defines the minimum 
        reciprocal distance of the three points in the same triplet. Default is
        10.

    Returns
    -------
    ( 1d numpy-array, 1d numpy-array, 1d numpy-array ) [float, float, float]
        The triplet and the coordinates in a tuple  (triplet, X, Y).

    """
    module_path = module_path = os.path.dirname(__file__)
    file_name = module_path + '/cuda/triplet.cu'
    X = np.array(list(dict_events.values()))[:, 0].ravel()
    Y = np.array(list(dict_events.values()))[:, 1].ravel()
    nevents = len(dict_events)
    #load and compile Cuda/C file 
    cudaCode = open(file_name,"r")
    myCUDACode = cudaCode.read()
    myCode = SourceModule(myCUDACode, no_extern_c=True)
    #import kernel in python
    FindTriplet = myCode.get_function("triplet")
    typeofdata = np.float32 # Define the data type for the allocations
    # Global memory allocation (empty variables)
    g_triplet   = gpuarray.zeros(12 * nevents, np.int32)
    g_vec = gpuarray.zeros(4 * maxhits * nevents, typeofdata)
    # Load the data of the events
    g_x = gpuarray.to_gpu(X.astype(typeofdata))
    g_y = gpuarray.to_gpu(Y.astype(typeofdata))
    #define geometry of GPU
    tripletsPerEvents = 4
    nThreadsPerBlock = maxhits
    nBlockPerGrid = nevents * tripletsPerEvents
    nGridsPerBlock = 1
    # Call the kerrnel 
    FindTriplet(g_x, g_y, g_triplet, g_vec, np.float32(threshold),
              block=(nThreadsPerBlock, 1, 1),# this control the threadIdx.x (.y and .z)
              grid=(nBlockPerGrid, 1, 1)# this control blockIdx.x ...
              )
    # Getting results
    triplet = g_triplet.get()
    # Free the memory
    g_x.gpudata.free()
    g_vec.gpudata.free()
    g_y.gpudata.free()
    g_triplet.gpudata.free()
    return triplet, X, Y
    
def init_triplets(dict_events, maxhits = 64, t=10, GPU = False):
    """
    init_triplets gives three array: the first contains the index of border hits
    in the sparse matrix of each event whose reciprocal distance is greater then
    t,the second contains the x coordinates of hits for all events in a single 
    array, the third contains the y coordinates as before.

    Parameters
    ----------
    dict_events : dictionary
        Dictionary whose keys are the name of .txt files and whose values are  
        the x, y coordinates of the relative event in a list.
        This format is the output of the function LordOfRing.ringfit.load_data.
    
    maxhits : int
        Maximum number of points per event (i.e. the maximum number of ones in
        each sparse matrix).

    t : float
        Treshold value for the selection of the triplets, it defines the minimum 
        reciprocal distance of the three points in the same triplet.
    GPU : Boolean
        If GPU is True the triplet are extracted with CUDA on GPU, else 
        they are extracted whit a loop on the events and the numpy funtion.
        Default is False.

    Returns
    -------
    ( 1d numpy-array, 1d numpy-array, 1d numpy-array ) [int, float, float]
        The triplet and the coordinates in a tuple  (triplet, X, Y).

    """
    if GPU and GPU_enabled():
        triplet, X, Y = CudaFindTriplet(dict_events, maxhits = maxhits, threshold = t)
    else:
        ########## AUX FUNCTIONS ##########################################################
        def euclidean(xa,ya,xb,yb):
            """Implementation of euclidean distance"""
            return np.sqrt((xa - xb)**2 + (ya - yb)**2)
        
        def triplet_with_threshold(idx_A, xi, yi, thr, sort_xy):
            """
            Idea for triplet detection:
            Starts with the max, min point (A); find his closer* point (B) (that 
            respect the threshold**) and choose it as the second point of the triplet.
            Then find the closer* point to B that respect the threshold** (C), this
            is the third point of the triplet.
            * For closer here we mean respect to the linear distance (i.e. the 
            distance just on one axis).
            ** The threshold instead is evaluated on the euclidean distance.
            Parameters
            ----------
            idx_A : int
                Index of the maximum, minimum element (i.e. the starting point 
                of the triplet).
            xi : 1d numpy-array [float]
                Array of x coordinates relative to the event i.
            yi : 1d numpy-array [float]
                Array of y coordinates relative to the event i.
            thr : float
                Threshold on the distance between the points of the same triplet.
            sort_xy : boolean
                Array in which we sort the final triplet (if the triplet is 
                relative to xmax or xmin this array will be yi and viceversa).
                if 0 sort respect to yi, else sort respect xi.
            Returns
            -------
            1d numpy-array
                The triplet in a numpy array (ciclic ordered yet).
            """
            
            # If sort_xy == 0 it means that we are evaluating xmax or xmin
            if sort_xy == 0: lin_arr = xi # wwe confront the x coordinates
            # else we are evaluating ymax or ymin
            else: lin_arr = yi # we confront the y coordinates
            # Evaluate the linar distance: the distance relative to the axis that
            # we are analyzing (if xmax or xmin -> xi)
            d_lin_A = np.abs(lin_arr[idx_A] - lin_arr) # <----- LINEAR DISTANCE
            
            # extract coordinate of the first element in the triplet
            A = [xi[idx_A], yi[idx_A]]
            # evaluate euclidean distance from A to all the others points
            d_A = euclidean(*A, xi, yi) # <----- EUCLIEAN DISTANCE
            # put an infinite where the threshold on distance is not respected
            # The distance used for the threshold is the euclidean but the mask is applied
            # on the linear distance
            thresh_A = np.where(d_A>thr, d_lin_A, np.infty)
            
            # extract the minimum of this new array (now the values under the thr are
            # setted to infty...), this minimum is the index for the point B
            idx_B = np.argmin(thresh_A)
            # extract B coordinates
            B = [xi[idx_B], yi[idx_B]]
            # evaluate euclidean distance from B to all the others points
            d_B = euclidean(*B, xi, yi)
    
            # put an infinite in d_A where the threshold on distance is not respected
            # for both of distances arrays (d_A and d_B)
            thresh_AB = np.where((d_B > thr) & (d_A > thr), d_lin_A, np.infty)
            
            # we do the argmin of this new array (extracted from d_A) for the index of the 
            # third element of the triplet.
            idx_C = np.argmin([thresh_AB])
            # create the triplet array (it contain just the indices)
            triplet_idx = np.array([idx_A, idx_B, idx_C])
            if sort_xy == 0 : sort_array = yi
            else: sort_array = xi
            # Sort the indices on sort_array
            triplet_idx_sorted = np.argsort(sort_array[triplet_idx])
            # return the indices sorted
            return triplet_idx[triplet_idx_sorted]
        ######################################################################################
        
        nevents = len(dict_events)  # number of events
    
        # Triplet initialized as empty matrix
        #            (event1)
        #           | XMAX1[3] , XMIN1[3], YMAX1[3], YMIN1[3] |
        # Triplet = |(event2)                                 |
        #           | XMAX2[3] , XMIN2[3], YMAX2[3], YMIN2[3] |
        # Later we will flat this array in 1D
        triplet = np.zeros((nevents, 3*4))    
        # extract directly from dict the X and Y with the shape:
        # (nevents, maxhits), if an event has a number of hits lower than 
        # 'maxhits' then his arrays of coordinates have been filled with zeros
        # untill they reach the lenght of 'maxhits'.
        X = np.array(list(dict_events.values()))[:, 0]
        Y = np.array(list(dict_events.values()))[:, 1]
    
        # For each event find the index of the points of the triplets
        for i, (xi, yi) in enumerate(dict_events.values()):
            
            # definition of the triplet with the funciton triplet_with_threshold
            idx_xmax = triplet_with_threshold(np.argmax(xi[xi!=0]), xi, yi, t, 0)
            idx_xmin = triplet_with_threshold(np.argmin(xi[xi!=0]), xi, yi, t, 0)
            idx_ymax = triplet_with_threshold(np.argmax(yi[yi!=0]), xi, yi, t, 1)
            idx_ymin = triplet_with_threshold(np.argmin(yi[yi!=0]), xi, yi, t, 1)
            
            # Fill the triplet with maximum and minumum relative to the event
            triplet[i,  :3]  = idx_xmax + i * maxhits #XMAX
            triplet[i, 3:6]  = idx_xmin + i * maxhits #XMIN
            triplet[i, 6:9]  = idx_ymax + i * maxhits #YMAX
            triplet[i, 9: ]  = idx_ymin + i * maxhits #YMIN
    
        triplet = triplet.flatten()
        X = X.flatten()
        Y = Y.flatten()
    return triplet.astype(int), X, Y


def pypto(x, y, trip): 
    """
    This function returns the "ptolemy value" given four points
    x and y are single coordinates, trip contain 6 coordinates, 3 for the x 
    and 3 for the y coordinates of the points of the triplet. 
    
    Parameters
    ----------
    x : float or 1d numpy-array [float]
        x coordinates of the point D in the Ptolemy Theorem
    y : float or 1d numpy-array [float]
        y coordinates of the point D in the Ptolemy Theorem
    tripl: list of float
        List containing the coordinates of the triplet points A, B, C:
        [Ax, Ay, Bx, By, Cx, Cy]
        
    Returns
    -------
    float or 1d numpy-array [float]
        The Ptolemy value (or values in case of x, y as array).
    """
    def d(xa, ya, xb, yb): # distance function 
        dist = np.sqrt( ( xa - xb )**2 + ( ya - yb )**2)
        return dist
    A = [trip[0], trip[1]] # [xa, ya] # A, B, C are composed each by 2 numbers:  
    B = [trip[2], trip[3]] # [xb, yb] # the coordinates of A, of B, of C
    C = [trip[4], trip[5]] # [xc, yc]
    D = [x, y]             # [xd, yd] # here I can pass a numpy-array and in this case
                           #            the function will return an arrays.
           # The ptolemy value for the triplet 'trip' 
    return d(*A, *B) * d(*C, *D) + d(*A, *D) * d(*C, *B) - d(*A, *C) * d(*B, *D)

def ptolemy_candidates(triplet, X, Y, idx_event, idx_trip, thr, nevents, maxhits):
    """
    This function is able to find the points that respect the ptolemy 
    theorem giving a triplet and an event.
    
    Parameters
    ----------
    triplet : 1d numpy-array of size (4 * 3 * nevent) [float]
        Contain the indexs of the triplets chosen for the application of the 
        Ptolemy theorem for each event (generated with
        LorfOfRings.ringfit.init_triplet).
        
    X : 1d numpy-array of size (MAXHITS * nevent) [float]
        The x coordinates of ones in all the sparse matrix (generated with 
        LorfOfRings.ringfit.init_triplet). 
        
    Y : 1d numpy-array of size (MAXHITS * nevent) [float]
        The y coordinates of ones in all the sparse matrix (generated with 
        LorfOfRings.ringfit.init_triplet).
    
    idx_event : int
        Index of the event analized in this function.
    
    idx_trip : int
        Index of the triplet relative to the event 'idx_event' analized in 
        this function.
    
    thr : float
        Threshold for the Ptolemy theorem: if a point fits in the equation 
        of the theorem with a maximal deviation of 'thr' it is stored as 
        a good candidate for the triplet 'idx_trip'.

    nevents : int
        The total number of events in the dataset analized.
    
    maxhits : int
        Maximum number of points per event (i.e. the maximum number of ones in
        each sparse matrix).
    
    Returns
    -------
    (1d numpy-array, 1d numpy-array, 1d numpy-array, 1d numpy-array)
    [float, float, int, float]
        The X, Y coordinates of the candidates to the circle fit, 
        the index of the coordinates of the triplet relative to that event, 
        the coordinates of the triplet in the same format passed to the 
        function pypto ([Ax, Ay, Bx, By, Cx, Cy]).
    """
    # since triplet contain only indices of the element of X and Y 
    # we need to extract from this array the coordinates of the 
    # elements of the triplets in order to evaluate Ptolemy: 
    xx = X[triplet]
    yy = Y[triplet]

    # Here we extract only the one triplet that we are analyzing in this loop
    # step, at this purpose we create an empty triplet that will be filled with 
    # the coordinates of the triplet that we are looking for (in this loop step).
    # NOTE: triplet contain index, trip_coord will contain coordinates
    trip_coord = np.empty(6)
    idx_tripl1 = 4*3*idx_event + idx_trip*3
    #            _____________   __________
    #                  |             |
    #                  #-------------|----------> # (4 triplet per event * 3 element per triplet = 12 index per event)
    #                                |            # we need to jump the first 12 * idx_event points of the array in order to  
    #                                |            # arrive at the right 12 element (index of triplets) of the event 'idx_event'.
    #                                #----------> # ( 3 element per triplet ) * idx_triplet allow us to arrive at the right 
    #                                             # triplet of that event 
    #----------------------------------------------------------------------------------------------------------
    # Example of triplets coordinates (in xx, yy array):
    #         <-----------------evt0--------------->  <----------evt1----------> ... etc...      
    #   xx = [xx0, xx1, xx2, xx3, xx4, xx5,..., xx11, xx12, x13, xx14, ..., xx23 ... , xx(12*nevents - 1)]
    #         |            ||             |          |               |
    #         <---trip0--->  <---trip1---> ..2..3..   <----trip0---->
    #         |            ||             |          |               |
    #   yy = [yy0, yy1, yy2, yy3, yy4, yy5,..., yy11, yy12, yy13, yy14, ..., yy23 ... , yy(12*nevents - 1)]
    #         <-----------------evt0--------------->  <----evt1----> ... etc...      
    #----------------------------------------------------------------------------------------------------------
    # The index where the idx_trip triplet end is just 3 idx later, just add + 3 to the idx_trip1 for this second index
    idx_tripl2 = idx_tripl1 + 3
    
    # we write the coordinates of that triplet in the list 'trip_coord'
    # alternating one x and one y
    triplet_x = xx[idx_tripl1:idx_tripl2]
    triplet_y = yy[idx_tripl1:idx_tripl2]
    # here we create the triplet coordinates vector like this: [tx1, ty1, tx2, ty2, tx3, ty3]
    # we stack on axis 1 the coordinates x, y of the triplet, the ravel function is the same then flatten.
    trip_coord = np.stack((triplet_x, triplet_y), axis = 1).ravel()

    #---------------------------------
    # Example of X with maxhits = 64
    #      <---evt0---> ...
    # X = [x0, ..., x63, x64, ... ]
    #--------------------------------
    # separate only the maxhits points that belong to the event idx_event (watch the previous figure)
    Xinterval = X[ maxhits*idx_event : maxhits*(idx_event+1) ]
    Yinterval = Y[ maxhits*idx_event : maxhits*(idx_event+1) ]
    
    # Evaluate the ptolemy value in each of the 'maxhits' point of the event at the same time
    ypto = pypto(Xinterval, Yinterval, trip_coord) # this is an array of 'maxhits element' with the
                                                   # ptolemy's values for each couple of (Xinterval, Yinterval)
    
    # Mask in Xca, Yca the coordinates that respect the ptolemy theorem for the triplet idx_triplet 
    Xca = Xinterval[ypto<thr] # 'ca' is for 'canditate' like in the cuda code
    Yca = Yinterval[ypto<thr]
    # extract the index of the coordinates of the event idx_event using the idx_trip explained before
    my_triplet = triplet[idx_tripl1:idx_tripl2] - idx_event * maxhits # we remove idx_event * maxhits because
                                                                      # the index referred to whole arrays X, Y. 
                                                                      # we are just condidering a portion for this event
                                                                      # that start from idx_event * maxhits
    return Xca, Yca, my_triplet, trip_coord


def taubin(Xca, Yca, min_pts):
    """
    This function execute the circle fit of a single event with the 
    Taubin's Method.
    
    Parameters
    ----------
    Xca : 1d numpy-array [float]
        x coordinates of points of the circle for the fit.
    Yca : 1d numpy-array [float]
        y coordinates of points of the circle for the fit.
    min_pts : 
        Minimum lenght of Xca, Yca for execute the fit, this parameter is 
        for remove unaccurated fits. It has to be greater then 3.
    
    Returns
    -------
    (1d numpy-array, 1d numpy-array, 1d numpy-array)
        The radius, Xc, Yc evaluated by the fit rutine.
    """
    if (len(Xca)<min_pts) or (len(Yca)<min_pts):
        return 0, 0, 0
    max_iteration = 5
    # Initialize the quantity required by Taubin 
    xm, ym = np.mean(Xca) , np.mean(Yca)
    u = Xca - xm
    v = Yca - ym
    z = u**2 + v**2
    u2, v2 , z2 = u**2, v**2, z**2
    u2av, v2av, z2av = np.mean(u2), np.mean(v2), np.mean(z2)
    zav = np.mean(z)
    uvav, uzav, vzav = np.mean(u*v), np.mean(u*z), np.mean(v*z)
    covXY = u2av*v2av - uvav**2
    varZ  = z2av - zav**2
    c0 = uzav * (uzav * v2av - vzav * uvav) + \
         vzav * (vzav * u2av - uzav * uvav) - varZ * covXY
    c1 = varZ * zav + 4 * covXY * zav - uzav**2 - vzav * vzav
    c2 = -3 * zav * zav - z2av
    c3 = 4 * zav
    c22 = 2*c2
    c33 = 3*c3
    # Starting the Newton Methods for findng roots
    eta = 0
    poly = c0
    for i in range(max_iteration):
        derivative = c1 + eta * (c22 + eta * c3)
        if derivative == 0:
            break
        eta_new = eta - poly/derivative
        if (eta_new == eta) or (np.isnan(eta_new)):
            break
        poly_new = c0 + eta_new * (c1 + eta_new * (c2 + eta_new * c3))
        if np.abs(poly_new) >=np.abs(poly):
            break
        eta = eta_new
        poly = poly_new
    # End of Newton's methods, now we extract the circle's parameters
    det = eta * eta - eta * zav + covXY
    if det == 0:
        r, xc, yc = 0, 0, 0
    else: 
        uc = (uzav * (v2av - eta) - vzav * uvav)/(2 * det)
        vc = (vzav * (u2av - eta) - uzav * uvav)/(2 * det)
        alpha = uc * uc + vc * vc + zav
        r = np.sqrt(alpha)
        xc = uc + xm 
        yc = vc + ym
    return r, xc, yc

def py_fit(X, Y, triplet, maxhits, nevents, thr, min_pts, rsearch, drsearch):
    """
    This function loops over the event executing sequentially a number 'nevents'
    of fits in python (optimized with numpy).
    
    Parameters
    ----------

    X : 1d numpy-array of size (MAXHITS * nevent) [float]
        The x coordinates of ones in all the sparse matrix (generated with 
        LorfOfRings.ringfit.init_triplet). 
        
    Y : 1d numpy-array of size (MAXHITS * nevent) [float]
        The y coordinates of ones in all the sparse matrix (generated with 
        LorfOfRings.ringfit.init_triplet).

    triplet : 1d numpy-array of size (4 * 3 * nevent) [float]
        Contain the indexs of the triplets chosen for the application of the 
        Ptolemy theorem for each event (generated with
        LorfOfRings.ringfit.init_triplet).
        
    maxhits : int
        Maximum number of points per event (i.e. the maximum number of ones in
        each sparse matrix).
        
    nevents : int
        The total number of events in the dataset analized.
    
    thr : float
        Threshold for the Ptolemy theorem: if a point fits in the equation 
        of the theorem with a maximal deviation of 'thr' it is stored as 
        a good candidate for the triplet 'idx_trip'.
        
    min_pts : 
        Minimum lenght of Xca, Yca for execute the fit, this parameter is 
        for remove unaccurated fits.
        
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
    
    Returns
    -------
    (2d numpy-array, 2d numpy-array, 2d numpy array) [float, float, float]
        Array of Radius, Xcenters, Ycenters of lenght nevents. 
        Each event has four prediction of center and radius, one for each 
        triplet evaluated (4 triplet for event). For this reason the result's 
        arrays are 2d: different rows index different event, different columns 
        index different triplet.
    """
    # Initialize empty array of the results
    rr = np.empty((nevents,4))
    xc = np.empty((nevents,4))
    yc = np.empty((nevents,4))
    # start loop over events and index
    for idx_event in range(nevents):
        for idx_trip in range(4):
            # Extract the candidate for the fit
            Xca, Yca, _, _ = ptolemy_candidates(triplet, X, Y, idx_event, idx_trip, thr, nevents, maxhits)
            # Taubin algorithm for the triplet 'idx_trip' of the event 'idx_event'
            r_in, xc_in, yc_in = taubin(Xca, Yca, min_pts)
            # check for radii in the interval rsearch +- rsearch
            if (r_in < rsearch + drsearch) and (r_in > rsearch - drsearch):
                r_in = -1*r_in
            # Write the result in the output matrices
            rr[idx_event, idx_trip] = r_in
            xc[idx_event, idx_trip] = xc_in
            yc[idx_event, idx_trip] = yc_in
    return rr, xc, yc

def CUDA_fit(file_name, triplet_file_name, dict_events, maxhits, triplet_threshold, ptolemy_threshold, rsearch, drsearch):
    """
    CUDA_fit manages the gpu function call, defining the structure of 
    threads and blocks and allocating the necessary memory, this function is 
    called by the function multi_ring_fit and should not be called directly by the user.
    
    Parameters
    ----------
    file_name : string
        String containing the name of the file with the cuda fit Kernel.
    
    triplet_file_name : string
        String containing the name of the file with the cuda triplet search Kernel.

    maxhits : int
        Maximum number of points per event (i.e. the maximum number of ones in
        each sparse matrix).
    
    triplet_threshold : float 
        Minimum distance between two points of the same triplet.
        
    ptolemy_threshold : float
        Maximum value of the difference defined by the Ptolemy theorem: the points
        for which this difference is greater than ptolemy_threshold are excluded
        by the fit algorithm.
    
    rsearch : float
        The radius that will be searched in the data after the fit, if zero 
        the fourth argument returned by the function will be an empty 
        numpy array.
    
    drsearch : float
        The error on the radius rsearch, the function will return all the 
        events having a fitted radius in 
        (rsearch - drsearch, rsearch + drsearch)
        
    Returns
    -------
    (1d numpy-array, 1d numpy-array, 1d numpy array) [float, float, float]
        Array of Radius, Xcenters, Ycenters of lenght 4 * nevents. 
        Each event has four prediction of center and radius, one for each 
        triplet evaluated (4 triplet for event).
    """
    nevents = len(dict_events)
    #---------------------------------
    # Example of X with maxhits = 64
    #      <---evt0---> ...
    # X = [x0, ..., x63, x64, ... ]
    #--------------------------------
    X = np.array(list(dict_events.values()))[:, 0].flatten()
    Y = np.array(list(dict_events.values()))[:, 1].flatten()

    # ----- GPU GEOMETRY -----------------------
    tripletsPerEvents = 4
    nThreadsPerBlock = maxhits
    nBlockPerGrid = nevents * tripletsPerEvents
    
    # ------------------ FIND TRIPLETS -----------------------------------------------
    #load and compile Cuda/C file 
    cudaCode = open(triplet_file_name,"r")
    myCUDACode = cudaCode.read()
    myCode = SourceModule(myCUDACode, no_extern_c=True)
    #import kernel in python
    FindTriplet = myCode.get_function("triplet")
    typeofdata = np.float32 # Define the data type for the allocations
    
    # Global memory allocation (empty variables)
    g_triplet = gpuarray.zeros(12 * nevents, np.int32)
    g_vec = gpuarray.zeros(4 * maxhits * nevents, typeofdata) # vector for internal operation in cuda file
    
    # Load the data of the events
    g_x = gpuarray.to_gpu(X.astype(typeofdata))
    g_y = gpuarray.to_gpu(Y.astype(typeofdata))

    nGridsPerBlock = 1
    # Call the kerrnel 
    FindTriplet(g_x, g_y, g_triplet, g_vec, np.float32(triplet_threshold),
               block=(nThreadsPerBlock, 1, 1),# this control the threadIdx.x (.y and .z)
               grid=(nBlockPerGrid, 1, 1)# this control blockIdx.x ...
               )
    # Free the memory
    g_vec.gpudata.free()

    # ------------------  FIT ROUTINE ------------------------------------------------
    #load and compile Cuda/C file 
    cudaCode = open(file_name,"r")
    myCUDACode = cudaCode.read()
    myCode = SourceModule(myCUDACode, no_extern_c=True)
    #import kernel in python
    MultiRing = myCode.get_function("multiring")
    typeofdata = np.float32 # Define the data type for the allocations
    # Global memory allocation (empty variables)
    g_xca = gpuarray.zeros( 4*maxhits*nevents, typeofdata)
    g_yca = gpuarray.zeros( 4*maxhits*nevents , typeofdata)
    g_xm  = gpuarray.zeros( 4*nevents, typeofdata)
    g_ym  = gpuarray.zeros( 4*nevents, typeofdata)
    g_u   = gpuarray.zeros( 4*maxhits*nevents, typeofdata )
    g_v   = gpuarray.zeros( 4*maxhits*nevents, typeofdata )
    g_z   = gpuarray.zeros( 4*maxhits*nevents, typeofdata )
    g_u2  = gpuarray.zeros( 4*maxhits*nevents, typeofdata )
    g_v2  = gpuarray.zeros( 4*maxhits*nevents, typeofdata )
    g_z2  = gpuarray.zeros( 4*maxhits*nevents, typeofdata )
    g_uz  = gpuarray.zeros( 4*maxhits*nevents, typeofdata )
    g_vz  = gpuarray.zeros( 4*maxhits*nevents, typeofdata )
    g_uv  = gpuarray.zeros( 4*maxhits*nevents, typeofdata )
    g_zav = gpuarray.zeros( 4*nevents, typeofdata )
    g_z2av= gpuarray.zeros( 4*nevents, typeofdata )
    g_u2av= gpuarray.zeros( 4*nevents, typeofdata )
    g_v2av= gpuarray.zeros( 4*nevents, typeofdata )
    g_uvav= gpuarray.zeros( 4*nevents, typeofdata )
    g_uzav= gpuarray.zeros( 4*nevents, typeofdata )
    g_vzav= gpuarray.zeros( 4*nevents, typeofdata )
    g_xc  = gpuarray.zeros( 4*nevents, typeofdata )
    g_yc  = gpuarray.zeros( 4*nevents, typeofdata )
    g_r   = gpuarray.zeros( 4*nevents, typeofdata )
    # Call the kerrnel 
    MultiRing(g_x, g_y, g_triplet, np.float32(ptolemy_threshold), 
              g_xca, g_yca, 
              g_xm, g_ym,
              g_u, g_v, g_z, 
              g_u2, g_v2, g_z2,
              g_uz, g_vz, g_uv, 
              g_zav, g_z2av, g_u2av, g_v2av,
              g_uvav, g_uzav, g_vzav, np.int32(nevents), 
              g_xc, g_yc, g_r,
              np.float32(rsearch), np.float32(drsearch),
              block=(nThreadsPerBlock, 1, 1),# this control the threadIdx.x (.y and .z)
              grid=(nBlockPerGrid, 1, 1)# this control blockIdx.x ...
              )
    # Getting results
    r, x, y = g_r.get(), g_xc.get(), g_yc.get()
    # Free the memory
    g_x.gpudata.free()
    g_y.gpudata.free()
    g_xm.gpudata.free()
    g_ym.gpudata.free()
    g_u.gpudata.free()
    g_v.gpudata.free()
    g_z.gpudata.free()
    g_u2.gpudata.free()
    g_v2.gpudata.free()
    g_z2.gpudata.free()
    g_uz.gpudata.free()
    g_vz.gpudata.free()
    g_uv.gpudata.free()
    g_zav.gpudata.free()
    g_z2av.gpudata.free()
    g_u2av.gpudata.free()
    g_v2av.gpudata.free()
    g_xc.gpudata.free()
    g_yc.gpudata.free()
    g_r.gpudata.free()
    g_xca.gpudata.free()
    g_yca.gpudata.free()
    g_triplet.gpudata.free()
    return r, x, y


"""
This module fit multiple circle in sparse matrix
"""

import numpy as np
import os
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda import gpuarray
    from pycuda.compiler import SourceModule
    from pycuda.tools import DeviceData
    GPU = True
except:
    GPU = False

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
        x,y coordinates of the relative event.

    """
    dict_data = {}
    for circle_name in list_events:
        circle = np.loadtxt('data/'+circle_name)
        coord = np.argwhere(circle!=0)
        X = coord[:, 0] + 1
        Y = coord[:, 1] + 1
        dict_data[circle_name] = [X, Y]
    return dict_data


def init_triplets(dict_events, maxhits = 64, t=10):
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

    Returns
    -------
    ( 1d numpy-array, 1d numpy-array, 1d numpy-array ) [int, float, float]
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

    def d3(x, y, z, t, max):
        # this function return boolean array of len = 3:
        # - True if the d[i, j] > t
        # - False otherwise. 
        dxy = np.linalg.norm(x-y)
        dyz = np.linalg.norm(y-z)
        dxz = np.linalg.norm(x-z)
        if max: # for xmax and ymax
            if (dyz>t): # if the element y respects the threshold check only xy
                return np.array([(dxy>t) and (dxz>t), True, True]).astype(bool)
            if (dxz>t): # if the element x respects the threshold check only xy
                return np.array([True, (dxy>t) and (dyz>t), True]).astype(bool)
            # if both x and y don't respect the t.. check both of them (dxz and dyz)
            return np.array([(dxz>t), (dyz>t), True]).astype(bool)
        else: # for xmin and ymin 
            if (dxy>t): # if the element y respects the threshold check only yz
                return np.array([True, True, (dyz>t) and (dxz>t)]).astype(bool)
            if (dxz>t): # if the element z respects the threshold check only yz
                return np.array([True, (dyz>t) and (dxy>t), True]).astype(bool)

            return np.array([True, (dxy>t), (dxz>t)]).astype(bool)

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
            bminx = d3(*xsortv[:3], t, max = False)
            bmaxx = d3(*xsortv[-3:], t, max = True)
            bminy = d3(*ysortv[:3], t, max = False)
            bmaxy = d3(*ysortv[-3:], t, max = True)

            # if all the values in all the 4 bool array are true
            if bminx.all() and bmaxx.all() and bminy.all() and bmaxy.all():
                # we've the right triplets and we exit
                boolcheck = False
            else:
                # sobstitute in boolarray the first/last three bool elements
                boolarrayx[:3] = bminx
                boolarrayx[-3:] = bmaxx
                boolarrayy[:3] = bminy
                boolarrayy[-3:] = bmaxy
                
                if (np.sum(boolarrayx) < 3) or (np.sum(boolarrayy) < 3): # if the element are finished
                    boolcheck = False # get out with this index
                else:
                    # delete (with the mask boolarrayx) the elements that not 
                    # satisfy the contition
                    idx_sx = idx_sx[boolarrayx]
                    xsortv = xsortv[boolarrayx]
                    
                    # same for the y index ordered array
                    idx_sy = idx_sy[boolarrayy]
                    ysortv = ysortv[boolarrayy]
                
        # sorting in the single triplet for the opposite variable (for ptolemy)
        iy_sort_xmax = np.argsort(xsortv[-3:, 1]) # index sorted based on y for xmax
        iy_sort_xmin = np.argsort(xsortv[:3, 1])  # ...
        ix_sort_ymax = np.argsort(ysortv[-3:, 0])
        ix_sort_ymin = np.argsort(ysortv[:3, 0])
        
        # Fill the triplet with maximum and minumum relative to the event
        triplet[i,  :3]  = idx_sx[-3:][iy_sort_xmax] + i * maxhits #XMAX
        triplet[i, 3:6]  = idx_sx[ :3][iy_sort_xmin] + i * maxhits #XMIN
        triplet[i, 6:9]  = idx_sy[-3:][ix_sort_ymax] + i * maxhits #YMAX
        triplet[i, 9: ]  = idx_sy[ :3][ix_sort_ymin] + i * maxhits #YMIN
    triplet = triplet.flatten()
    X = X.flatten()
    Y = Y.flatten()
    return triplet.astype(int), X, Y
    

def multi_ring_fit(X, Y, triplet, MAXHITS=64):
    """
    multi_ring_fit executes multiple circle fits on arbitrary number of events
    giving the values predicted by the alghoritm.

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

    Returns
    -------
    ( 2d numpy-array, 2d numpy-array, 2d numpy-array ) [float, float, float]
        The radius, X center, Ycenter of all the events in three array. These
        values are distributed in different row, one for each event.

    """
    # Modify a constant (MAXHINTS) directly in the .cu file starting from MultiRing.cu
    file_name = f"{MAXHITS}MultiRing.cu" # create file instead of overwrite the original 

    if os.path.exists(file_name)==False: # if doesn't exist write it
        lines = open(os.path.dirname(__file__)+'/MultiRing.cu', 'r').readlines()
        line_num = 4 # I know that MAXHITS is at line 4
        lines[line_num] = f'#define MAXHITS (int) {MAXHITS}\n'
        out = open(file_name, 'w')
        out.writelines(lines)
        out.close()

    if GPU:
        #load and compile Cuda/C file 
        cudaCode = open(file_name,"r")
        myCUDACode = cudaCode.read()
        myCode = SourceModule(myCUDACode, no_extern_c=True)
        #import kernel in python
        MultiRing = myCode.get_function("multiring")

        nevents = int(len(X)/MAXHITS) # Deduce the number of events thanks to X and MAXHITS
        typeofdata = np.float32 # Define the data type for the allocations
        # Global memory allocation (empty variables)
        g_xca = gpuarray.zeros( 4*MAXHITS*nevents, typeofdata)
        g_yca = gpuarray.zeros( 4*MAXHITS*nevents , typeofdata)
        g_xm  = gpuarray.zeros( 4*nevents, typeofdata)
        g_ym  = gpuarray.zeros( 4*nevents, typeofdata)
        g_u   = gpuarray.zeros( 4*MAXHITS*nevents, typeofdata )
        g_v   = gpuarray.zeros( 4*MAXHITS*nevents, typeofdata )
        g_z   = gpuarray.zeros( 4*MAXHITS*nevents, typeofdata )
        g_u2  = gpuarray.zeros( 4*MAXHITS*nevents, typeofdata )
        g_v2  = gpuarray.zeros( 4*MAXHITS*nevents, typeofdata )
        g_z2  = gpuarray.zeros( 4*MAXHITS*nevents, typeofdata )
        g_uz  = gpuarray.zeros( 4*MAXHITS*nevents, typeofdata )
        g_vz  = gpuarray.zeros( 4*MAXHITS*nevents, typeofdata )
        g_uv  = gpuarray.zeros( 4*MAXHITS*nevents, typeofdata )
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

        # Load the data of the events
        g_x = gpuarray.to_gpu(X.astype(typeofdata))
        g_y = gpuarray.to_gpu(Y.astype(typeofdata))
        g_triplet = gpuarray.to_gpu(triplet.astype(np.int32))

        #define geometry of GPU
        tripletsPerEvents = 4
        nThreadsPerBlock = MAXHITS 
        nBlockPerGrid = nevents * tripletsPerEvents
        nGridsPerBlock = 1

        # Call the kerrnel 
        MultiRing(g_x, g_y, g_triplet, 
                  g_xca, g_yca, 
                  g_xm, g_ym,
                  g_u, g_v, g_z, 
                  g_u2, g_v2, g_z2,
                  g_uz, g_vz, g_uv, 
                  g_zav, g_z2av, g_u2av, g_v2av,
                  g_uvav, g_uzav, g_vzav, np.int32(nevents), 
                  g_xc, g_yc, g_r, 
                  block=(nThreadsPerBlock, 1, 1),# this control the threadIdx.x (.y and .z)
                  grid=(nBlockPerGrid, 1, 1)# this control blockIdx.x ...
                  )
        # Getting results
        r, xc, yc = g_r.get(), g_xc.get(), g_yc.get()
        # Reshape the result divided by events
        r = np.reshape(r, (nevents, 4))
        xc = np.reshape(xc, (nevents, 4))
        yc = np.reshape(yc, (nevents, 4))

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

    else:
        r = np.zeros((nevents, 4))
        xc = np.zeros((nevents, 4))
        yc = np.zeros((nevents, 4))
    # Remove the file .cu generated
    os.remove(file_name)

    return r, xc, yc

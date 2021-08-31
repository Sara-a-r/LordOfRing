# include <stdio.h>
# include <cuda_runtime.h>

// const
#define MAXHITS (int) 128
#define MAX_ITERATIONS (int) 5

// CUDA Kernel
extern "C" {
__device__ int minCommSingleWarp(volatile float * shArr, volatile int * shIdx) {
    int idx = threadIdx.x % warpSize; //the lane index in the warp
    if (idx<16) {
      // This operation are execute on all idx at the same time
      // just like the classical figure of reduction but with min
      // with the or (||) in the last condition we ensure that if a point contain -1 it will be
      // substitute.
      if (((shArr[idx+16] < shArr[idx]) && (shArr[idx + 16] > 0 )) || (shArr[idx]==-1)){
          // if respect the minimum replace
          shArr[idx] = shArr[idx + 16]; // move the values
          shIdx[idx] = shIdx[idx + 16]; // move the index
      }
      if (((shArr[idx+8] < shArr[idx]) && (shArr[idx + 8] > 0 )) || (shArr[idx]==-1)){
          shArr[idx] = shArr[idx + 8];
          shIdx[idx] = shIdx[idx + 8];
      }
      if (((shArr[idx+4] < shArr[idx]) && (shArr[idx + 4] > 0 )) || (shArr[idx]==-1)){
          shArr[idx] = shArr[idx + 4];
          shIdx[idx] = shIdx[idx + 4];
      }
      if (((shArr[idx+2] < shArr[idx]) && (shArr[idx + 2] > 0 )) || (shArr[idx]==-1)){
          shArr[idx] = shArr[idx + 2];
          shIdx[idx] = shIdx[idx + 2];
      }
      if (((shArr[idx+1] < shArr[idx]) && (shArr[idx + 1] > 0 )) || (shArr[idx]==-1)){
          shArr[idx] = shArr[idx + 1];
          shIdx[idx] = shIdx[idx + 1];
      }
    }
    return shIdx[0];
}

__device__ int maxCommSingleWarp(volatile float * shArr, volatile int * shIdx) {
    int idx = threadIdx.x % warpSize; //the lane index in the warp
    if (idx<16) {
      // This operation are execute on all idx at the same time
      // just like the classical figure of reduction
      // like the minCommSingleWarp but this time without the check on -1 (for the max -1 is not a problem)
      if ((shArr[idx+16] > shArr[idx]) && (shArr[idx + 16] > 0 )){
          shArr[idx] = shArr[idx + 16];
          shIdx[idx] = shIdx[idx + 16];
      }
      if ((shArr[idx+8] > shArr[idx]) && (shArr[idx + 8] > 0 )){
          shArr[idx] = shArr[idx + 8];
          shIdx[idx] = shIdx[idx + 8];
      }
      if ((shArr[idx+4] > shArr[idx]) && (shArr[idx + 4] > 0 )){
          shArr[idx] = shArr[idx + 4];
          shIdx[idx] = shIdx[idx + 4];
      }
      if ((shArr[idx+2] > shArr[idx]) && (shArr[idx + 2] > 0 )){
          shArr[idx] = shArr[idx + 2];
          shIdx[idx] = shIdx[idx + 2];
      }
      if ((shArr[idx+1] > shArr[idx]) && (shArr[idx + 1] > 0 )){
          shArr[idx] = shArr[idx + 1];
          shIdx[idx] = shIdx[idx + 1];
      }
    }
    return shIdx[0];
}

__device__ void VecMin(const float *a, int *out, int arraySize) {
    int idx = threadIdx.x; // Index of the thread used

    __shared__ float r[MAXHITS];
    __shared__ int idx_r[MAXHITS];
    r[idx] = a[idx];
    idx_r[idx] = idx;

    if (arraySize <= warpSize){
      minCommSingleWarp( r , idx_r);
      *out = idx_r[0];
    }
    else {
      minCommSingleWarp( &r[idx & ~(warpSize - 1)] , &idx_r[idx & ~(warpSize - 1)] );
      __syncthreads();
      if (idx<warpSize) { //first warp only
        r[idx] = idx * warpSize < MAXHITS ? r[idx*warpSize] : 0;
        idx_r[idx] = idx * warpSize < MAXHITS ? idx_r[idx*warpSize] : 0;
        minCommSingleWarp(r, idx_r);
        if (idx == 0)
            *out = idx_r[0];
      }
    }
}

__device__ void VecMax(const float *a, int *out, int arraySize) {
    int idx = threadIdx.x; // Index of the thread used

    __shared__ float r[MAXHITS];
    __shared__ int idx_r[MAXHITS];
    r[idx] = a[idx];
    idx_r[idx] = idx;

    if (arraySize <= warpSize){
      maxCommSingleWarp( r , idx_r);
      *out = idx_r[0];
    }
    else {
      maxCommSingleWarp( &r[idx & ~(warpSize - 1)] , &idx_r[idx & ~(warpSize - 1)] );
      __syncthreads();
      if (idx<warpSize) { //first warp only
        r[idx] = idx * warpSize < MAXHITS ? r[idx*warpSize] : 0;
        idx_r[idx] = idx * warpSize < MAXHITS ? idx_r[idx*warpSize] : 0;
        maxCommSingleWarp(r, idx_r);
        if (idx == 0)
            *out = idx_r[0];
      }
    }
}

__device__ void LinDist(const int idxA, const int idxB, int linAxis, const float * x, const float * y,
                        float * outArr, float threshold, int arraySize){
    /*
    Evaluate the linear distance from the point indexed by idxA and all the others
    respect to a given axes decided by linAxis, put this distance in outArr.
    Check if this the euclidean distance of the point idxA and idx B (from all the others)
    is greater then threshold, otherwise put in outArr -1 (in the position not respecting the threshold)
    PARAMETERS
     - idxA : index of the first element of triplet
     - idxB : index of the second element of triplet
          (if not exist yet you pass here the same of idxA, so just evaluate the
          euclidean distance of the first point two times).
     - linAxis : 0 or 1
          If 0 the linear axis is x, otherwise is y.
     - x, y : pointer to arrays
          The arrays of the coordinates of the events.
     - outArr : pointer to output array
     - threshold : float value
          Threshold on the distance of point in the same triplet.
     -
    */
    int idx = threadIdx.x; // with this index we evaluate the distance relative
                           // to all the other events.
    float dist1 = sqrt( pow(x[idxA] - x[idx] , 2) + pow(y[idxA] - y[idx] , 2) );
    float dist2 = sqrt( pow(x[idxB] - x[idx] , 2) + pow(y[idxB] - y[idx] , 2) );
    // if the distances are lower than threshold or x[idx] is empty
    if ((dist1 < threshold) || (dist2 < threshold) || (x[idx] == 0)) {
        outArr[idx] = -1; // write 0 in the output array
    }
    else{
        if (linAxis == 0) outArr[idx] = abs(x[idxA] - x[idx]);
        if (linAxis == 1) outArr[idx] = abs(y[idxA] - y[idx]);
    }
    }

__global__ void triplet( float * x, float * y , int * triplet , float * vec, float threshold) {
    /*
    PARAMETERS
     - x, y : pointer to arrays of size (nevents * MAXHITS)
          The arrays of the coordinates of all the events.
     - triplet : pointer to array of size (12 * nevents)
          Array of triplet, this is the output of the function.
     - vec : pointer to array of size (4 * nevents * MAXHITS)
          Auxiliar array for evaluate the distance and other stuff.
     - threshold : float value
          Threshold on the distance of point in the same triplet.
    */
    // Distribute jobs to blocks and threads (remember that each block takes only a triplet)
    unsigned int eventIdx = blockIdx.x / 4; // The actual number of block / is the index of the actual event
    unsigned int internaltripletIdx = blockIdx.x % 4; // % is the rest of the integer division, it tell the
                                                      // triplet number in the actual event (0,1,2,3)
    // each event has 4 triplets, each triplet has 3 number so each event has 12 number for his triplets.
    // The array triplet has length 12 * nevents, so in order to give indexs to this array we start from
    // 12 * eventsIdx, 3 * internaltripletIdx is the starting point of the actual triplet
    // Example of triplets array:
    //           <-----------------evt0------------------------> <----------evt1---------------> ... etc...
    // triplet = [idx0, idx1, idx2, idx3, idx4, idx5,..., idx11, idx12, idx13, idx14, ..., idx23 ... , idx(12*nevents - 1)]
    //           |     BLOCK0    ||      BLOCK1      |           |      BLOCK4      |
    //           <-----trip0----->  <-----trip1---->   ..2..3..  <-------trip0------>
    unsigned int tripletIdx = 12 * eventIdx + 3 * internaltripletIdx; // index where the actual triplet start
    // shared mean that all the thread of the block has to see this variable
    __shared__ int Axis; // 0 or 1: 0 for XMAX and XMIN, 1 for YMAX and YMIN
    if (internaltripletIdx == 0){ // Stuff for triplet of XMAX: first triplet of each events
    // &x[eventIdx * MAXHITS] is the starting point of the x array for the event 'eventIdx', we use the &
    // for exclude the previous points of this array and pass only the following points. The last argument off
    // the function VecMax is the lenght of the array &x[eventIdx * MAXHITS], this tell the function where to stop the search of
    // the maximum. We write the sum in triplet[tripletIdx]
      VecMax(&x[eventIdx * MAXHITS], &triplet[tripletIdx], MAXHITS);__syncthreads();
      Axis = 0; // we are processing XMAX so Axis = 0
    }
    //Same for all the others
    if (internaltripletIdx == 1){ // Stuff for triplet of XMIN
      VecMin(&x[eventIdx * MAXHITS], &triplet[tripletIdx], MAXHITS);__syncthreads();
      Axis = 0; // we are processing XMIN so Axis = 0
    }
    if (internaltripletIdx == 2){ // Stuff for triplet of YMAX
      VecMax(&y[eventIdx * MAXHITS], &triplet[tripletIdx], MAXHITS);__syncthreads();
      Axis = 1; // we are processing YMAX so Axis = 1
    }
    if (internaltripletIdx == 3){ // Stuff for triplet of YMIN
      VecMin(&y[eventIdx * MAXHITS], &triplet[tripletIdx], MAXHITS);__syncthreads();
      Axis = 1; // we are processing YMIN so Axis = 1
    }
    /*
    Evaluate the linear distance from the point indexed by triplet[tripletIdx].
    At the same time put -1 instead of linear distance if a point doesn't respect
    the threshold on the euclidean distance.
    */
    // In this case the euclidean distance is evaluate two time for the same point
    // (we have only the point A) so we pass two time the same index.
    LinDist(triplet[tripletIdx], triplet[tripletIdx], Axis,
            &x[eventIdx * MAXHITS], &y[eventIdx * MAXHITS],
            &vec[MAXHITS * blockIdx.x], threshold, MAXHITS); // in &vec[MAXHITS * blockIdx.x] we put the output
                                                             // This is different for every single block...
    // Find the minimum in the vector of linear distances and write it in triplet[tripletIdx + 1]
    // this is the index of the point B: the second point of the triplet.
    VecMin(&vec[MAXHITS * blockIdx.x], &triplet[tripletIdx + 1], MAXHITS);__syncthreads();
    // Evaluate again the linear distance but this time with two threshold:
    // - on the euclidean distance from A and from B
    LinDist(triplet[tripletIdx], triplet[tripletIdx + 1], Axis,
            &x[eventIdx * MAXHITS], &y[eventIdx * MAXHITS],
            &vec[MAXHITS * blockIdx.x], threshold, MAXHITS);
    // finally find the minimum of the new vector of linear distance
    // (is always &vec[MAXHITS * blockIdx.x] but this time is evaluate with threshold on A and B)
    // write the results in &triplet[tripletIdx + 2]: the index of the third point of the triplet C
    VecMin(&vec[MAXHITS * blockIdx.x], &triplet[tripletIdx + 2], MAXHITS);__syncthreads();

    if (threadIdx.x < 3){ // The three indices founded before are referred to the single event, we
                          // need to refer them to the "global" arrays x and y, so we add eventIdx * MAXHITS;
                          // to each element founded.
        triplet[tripletIdx + threadIdx.x] += eventIdx * MAXHITS;
    }
    int aux;
    if (threadIdx.x == 0){ // in the end we reorder the array sequentially (in the thread 0)
        if (Axis == 0){
            if (y[triplet[tripletIdx]] > y[triplet[tripletIdx + 1]]){
                aux = triplet[tripletIdx];
                triplet[tripletIdx] = triplet[tripletIdx + 1];
                triplet[tripletIdx + 1] = aux;
            }
            if (y[triplet[tripletIdx]] > y[triplet[tripletIdx + 2]]){
                aux = triplet[tripletIdx];
                triplet[tripletIdx] = triplet[tripletIdx + 2];
                triplet[tripletIdx + 2] = aux;
            }
            if (y[triplet[tripletIdx + 1]] > y[triplet[tripletIdx + 2]]){
                aux = triplet[tripletIdx + 1];
                triplet[tripletIdx + 1] = triplet[tripletIdx + 2];
                triplet[tripletIdx + 2] = aux;
            }
        }
        if (Axis == 1){
            if (x[triplet[tripletIdx]] > x[triplet[tripletIdx + 1]]){
                aux = triplet[tripletIdx];
                triplet[tripletIdx] = triplet[tripletIdx + 1];
                triplet[tripletIdx + 1] = aux;
            }
            if (x[triplet[tripletIdx]] > x[triplet[tripletIdx + 2]]){
                aux = triplet[tripletIdx];
                triplet[tripletIdx] = triplet[tripletIdx + 2];
                triplet[tripletIdx + 2] = aux;
            }
            if (x[triplet[tripletIdx + 1]] > x[triplet[tripletIdx + 2]]){
                aux = triplet[tripletIdx + 1];
                triplet[tripletIdx + 1] = triplet[tripletIdx + 2];
                triplet[tripletIdx + 2] = aux;
            }
        }
    }
    }
}

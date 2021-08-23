# include <stdio.h>
# include <cuda_runtime.h>

// const
#define MAXHITS (int) 64
#define MAX_ITERATIONS (int) 5

// CUDA Kernel
extern "C" {
__device__ int sumCommSingleWarp(volatile float* shArr) {
    int idx = threadIdx.x % warpSize; //the lane index in the warp
    if (idx<16) {
      // This operation are execute on all idx at the same time 
      // just like the classical figure of reduction
      shArr[idx] += shArr[idx+16];
      shArr[idx] += shArr[idx+8];
      shArr[idx] += shArr[idx+4];
      shArr[idx] += shArr[idx+2];
      shArr[idx] += shArr[idx+1];
    }
    return shArr[0];
}

__device__ void VecSum(const float *a, float *out, float *r, int arraySize) {
    int idx = threadIdx.x; // Index of the thread used
    
    r[idx] = a[idx];

    if (arraySize <= warpSize){
      sumCommSingleWarp( r );
      *out = r[0];
    }
    else {
      sumCommSingleWarp( &r[idx & ~(warpSize - 1)] );
      __syncthreads();
      if (idx<warpSize) { //first warp only
        r[idx] = idx * warpSize < MAXHITS ? r[idx*warpSize] : 0;
        sumCommSingleWarp(r);
        if (idx == 0)
            *out = r[0];
      }
    }
}

__device__ void Taubin (float * x, float * y , // coordinates
                        float * xm, float * ym, // mean of coordinates
                        float * u, float * v, float * z, // auxiliar vectors
                        float * u2, float * v2,  float * z2, // (aux vects)^2
                        // linear combination of aux vectors
                        float * uz, float * vz, float * uv, 
                        // mean of auxiliar vectors and their coordinates
                        float * zav, float * z2av, float * u2av, float * v2av, 
                        float * uvav, float * uzav, float * vzav,
                        int lenght, int nevents, // lenght of coordinates arrays
                        // Global memory : results
			            float * xCenter, float * yCenter, float * radius) 
{ 
  int actualsize = 4 * nevents;                  // do you meant this?? :(

  // Distribute jobs to blocks and threads
  unsigned int eventIdx = blockIdx.x / 4;
  unsigned int tripletIdx = blockIdx.x % 4;
  unsigned int beginIdx = 4 * MAXHITS * eventIdx + MAXHITS * tripletIdx ;
  unsigned int hitIdx = 4 * MAXHITS * eventIdx + MAXHITS * tripletIdx + threadIdx.x;

  __shared__ float s_reduction[MAXHITS];
  
  // Execute function only on busy blocks
  if( eventIdx < actualsize )
  {
    // Compute center of gravity
    VecSum (&x[ beginIdx ] , &xm[ blockIdx.x], s_reduction , MAXHITS );
    VecSum (&y[ beginIdx ] , &ym[ blockIdx.x], s_reduction , MAXHITS );
   
    // Erase utility arrays
    uzav [4* eventIdx + tripletIdx ] = 0.f;
    z2av [4* eventIdx + tripletIdx ] = 0.f;
    u2av [4* eventIdx + tripletIdx ] = 0.f;
    v2av [4* eventIdx + tripletIdx ] = 0.f;
    uvav [4* eventIdx + tripletIdx ] = 0.f;
    uzav [4* eventIdx + tripletIdx ] = 0.f;
    vzav [4* eventIdx + tripletIdx ] = 0.f;

    // If current hit belongs to triplet fill utility arrays
    if( x[ hitIdx ] != 0 && y[ hitIdx ] !=0 )
    {
      u[ hitIdx ]  = x[ hitIdx ] - xm[ blockIdx.x] / lenght ;
      v[ hitIdx ]  = y[ hitIdx ] - ym[ blockIdx.x] / lenght ;
      u2[ hitIdx ] = u[ hitIdx ]  * u[ hitIdx ];
      v2[ hitIdx ] = v[ hitIdx ]  * v[ hitIdx ];
      uv[ hitIdx ] = u[ hitIdx ]  * v[ hitIdx ];
      z[ hitIdx ]  = u2[ hitIdx ] + v2[ hitIdx ];
      z2[ hitIdx ] = z[ hitIdx ]  * z[ hitIdx ];
      uz[ hitIdx ] = u[ hitIdx ]  * z[ hitIdx ];
      vz[ hitIdx ] = v[ hitIdx ]  * z[ hitIdx ];
    }
    // Compute sum of arrays
    VecSum (&u2[ beginIdx ], &u2av [ blockIdx.x ], s_reduction , MAXHITS );
    VecSum (&v2[ beginIdx ], &v2av [ blockIdx.x ], s_reduction , MAXHITS );
    VecSum (&uv[ beginIdx ], &uvav [ blockIdx.x ], s_reduction , MAXHITS );
    VecSum (&z [ beginIdx ], &zav  [ blockIdx.x ], s_reduction , MAXHITS );
    VecSum (&z2[ beginIdx ], &z2av [ blockIdx.x ], s_reduction , MAXHITS );
    VecSum (&uz[ beginIdx ], &uzav [ blockIdx.x ], s_reduction , MAXHITS );
    VecSum (&vz[ beginIdx ], &vzav [ blockIdx.x ], s_reduction , MAXHITS );

    // The following code is executed once per block
    if( threadIdx.x == 0)
    {
      // Compute average
      float zav0  = zav [ blockIdx.x] / lenght ;
      float z2av0 = z2av[ blockIdx.x] / lenght ;
      float u2av0 = u2av[ blockIdx.x] / lenght ;
      float v2av0 = v2av[ blockIdx.x] / lenght ;
      float uvav0 = uvav[ blockIdx.x] / lenght ;
      float uzav0 = uzav[ blockIdx.x] / lenght ;
      float vzav0 = vzav[ blockIdx.x] / lenght ;
      // Characteristic polynomial coefficients
      float CovXY = u2av0 * v2av0 - uvav0 * uvav0 ;
      float VarZ = z2av0 - zav0 * zav0;
      float c0 = uzav0 *( uzav0 * v2av0 - vzav0 * uvav0 ) + vzav0 *( vzav0 * u2av0 - uzav0 * uvav0 ) - VarZ * CovXY ;
      float c1 = VarZ * zav0 + 4* CovXY * zav0 - uzav0 * uzav0 - vzav0 * vzav0 ;
      float c2 = -3* zav0 * zav0 - z2av0 ;
      float c3 = 4* zav0 ;
      float c22 = c2 *2;
      float c33 = c3 *3;
      // Find the roots of the characteristic polynomial
      //P( eta ) = c0 + c1* eta + c2* eta ^2 + c3*eta ^3 = 0
      float eta , eta_new , poly , poly_new , derivative ;
      eta = 0.f;
      poly = c0;
      // Newton 's method
      for ( int i = 0; i < MAX_ITERATIONS ; ++i ){
        derivative = c1 + eta *( c22 + eta* c33 );
        eta_new = eta - poly / derivative ;
        if( eta_new == eta || isnan ( eta_new ) ) break ;
        poly_new = c0 + eta_new *( c1 + eta_new *( c2 + eta_new *c3));
        if( abs ( poly_new ) >= abs ( poly ) ) break ;
        eta = eta_new ;
        poly = poly_new ;
      }
      // Compute ring parameters
      float det = eta * eta - eta * zav0 + CovXY ;
      float uc = ( uzav0 *( v2av0 - eta ) - vzav0 * uvav0 ) / (2* det );
      float vc = ( vzav0 *( u2av0 - eta ) - uzav0 * uvav0 ) / (2* det );
      float alpha = uc*uc + vc*vc + zav0 ;
      // Converted output result
      radius [ blockIdx.x] = sqrtf ( alpha ) ;
      xCenter[ blockIdx.x] = uc + xm[ blockIdx.x] / lenght;
      yCenter[ blockIdx.x] = vc + ym[blockIdx.x]  / lenght;
    }
  }
}

__device__ inline int Ptolemy ( const float& x , const float& y , 
                               const float * triplet )
{
    float AB = sqrt( pow(triplet[0] - triplet[2], 2) + pow(triplet[1] - triplet[3], 2));
    float CD = sqrt( pow(triplet[4] - x, 2) + pow(triplet[5] - y, 2));
    float AD = sqrt( pow(triplet[0] - x, 2) + pow(triplet[1] - y, 2));
    float BC = sqrt( pow(triplet[2] - triplet[4], 2) + pow(triplet[3] - triplet[5], 2));
    float AC = sqrt( pow(triplet[0] - triplet[4], 2) + pow(triplet[1] - triplet[5], 2));
    float BD = sqrt( pow(triplet[2] - x, 2) + pow(triplet[3] - y, 2));
    float ptoval = AB * CD + AD * BC - AC * BD;
    if (ptoval < 0.2) return 1;
    else return 0;
}

__global__ void multiring( float * x, float * y , int * triplet, // coordinates and triplets
                           float * x_candidate, float * y_candidate, // what pass after ptolemy
                           float * xm, float * ym, // mean of coordinates
                           float * u, float * v, float * z, // auxiliar vectors
                           float * u2, float * v2,  float * z2, // (aux vects)^2
                           // linear combination of aux vectors
                           float * uz, float * vz, float * uv, 
                           // mean of auxiliar vectors and their coordinates
                           float * zav, float * z2av, float * u2av, float * v2av, 
                           float * uvav, float * uzav, float * vzav, int nevents,
                           // Global memory : results
			               float * xCenter, float * yCenter, float * radius )
{
    // Allocate block shared memory
    __shared__ float triplet_s [6];
    __shared__ unsigned int length;
 
    // Distribute jobs to blocks and threads
    unsigned int eventIdx = blockIdx.x / 4; // 4 blocks for each event
    unsigned int tripledIdx = blockIdx.x % 4;
    unsigned int hitIdx = threadIdx.x + eventIdx * MAXHITS; // idx of all hits of all events, change every 4 blocks
    unsigned int utilsIdx = MAXHITS * blockIdx.x + threadIdx.x; // change every block
    unsigned int elemIdx = 12 * eventIdx + threadIdx.x + 3 * tripledIdx;

    // Copy the current triplet in shared memory
    if ( threadIdx.x < 3) {
        triplet_s [2* threadIdx.x] = x[ triplet[ elemIdx ]];
        triplet_s [2* threadIdx.x + 1] = y[ triplet[ elemIdx ]];
    }__syncthreads ();
 
    // Pattern recognition : copy ' good ' hits
    if ( Ptolemy ( x[ hitIdx ] , y[ hitIdx ], triplet_s ) && ( x [ hitIdx ] != 0) && ( y[ hitIdx ] != 0) ) {
        x_candidate [ utilsIdx ] = x[ hitIdx ];
        y_candidate [ utilsIdx ] = y[ hitIdx ];
    } else {
        x_candidate [ utilsIdx ] = 0;
        y_candidate [ utilsIdx ] = 0;
    }
    __syncthreads ();
 
    // Compute number of hit candidates per ring
    length = 0;
    if ( x_candidate [ utilsIdx ] != 0) atomicAdd (& length , 1);__syncthreads() ;
    // Execute ring fit
    if (length < MAXHITS/8){    //there are less than 8 points (not enough accurate fit)
        xCenter = 0;
        yCenter = 0;
        radius = 0;
    }
    else {
        Taubin (x_candidate, y_candidate,
                xm,   ym, 
                u,    v,    z, 
                u2,   v2,   z2,
                uz,   vz,   uv, 
                zav,  z2av, u2av,  v2av, 
                uvav, uzav, vzav,
                length, nevents, 
                xCenter, yCenter, radius);}
}
}
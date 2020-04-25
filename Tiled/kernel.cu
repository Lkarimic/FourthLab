/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x, by = blockIdx.y,
       tx = threadIdx.x, ty = threadIdx.y;
      int Row = by * TILE_WIDTH + ty;
      int Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;
    int width  = numAColumns > numBRows ? numAColumns : numBRows;

    for (int m = 0; m < ceil((float) width / TILE_WIDTH); ++m) {
       if (Row < numARows && m*TILE_WIDTH+tx < numAColumns)
          ds_A[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
       else
          ds_A[ty][tx] = 0;
       if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows)
          ds_B[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
       else
          ds_B[ty][tx] = 0;

       __syncthreads();
       for (int k = 0; k < TILE_WIDTH; ++k)
          Pvalue += ds_A[ty][k] * ds_B[k][tx];
       __syncthreads();
     }
     if (Row < numCRows && Col < numCColumns)
    C[Row*numCColumns+Col] = Pvalue;
}


    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     ********************************************************************/



void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------
    const unsigned int BLOCK_SIZE = 16; // Use 16x16 thread block
    dim3 dimGrid(ceil(n/(float)BLOCK_SIZE),(ceil(m/(float)BLOCK_SIZE)), 1 );
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);


    //INSERT CODE HERE to define thread blocks and layout

    // Invoke CUDA kernel -----------------------------------------------------
mysgemm<<<dimGrid, dimBlock>>>(m, n, k, A, B, C);
    //INSERT CODE HERE



}

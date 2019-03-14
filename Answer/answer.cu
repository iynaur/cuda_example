#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include "answer.cuh"

// Compute vector sum C = A+B
//CUDA kernel. Each thread performes one pair-wise addition

__global__ void vector_add(float *a, float *b, float *c)
{
    /* insert code to calculate the index properly using blockIdx.x, blockDim.x, threadIdx.x */
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	c[index] = a[index] + b[index];
}

/* experiment with N */
/* how large can it be? */
#define THREADS_PER_BLOCK 1000

void add(float *a, float *b, float *c, int N)
{
	float *d_a, *d_b, *d_c;
	int size = N * sizeof( float );
	/* allocate space for device copies of a, b, c */

	cudaMalloc( (void **) &d_a, size );
	cudaMalloc( (void **) &d_b, size );
	cudaMalloc( (void **) &d_c, size );

	/* copy inputs to device */
	/* fix the parameters needed to copy data to the device */
	cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice );
	cudaMemcpy( d_b, b, size, cudaMemcpyHostToDevice );

	/* launch the kernel on the GPU */
	/* insert the launch parameters to launch the kernel properly using blocks and threads */ 
    vector_add<<< (N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( d_a, d_b, d_c );

    //Synchronize threads
    cudaThreadSynchronize();

	/* copy result back to host */
	/* fix the parameters needed to copy data back to the host */
	cudaMemcpy( c, d_c, size, cudaMemcpyDeviceToHost );


	/* clean up */

	cudaFree( d_a );
	cudaFree( d_b );
	cudaFree( d_c );
	
    return;
}

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

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
#define N (100000)
#define THREADS_PER_BLOCK 1000

int main()
{
    float *a, *b, *c;
	float *d_a, *d_b, *d_c;
	int size = N * sizeof( float );
	/* allocate space for device copies of a, b, c */
	/* allocate space for host copies of a, b, c and setup input values */

//Allocate memory for each vector on host
	a = (float *)malloc( size );
	b = (float *)malloc( size );
	c = (float *)malloc( size );

	for( int i = 0; i < N; i++ )
	{
		a[i] = b[i] = i;
		c[i] = 0;
	}

	cudaMalloc( (void **) &d_a, size );
	cudaMalloc( (void **) &d_b, size );
	cudaMalloc( (void **) &d_c, size );

	/* copy inputs to device */
	/* fix the parameters needed to copy data to the device */
	cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice );
	cudaMemcpy( d_b, b, size, cudaMemcpyHostToDevice );

	/* launch the kernel on the GPU */
	/* insert the launch parameters to launch the kernel properly using blocks and threads */ 
	vector_add<<< 100, 1000 >>>( d_a, d_b, d_c );

//Synchronize threads
cudaThreadSynchronize();

	/* copy result back to host */
	/* fix the parameters needed to copy data back to the host */
	cudaMemcpy( c, d_c, size, cudaMemcpyDeviceToHost );


	printf( "c[0] = %f\n",c[0] );
	printf( "c[%d] = %f\n",N-1, c[N-1] );

	/* clean up */

	free(a);
	free(b);
	free(c);
	cudaFree( d_a );
	cudaFree( d_b );
	cudaFree( d_c );
	
	return 0;
} /* end main */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

// Compute vector sum C = A+B
//CUDA kernel. Each thread performes one pair-wise addition

__global__ void vecAddKernel(float *A, float *B, float *C, int n)
{
//Get our global thread ID
int i = threadIdx.x;

if (i<n) C[i] = A[i] + B[i];
}

int main(int argc, char* argv[])
{

//Size of vectors
int n = 100;

int size = n * sizeof(float);


//Host input vectors
float *h_A, *h_B;
//Host output vector
float *h_C;

//Device input vectors
float *d_A, *d_B;
//Device output vector
float *d_C;

//Allocate memory for each vector on host
h_A = (float*)malloc((size));
h_B = (float*)malloc((size));
h_C = (float*)malloc((size));

for (int i=0; i<n; ++i) h_A[i]=h_B[i]=i;

//Allocate memory for each vector on GPU
cudaMalloc( (void **) &d_A, size);
cudaMalloc( (void **) &d_B, size);
cudaMalloc( (void **) &d_C, size);

//Copy host vectors to device
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);



//Number of threads in each block

vecAddKernel<<<1,100>>>(d_A, d_B, d_C, n);

//Synchronize threads
cudaThreadSynchronize();

//Copy array back to host
cudaMemcpy( h_C, d_C, size, cudaMemcpyDeviceToHost );

for (int i=0; i<n; ++i) {
	printf( "c[%d] = %f\n",i, h_C[i] );
	printf( "a[%d] = %f\n",i, h_A[i] );
}


//Release device memory
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);

//Release host memory
free(h_A);
free(h_B);
free(h_C);

return 0;
}

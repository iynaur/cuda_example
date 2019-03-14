# cuda_example

Credit:
Mark Harris - NVIDIA Corporation

Description:
A simple kernel which adds two vectors (1D arrays) with the GPU. This is a
good example showing off memory allocation and movement use the CUDA C 
runtime API, while using a very simple kernel function. There are many 
variations you can use on this example, including using the Thrust library
to handle memory allocation and movement instead of cudaMalloc and cudaMemcpy
explicitly.

Files:
  Exercise:
    exercise.cu - one version of a hands-on lab exercise
  Answer:
    answer.cu - the solution to the presented exercise

Compile:
Using the NVIDIA C Compiler
  nvcc -o vectoradd answer.cu

Flow:
1. After having introduced the concepts of cudaMemcpy, cudaMalloc and block
and threads, have the students modify the code in exercise.cu to make it
compile and do the correct work on the GPU.

References:
None

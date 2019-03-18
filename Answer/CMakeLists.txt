cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(cmake_and_cuda )

FIND_PACKAGE(CUDA REQUIRED)
# C++和CUDA的编译参数，可选。
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;-std=c++11;)
# 头文件路径，按需

#include(CTest)

#add_library(particles SHARED
#  answer.cu
#  )

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
#target_compile_features(particles PUBLIC cxx_std_11)
 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
#set_target_properties( particles
#                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

CUDA_ADD_EXECUTABLE(vecadd  main.cpp answer.cu)

set_property(TARGET vecadd
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(vecadd PRIVATE particles)



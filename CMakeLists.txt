cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
Project(CUDA_C CXX C CUDA)
set(CMAKE_CUDA_FLAGS "-arch=compute_75 -g -G -O3")
include_directories(cuda_util)

add_subdirectory(1_hello_world)
add_subdirectory(2_add_vector)
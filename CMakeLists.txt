cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
Project(CUDA_C CXX C CUDA)
set(CMAKE_CUDA_FLAGS "-arch=compute_75 -g -G -O3")

add_subdirectory(1_hello_world)
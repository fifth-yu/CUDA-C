#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <stdlib.h>
#include<time.h>
#include<stdint.h>
#include<stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CUDA_CHECK(call)\
{\
    const cudaError_t error=call;\
    if(error!=cudaSuccess) {\
        printf("ERROR: %s:%d,",__FILE__,__LINE__);\
        printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
        exit(1);\
    }\
}

struct timespec CurrentTime();
void InitVector(float* vector, int size);
uint64_t ComputeTime(struct timespec* start, struct timespec* end);

#ifdef __cplusplus
}
#endif

#endif
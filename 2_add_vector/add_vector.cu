#include "cuda_util.h"
#include "stdlib.h"

void AddVectorOnCpu(float *vec1, float *vec2, float *retVec, int size)
{
    for (int i = 0; i < size; i++) {
       retVec[i] = vec1[i] + vec2[i];
    }

    return;
}

__global__ void AddVectorOnGpu(float *vec1, float *vec2, float *retVec, int size)
{
    int id = threadIdx.x;
    if (id < size) {
        retVec[id] = vec1[id] + vec2[id];
    }
}

void PrepareData(float **vec1, float **vec2, float **retVec, int vecSize)
{
    *vec1 = (float *)malloc(sizeof(float) * vecSize);
    if (vec1 == NULL) {
        printf("Error: Failed to malloc vec1\n");
        return;
    }
    *vec2 = (float *)malloc(sizeof(float) * vecSize);
    if (vec2 == NULL) {
        printf("Error: Failed to malloc vec2\n");
        free(*vec2);
        return;
    }
    *retVec = (float *)malloc(sizeof(float) * vecSize);
    if (retVec == NULL) {
        printf("Error: Failed to malloc retVec\n");
        free(*vec1);
        free(*vec2);
        return;
    }
    InitVector(*vec1, vecSize);
    InitVector(*vec2, vecSize);
    InitVector(*retVec, vecSize);
}

void CompareResult(float *retVecCpu, float *retVecGpu, int vecSize)
{
    for (int i = 0; i < vecSize; i++) {
        if (abs(retVecCpu[i] - retVecGpu[i]) > retVecCpu[i] * 1e-4) {
            printf("Error: the computing result is not inconsistent.\n");
            return;
        }
    }
    printf("Info: the computing result is inconsistent.\n");
}

int main(int argc, const char **argv)
{
    if (argc < 2) {
        printf("Error: Please input vector size\n");
        return 1;
    }
    int vecSize = atoi(argv[1]);
    if (vecSize <= 0) {
        printf("Error: the vector size is less than 0, size (%d), argv (%s).\n", vecSize, argv[1]);
        return 1;
    }
    // 准备数据
    float *vec1 = NULL;
    float *vec2 = NULL;
    float *retVec = NULL;
    PrepareData(&vec1, &vec2, &retVec, vecSize);
    if (vec1 == NULL || vec2 == NULL || retVec == NULL) {
        printf("Error: Prepare data, vec1 (%p), vec2 (%p), retVec (%p)\n", vec1, vec2, retVec);
        return 1;
    }
    // 测试CPU
    struct timespec startCpu = CurrentTime();
    AddVectorOnCpu(vec1, vec2, retVec, vecSize);
    struct timespec endCpu = CurrentTime();
    printf("Info: Succeed to add vector on cpu.\n");
    // 测试GPU
    float *deviceVec1 = NULL;
    float *deviceVec2 = NULL;
    float *deviceRetVec = NULL;
    CUDA_CHECK(cudaMalloc(&deviceVec1, sizeof(float) * vecSize));
    CUDA_CHECK(cudaMalloc(&deviceVec2, sizeof(float) * vecSize));
    CUDA_CHECK(cudaMalloc(&deviceRetVec, sizeof(float) * vecSize));
    CUDA_CHECK(cudaMemcpy(deviceVec1, vec1, sizeof(float) * vecSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceVec2, vec1, sizeof(float) * vecSize, cudaMemcpyHostToDevice));
    
    float *hostRetVec = (float *)malloc(sizeof(float) * vecSize);
    if (hostRetVec == NULL) {
        printf("Error: failed to malloc hostRetVec.\n");
        return 1;
    }
    dim3 block(vecSize);
    dim3 grid(vecSize / block.x);
    struct timespec startGpu = CurrentTime();
    AddVectorOnGpu<<<grid, block>>>(deviceVec1, deviceVec2, deviceRetVec, vecSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    // CUDA_CHECK(cudaMemcpy(hostRetVec, deviceRetVec, sizeof(float) * vecSize, cudaMemcpyDeviceToHost));
    struct timespec endGpu = CurrentTime();
    CUDA_CHECK(cudaMemcpy(hostRetVec, deviceRetVec, sizeof(float) * vecSize, cudaMemcpyDeviceToHost));
    printf("Info: Succeed to add vector on gpu.\n");
    // 对比测试性能
    printf("-----------------------------------------------------\n");
    printf(" type        vecSize       spendTime(ns)\n");
    printf("-----------------------------------------------------\n");
    printf("  %s           %d          %lu\n", "CPU", vecSize, ComputeTime(&startCpu, &endCpu));
    printf("  %s           %d          %lu\n", "GPU", vecSize, ComputeTime(&startGpu, &endGpu));
    printf("-----------------------------------------------------\n");
    // 对比计算结果
    CompareResult(retVec, hostRetVec, vecSize);
    // 释放内存
    free(vec1);
    free(vec2);
    free(retVec);
    free(hostRetVec);
    cudaFree(deviceVec1);
    cudaFree(deviceVec2);
    cudaFree(deviceRetVec);
    return 0;
}
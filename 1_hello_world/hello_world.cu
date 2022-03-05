#include <stdio.h>

__global__ void HelloWorld()
{
	printf("GPU: Hello world!\n");
}

int main()
{
	printf("CPU: Hello world!\n");
	HelloWorld<<<1, 10>>>();
	cudaDeviceReset();
	return 0;
}

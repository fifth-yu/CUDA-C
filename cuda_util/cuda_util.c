
#include "cuda_util.h"
#include<time.h>
#include<stdint.h>

#define SECOND_TO_NSECOND 1e9

struct timespec CurrentTime()
{
	struct timespec time = { 0 };
	int ret = clock_gettime(CLOCK_MONOTONIC, &time);
	if (ret != 0) {
		printf("Error: Failed to get system time, ret (%d).", ret);
		return time;
	}
	return time;
}

uint64_t ComputeTime(struct timespec* start, struct timespec* end)
{
	struct timespec time = { 0 };
	if (start->tv_nsec < end->tv_nsec) {
		time.tv_nsec = end->tv_nsec + SECOND_TO_NSECOND - start->tv_nsec;
		end->tv_sec -= 1;
	} else {
		time.tv_nsec = end->tv_nsec - start->tv_nsec;
	}
	time.tv_sec = end->tv_sec - start->tv_sec;

	return (uint64_t)((uint64_t)(time.tv_sec * SECOND_TO_NSECOND) + (uint64_t)time.tv_nsec);
}

void InitVector(float* vector, int size)
{
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < size; i++)
	{
		vector[i] = (float)(rand() & 0xffff) / 1000.0f;
	}
}
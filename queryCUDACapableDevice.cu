#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#define CUDA_CHECK(call)												\
	do{																	\
		cudaError_t err = call;                                         \
		if (err != cudaSuccess){										\
			fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));       \
			abort();													\
		}                                                               \
	}                                                                   \
	while (0)

#define HOST_CHECK(cond, msg)                             \
    do {                                                  \
        if (!(cond)) {                                    \
            fprintf(stderr, "Host error: %s\n", (msg));   \
            abort();			                          \
        }                                                 \
    } while (0)

static cudaDeviceProp queryCUDACapableDevices(int device_num)
{
	cudaDeviceProp info = { 0 };

	int nDevices = 0;
	CUDA_CHECK(cudaGetDeviceCount(&nDevices));
	HOST_CHECK(nDevices > 0, "No CUDA capable devices detected!");
	HOST_CHECK(device_num >= 0 && device_num < nDevices, "Invalid device number!");

	CUDA_CHECK(cudaGetDeviceProperties(&info, device_num));

	printf("Selected device: %s\n", info.name);
	printf("  Streaming MultiProcessor count: %i\n", info.multiProcessorCount);
	printf("  Warp-size: %i\n", info.warpSize);
	printf("  Max thread dim (x, y, z): %i %i %i\n", info.maxThreadsDim[0], info.maxThreadsDim[1], info.maxThreadsDim[2]);
	printf("  Max threads per block: %i\n", info.maxThreadsPerBlock);
	printf("  Max threads per multi processor: %i\n", info.maxThreadsPerMultiProcessor);
	printf("  Max blocks per multi processor: %i\n", info.maxBlocksPerMultiProcessor);
	printf("  Memory Clock Rate (MHz): %i\n", info.memoryClockRate / 1024);
	printf("  Memory Bus Width (bits): %i\n", info.memoryBusWidth);
	printf("  Global memory (GBytes): %.1f\n", (float)(info.totalGlobalMem) / 1024.0f / 1024.0f / 1024.0f);
	printf("  Shared memory per block (KBytes): %.1f\n", (float)(info.sharedMemPerBlock) / 1024.0f);
	printf("  Constant memory (KBytes): %.1f\n", (float)(info.totalConstMem) / 1024.0f);
	printf("  Compute capability (major.minor): %i.%i\n", info.major, info.minor);
	printf("  Concurrent kernels: %s\n", info.concurrentKernels ? "yes" : "no");
	printf("  Concurrent computation/communication: %s\n", info.deviceOverlap ? "yes" : "no");

	return info;
}

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		fprintf(stderr, "Usage: %s <device_number>\n", argv[0]);
		return EXIT_FAILURE;
	}

	char* endptr;
	errno = 0;
	long dev_num = strtol(argv[1], &endptr, 10);

	if (errno != 0 || *endptr != '\0' || dev_num < 0)
	{
		fprintf(stderr, "Invalid device number: %s\n", argv[1]);
		return EXIT_FAILURE;
	}

	queryCUDACapableDevices((int)dev_num);
	return EXIT_SUCCESS;
}
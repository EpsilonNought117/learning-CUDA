#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda.h>
#include <immintrin.h>
#include <cuda_runtime.h>

// Error checking macros
#define CUDA_CHECK(call)				\
	do {								\
		cudaError_t err = call;			\
		if (err != cudaSuccess) {		\
			fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
			abort();					\
		}								\
	} while (0)

#define HOST_CHECK(cond, msg)			\
	do {								\
		if (!(cond)) {					\
			fprintf(stderr, "Host error: %s\n", (msg)); \
			abort();					\
		}								\
	} while (0)

// Constants
#define MIN_LOGIT		-50.0f
#define MAX_LOGIT		50.0f
#define COMBINED_MAX	((1UL << 30) - 1)
#define THREADS_PER_BLK	512UL
#define ELEM_COUNT      (10UL * 1000 * 1000UL)

// Utility
static inline float getRandLogitInRange(float maxVal, float minVal)
{
	float combined = (float)(((unsigned int)rand() << 15) | rand());

	// MSVC's rand() returns a value between 0 and (2^15 - 1)

	float normalized = combined / (float)COMBINED_MAX;
	return minVal + normalized * (maxVal - minVal);
}

// Host CPU softmax implementation
float findMax_h(float* arr, unsigned int size) 
{
	float max_val = arr[0];

	for (unsigned int i = 1; i < size; i++)
		if (arr[i] > max_val) max_val = arr[i];

	return max_val;
}

void subMax_h(float* arr, unsigned int size, float maxVal)
{
	for (unsigned int i = 0; i < size; i++)
		arr[i] -= maxVal;
}

float expAndSum_h(float* arr, unsigned int size)
{
	float sum = 0.0f;

	for (unsigned int i = 0; i < size; i++)
	{
		arr[i] = expf(arr[i]);
		sum += arr[i];
	}

	return sum;
}

void normalize_h(float* arr, unsigned int size, float sum)
{
	for (unsigned int i = 0; i < size; i++)
		arr[i] /= sum;
}

// GPU kernels
__global__ void findMax_d(float* input, unsigned int size, float* reduced)
{
	__shared__ float smem[THREADS_PER_BLK];

	unsigned int tid = threadIdx.x;
	float maxVal = -INFINITY;

	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
		maxVal = fmaxf(maxVal, input[i]);

	smem[tid] = maxVal;
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
			smem[tid] = fmaxf(smem[tid], smem[tid + s]);
	
		__syncthreads();
	}
	
	if (tid == 0)
		reduced[blockIdx.x] = smem[0];
}

__global__ void subAndExp_d(float* input, unsigned int size, float maxVal)
{
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
		input[i] = expf(input[i] - maxVal);
}

__global__ void sum_d(float* input, unsigned int size, float* reduced)
{
	__shared__ float smem[THREADS_PER_BLK];

	unsigned int tid = threadIdx.x;
	float sum = 0.0f;

	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
		sum += input[i];

	smem[tid] = sum;
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
			smem[tid] += smem[tid + s];
	
		__syncthreads();
	}
	
	if (tid == 0)
		reduced[blockIdx.x] = smem[0];
}

__global__ void normalize_d(float* input, unsigned int size, float sumVal)
{
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
		input[i] /= sumVal;
}

int main(void) {
	const unsigned int total_elements = ELEM_COUNT;
	const unsigned int threads_per_block = THREADS_PER_BLK;
	const unsigned int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

	// Allocate host memory
	float* h_input = (float*)malloc(sizeof(float) * total_elements);
	float* h_cpu_output = (float*)malloc(sizeof(float) * total_elements);
	float* h_gpu_output = (float*)malloc(sizeof(float) * total_elements);
	float* h_max_reduced = (float*)malloc(sizeof(float) * num_blocks);
	float* h_sum_reduced = (float*)malloc(sizeof(float) * num_blocks);
	HOST_CHECK(h_input && h_cpu_output && h_gpu_output && h_max_reduced && h_sum_reduced, "Host malloc failed");

	// Initialize input
	for (unsigned int i = 0; i < total_elements; i++)
		h_input[i] = getRandLogitInRange(MAX_LOGIT, MIN_LOGIT);

	memcpy(h_cpu_output, h_input, sizeof(float) * total_elements);

	// CPU softmax
	float max_cpu = findMax_h(h_cpu_output, total_elements);
	subMax_h(h_cpu_output, total_elements, max_cpu);
	float sum_cpu = expAndSum_h(h_cpu_output, total_elements);
	normalize_h(h_cpu_output, total_elements, sum_cpu);

	// GPU softmax
	float* d_input, * d_max, * d_sum;
	CUDA_CHECK(cudaMalloc(&d_input, sizeof(float) * total_elements));
	CUDA_CHECK(cudaMalloc(&d_max, sizeof(float) * num_blocks));
	CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float) * num_blocks));
	CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(float) * total_elements, cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	float elapsed_ms;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventRecord(start));

	findMax_d << <num_blocks, threads_per_block >> > (d_input, total_elements, d_max);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaMemcpy(h_max_reduced, d_max, sizeof(float) * num_blocks, cudaMemcpyDeviceToHost));

	float max_gpu = -INFINITY;
	for (unsigned int i = 0; i < num_blocks; ++i)
		if (h_max_reduced[i] > max_gpu) max_gpu = h_max_reduced[i];

	subAndExp_d << <num_blocks, threads_per_block >> > (d_input, total_elements, max_gpu);
	CUDA_CHECK(cudaGetLastError());

	sum_d << <num_blocks, threads_per_block >> > (d_input, total_elements, d_sum);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaMemcpy(h_sum_reduced, d_sum, sizeof(float) * num_blocks, cudaMemcpyDeviceToHost));

	float sum_gpu = 0.0f;
	for (unsigned int i = 0; i < num_blocks; ++i)
		sum_gpu += h_sum_reduced[i];

	normalize_d << <num_blocks, threads_per_block >> > (d_input, total_elements, sum_gpu);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaMemcpy(h_gpu_output, d_input, sizeof(float) * total_elements, cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
	printf("Softmax completed. CUDA time: %.3f ms\n", elapsed_ms);

	// Write to CSV
	FILE* file = fopen("softmax_outputs.csv", "w");
	HOST_CHECK(file != NULL, "Failed to open CSV file");
	fprintf(file, "Sno.,GPU Output,CPU Output\n");
	for (unsigned int i = 0; i < total_elements; i++)
		fprintf(file, "%u,%.15g,%.15g\n", i, h_gpu_output[i], h_cpu_output[i]);
	fclose(file);
	printf("Output written to softmax_outputs.csv\n");

	// Cleanup
	free(h_input);
	free(h_cpu_output);
	free(h_gpu_output);
	free(h_max_reduced);
	free(h_sum_reduced);
	CUDA_CHECK(cudaFree(d_input));
	CUDA_CHECK(cudaFree(d_max));
	CUDA_CHECK(cudaFree(d_sum));
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	return 0;
}
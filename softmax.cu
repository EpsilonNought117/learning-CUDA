#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Step 1: Kernel for exponentiation
__global__ void expKernel(float* input, float* output, unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        output[idx] = expf(input[idx]);
    }
}

// Step 2: Reduction using atomic operations
__global__ void reduceAtomicKernel(float* input, float* sum, unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x * 256 + threadIdx.x;
    float local_sum = 0.0f;

    // Process multiple consecutive elements with block-level stride
    for (unsigned int i = idx; i < n && i < (blockIdx.x + 1) * blockDim.x * 256; i += blockDim.x)
    {
        local_sum += input[i];
    }

    // Atomic update to global sum
    if (local_sum != 0.0f)
    {
        atomicAdd(sum, local_sum);
    }
}

// Step 3: Kernel for normalization
__global__ void normalizeKernel(float* input, float sum, float* output, unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        output[idx] = input[idx] / sum;
    }
}

// CPU softmax for verification
void softmaxCPU(float* input, float* output, unsigned int n)
{
    float sum = 0.0f;

    for (unsigned int i = 0; i < n; i++)
    {
        output[i] = expf(input[i]);
        sum += output[i];
    }

    for (unsigned int i = 0; i < n; i++)
    {
        output[i] /= sum;
    }
}

// Calculate Mean Squared Error
float calculateMSE(float* gpu_result, float* cpu_result, unsigned int n)
{
    float mse = 0.0f;
    
    for (unsigned int i = 0; i < n; i++)
    {
        float diff = gpu_result[i] - cpu_result[i];
        mse += diff * diff;
    }

    return mse / n;
}

int main() {
    const unsigned int n = 10000000; // 10 million elements
    const unsigned int threadsPerBlock = 256;
    const unsigned int expBlocks = (n + threadsPerBlock - 1) / threadsPerBlock; // For expKernel and normalizeKernel
    const unsigned int reduceBlocks = (n + threadsPerBlock * 256 - 1) / (threadsPerBlock * 256); // For reduceAtomicKernel

    // Host arrays
    float* h_input = (float*)malloc(n * sizeof(float));
    float* h_output_gpu = (float*)malloc(n * sizeof(float));
    float* h_output_cpu = (float*)malloc(n * sizeof(float));

    // Initialize input with random values
    srand(42); // Fixed seed for reproducibility
    for (unsigned int i = 0; i < n; i++) {
        h_input[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f; // Range [-5, 5]
    }

    // Device arrays
    float* d_input, * d_exp, * d_sum, * d_output;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_exp, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));

    // Initialize global sum to 0
    float zero = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_sum, &zero, sizeof(float), cudaMemcpyHostToDevice));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice));

    // Timing variables (separate events for each kernel)
    cudaEvent_t start_exp, stop_exp, start_reduce, stop_reduce, start_norm, stop_norm;
    CUDA_CHECK(cudaEventCreate(&start_exp));
    CUDA_CHECK(cudaEventCreate(&stop_exp));
    CUDA_CHECK(cudaEventCreate(&start_reduce));
    CUDA_CHECK(cudaEventCreate(&stop_reduce));
    CUDA_CHECK(cudaEventCreate(&start_norm));
    CUDA_CHECK(cudaEventCreate(&stop_norm));
    float milliseconds;

    // Step 1: Exponentiation
    CUDA_CHECK(cudaEventRecord(start_exp, 0));
    expKernel << <expBlocks, threadsPerBlock >> > (d_input, d_exp, n);
    CUDA_CHECK(cudaEventRecord(stop_exp, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_exp));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_exp, stop_exp));
    printf("Exponentiation kernel time: %.3f ms\n", milliseconds);

    // Step 2: Atomic reduction
    CUDA_CHECK(cudaEventRecord(start_reduce, 0));
    reduceAtomicKernel << <reduceBlocks, threadsPerBlock >> > (d_exp, d_sum, n);
    CUDA_CHECK(cudaEventRecord(stop_reduce, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_reduce));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_reduce, stop_reduce));
    printf("Reduction kernel time: %.3f ms\n", milliseconds);

    // Copy sum to host
    float h_sum;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

    // Step 3: Normalization
    CUDA_CHECK(cudaEventRecord(start_norm, 0));
    normalizeKernel << <expBlocks, threadsPerBlock >> > (d_exp, h_sum, d_output, n);
    CUDA_CHECK(cudaEventRecord(stop_norm, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_norm));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_norm, stop_norm));
    printf("Normalization kernel time: %.3f ms\n", milliseconds);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, n * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU verification
    softmaxCPU(h_input, h_output_cpu, n);

    // Calculate MSE and display with original precision
    float mse = calculateMSE(h_output_gpu, h_output_cpu, n);
    printf("Mean Squared Error: %e\n", mse);

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_exp));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);
    CUDA_CHECK(cudaEventDestroy(start_exp));
    CUDA_CHECK(cudaEventDestroy(stop_exp));
    CUDA_CHECK(cudaEventDestroy(start_reduce));
    CUDA_CHECK(cudaEventDestroy(stop_reduce));
    CUDA_CHECK(cudaEventDestroy(start_norm));
    CUDA_CHECK(cudaEventDestroy(stop_norm));

    return 0;
}
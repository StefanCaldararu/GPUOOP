#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrixMul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


int main() {
    int N = 256;
    float* h_A = (float*)malloc(N * N * sizeof(float));
    float* h_B = (float*)malloc(N * N * sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = i == j ? 1.0f : 0.0f;
            h_B[i * N + j] = i == j ? 1.0f : 0.0f;
        }
    }

    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, N * N * sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    float* h_C = (float*)malloc(N * N * sizeof(float));
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    bool failed = false;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (h_C[i * N + j] != (i == j ? 1.0f : 0.0f) ){
                failed = true;
                printf("Basic GPU Matmul: Error: h_C[%d][%d] = %f\n", i, j, h_C[i * N + j]);
            }
        }
    }
    if(!failed){
        printf("Basic GPU Matmul: Success!\n");
    }

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}

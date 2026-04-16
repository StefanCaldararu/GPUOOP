#include "GPUMatrix.hpp"
#include <stdexcept>
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


GPUMatrix::GPUMatrix(int n) : n(n){
    cudaMalloc((void**)&data, n * n * sizeof(float));
}

GPUMatrix::GPUMatrix(const std::vector<std::vector<float>>& data) : GPUMatrix(data.size()) {

    std::vector<float> flat(n*n);

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            flat[i*n + j] = data[i][j];
        }
    }

    cudaMemcpy(this->data, flat.data(), n*n*sizeof(float), cudaMemcpyHostToDevice);
}

GPUMatrix::~GPUMatrix(){
    cudaFree(data);
}

std::vector<std::vector<float>> GPUMatrix::matmul(const GPUMatrix& other) const{
    if(other.size() != n){
        throw std::runtime_error("Matrix size mismatch in matmul");
    }
    float* result;
    cudaMalloc((void**)&result, n * n * sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMul<<<numBlocks, threadsPerBlock>>>(data, other.data, result, n);
    cudaDeviceSynchronize();

    std::vector<float> flat(n*n);
    cudaMemcpy(flat.data(), result, n*n*sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<std::vector<float>> ret(n, std::vector<float>(n));
    for (int i=0;i<n;i++)
        for (int j=0;j<n;j++)
            ret[i][j] = flat[i*n+j];

    cudaFree(result);
    return ret;
}

int GPUMatrix::size() const {
    return n;
}

GPUMatrix::GPUMatrix(GPUMatrix&& other) noexcept : n(other.n), data(other.data){
    other.n = 0;
    other.data = nullptr;
}

GPUMatrix& GPUMatrix::operator=(GPUMatrix&& other) noexcept {
    if (this != &other) {
        if (data) {
            cudaFree(data);
        }

        n = other.n;
        data = other.data;

        other.n = 0;
        other.data = nullptr;
    }
    return *this;
}

GPUMatrix::GPUMatrix(const GPUMatrix& other) : n(other.n), data(nullptr) {
    cudaMalloc((void**)&data, n * n * sizeof(float));
    cudaMemcpy(data, other.data, n * n * sizeof(float), cudaMemcpyDeviceToDevice);
}

GPUMatrix& GPUMatrix::operator=(const GPUMatrix& other) {
    if (this != &other) {
        if (data) {
            cudaFree(data);
        }

        n = other.n;

        cudaMalloc((void**)&data, n * n * sizeof(float));
        cudaMemcpy(data, other.data, n * n * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return *this;
}

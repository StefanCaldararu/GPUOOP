#ifndef GPUMATRIX_H
#define GPUMATRIX_H

#include <vector>

class GPUMatrix {
public:
    GPUMatrix(int n);
    GPUMatrix(const std::vector<std::vector<float>>& data);
    ~GPUMatrix();

    GPUMatrix(const GPUMatrix& other);
    GPUMatrix& operator=(const GPUMatrix& other);

    // Move support (TODO: RAII)
    GPUMatrix(GPUMatrix&& other) noexcept;
    GPUMatrix& operator=(GPUMatrix&& other) noexcept;

    std::vector<std::vector<float>> matmul(const GPUMatrix& other) const;

    int size() const;

private:
    int n;
    float* data;
};

#endif
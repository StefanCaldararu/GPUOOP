#ifndef GPUMATRIX_H
#define GPUMATRIX_H

#include <vector>

class GPUMatrix {
public:
    GPUMatrix(int n);
    GPUMatrix(const std::vector<std::vector<float>>& data);
    ~GPUMatrix();

    GPUMatrix(const GPUMatrix&) = delete;
    GPUMatrix& operator=(const GPUMatrix&) = delete;

    std::vector<std::vector<float>> matmul(const GPUMatrix& other) const;

    int size() const;

private:
    int n;
    float* data;
};

#endif
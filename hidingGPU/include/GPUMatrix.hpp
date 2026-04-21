#ifndef GPUMATRIX_H
#define GPUMATRIX_H

#include "Matrix.hpp"
#include <vector>

class GPUMatrix : public Matrix {
    public:
        GPUMatrix(int n);
        GPUMatrix(const std::vector<std::vector<float>>& data);
        ~GPUMatrix();
        
        GPUMatrix(const GPUMatrix&) = delete;
        GPUMatrix& operator=(const GPUMatrix&) = delete;
        
        std::vector<std::vector<float>> matmul(const Matrix& other) override;
        std::vector<std::vector<float>> getData() const override;

        void setData(const std::vector<std::vector<float>>& data) override;
        float getIndividualData(int i, int j) const override;
        int size() const override;
        private:
            std::vector<std::vector<float>> matmulImpl(const GPUMatrix& other) const;
            float* data;

};

#endif
#ifndef CPUMATRIX_H
#define CPUMATRIX_H

#include "Matrix.hpp"
#include <vector>

class CPUMatrix : public Matrix {
    public:
        CPUMatrix(int n) : Matrix(n) {
            data = new float[n*n];
        }
        CPUMatrix(const std::vector<std::vector<float>>& data) : CPUMatrix(data.size()) {
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    this->data[i*n+j] = data[i][j];
                }
            }
        }
        ~CPUMatrix() {
            delete [] data;
        }
        CPUMatrix(const CPUMatrix&) = delete;
        CPUMatrix& operator=(const CPUMatrix&) = delete;
        
        std::vector<std::vector<float>> matmul(const Matrix& other) override;
        std::vector<std::vector<float>> getData() const override;

        void setData(const std::vector<std::vector<float>>& data) override;
        float getIndividualData(int i, int j) const override;
        int size() const override;
        private:
            std::vector<std::vector<float>> matmulImpl(const CPUMatrix& other) const;
            float* data;

};

#endif
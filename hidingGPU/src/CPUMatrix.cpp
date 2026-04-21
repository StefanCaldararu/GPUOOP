#include "CPUMatrix.hpp"
#include <stdexcept>

std::vector<std::vector<float>> CPUMatrix::matmul(const Matrix& other){
    CPUMatrix otherCPU(other.getData());
    return matmulImpl(otherCPU);
}


std::vector<std::vector<float>> CPUMatrix::matmulImpl(const CPUMatrix& other) const{
    if (other.size() != n) {
        throw std::runtime_error("Matrix size mismatch in matmul");
    }

    std::vector<std::vector<float>> result(n, std::vector<float>(n));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += data[i*n + k] * other.getIndividualData(k, j);
            }
            result[i][j] = sum;
        }
    }
    
    return result;
}

std::vector<std::vector<float>> CPUMatrix::getData() const {
    std::vector<std::vector<float>> result(n, std::vector<float>(n));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = data[i*n + j];
        }
    }

    return result;
}

void CPUMatrix::setData(const std::vector<std::vector<float>>& newData) {
    if (newData.size() != n) {
        throw std::runtime_error("Matrix size mismatch in setData");
    }

    for (int i = 0; i < n; i++) {
        if (newData[i].size() != n) {
            throw std::runtime_error("Matrix must be square in setData");
        }
        for (int j = 0; j < n; j++) {
            data[i*n + j] = newData[i][j];
        }
    }
}

float CPUMatrix::getIndividualData(int i, int j) const {
    return data[i*n+j];
}

int CPUMatrix::size() const {
    return n;
}
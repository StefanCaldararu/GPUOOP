#ifndef MATRIX_H
#define MATRIX_H

#include <memory>
#include <vector>

class Matrix {
    public:
        Matrix(int n) : n(n) {}
        virtual ~Matrix() = default;
        virtual std::vector<std::vector<float>> matmul(const Matrix& other) = 0;

        virtual std::vector<std::vector<float>> getData() const = 0;
        virtual void setData(const std::vector<std::vector<float>>& data)  = 0;
        virtual float getIndividualData(int i, int j) const = 0;
        virtual int size() const = 0;
    protected:
        int n;

};

#endif
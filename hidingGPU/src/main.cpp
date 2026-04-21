#include <iostream>
#include <vector>
#include "CPUMatrix.hpp"
#include "GPUMatrix.hpp"

void printMatrix(const std::vector<std::vector<float>>& mat) {
    for (const auto& row : mat) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    std::vector<std::vector<float>> A_data = {
        {1, 0},
        {0, 1}
    };

    std::vector<std::vector<float>> B_data = {
        {5, 6},
        {7, 8}
    };

    GPUMatrix A(A_data);
    CPUMatrix B(B_data);

    std::vector<std::vector<float>> C = A.matmul(B);

    std::cout << "Result of A x B:" << std::endl;
    printMatrix(C);

    return 0;
}
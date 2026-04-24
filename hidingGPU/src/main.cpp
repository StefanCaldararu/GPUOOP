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
        {1, 0},
        {0, 1}
    };

    GPUMatrix A(A_data);
    CPUMatrix B(B_data);

    std::vector<std::vector<float>> vec = B_data;

    std::vector<std::vector<float>> result = A.matmul(B);

    bool failed = false;
    for(int i = 0; i < vec.size(); i++){
        for(int j = 0; j < vec[i].size(); j++){
            if(vec[i][j] != result[i][j]){
                failed = true;
                std::cout << " GPU Matmul Class: Error" << std::endl;
            }
        }
    }

    if(!failed){
        std::cout << "GPU Matmul Class: Success!" << std::endl;
    }

    return 0;
}
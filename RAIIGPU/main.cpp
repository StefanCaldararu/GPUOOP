#include "GPUMatrix.hpp"
#include <vector>
#include <iostream>

int main() {
    std::vector<std::vector<float>> vec = {{1., 0.}, {0., 1.}};
    GPUMatrix mat1(vec);
    GPUMatrix mat2 = mat1;
    std::vector<std::vector<float>> result = mat1.matmul(mat2);
    
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
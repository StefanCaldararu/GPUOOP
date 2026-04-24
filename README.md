# GPU OOP
This project contains code examples and the final paper for CSCI 5448: Object Oriented Analysis and Design. The final paper can be found [here](./main.pdf). In depth descriptions of each code example are included in the final paper. A brief description of each example is given in this README. Instructions on building and running each example are provided below.

## Running Locally
This project focuses on CUDA programming, and as such requires an NVIDIA GPU. Given that 

### System Requirements
You must have a system with an NVIDIA GPU, CUDA Toolkit v12.8 installed. Instructions on installing the CUDA Toolkit can be found [here](https://developer.nvidia.com/cuda-12-8-0-download-archive). Given that the developer does not have access to such a system, minimal instructions on building are included for local compilation and testing.

### Compiling Examples
For the first three examples, compilation is done directly through the `nvcc` compiler. Compilation for the final example is done through the `cmake` file provided. Compilation commands can be found in the Jupyter Notebook provided [here](./Cuda.ipynb).

## Running through Google Colab
To run the code on Google Colab, visit the [Colab Developer Website](https://developers.google.com/colab), ensure you are logged in to google, and open colab by clicking the link in the top right corner. Following this, select the "create new notebook" option. Select "file -> Open Notebook -> Upload -> browse" and select the downloaded [Cuda.ipynb](./Cuda.ipynb) file. In the top right corner, select "Connect T4" to connect to a GPU-enabled runtime. If this option doesn't appear at first, select the dropdown menu to the right and select "Change Runtime Type" to select the T4 runtime. Once connected, run each code block individually to enable the `nvcc` compiler, clone the repository, and run each example individually!

## Example 1: Basic GPU Example
This is a basic GPU matrix multiplication example, designed to familiarize the user with CUDA programming.

## Example 2: Class GPU Example
This example wraps the GPU code in a C++ class to abstract the usage of CUDA away from someone using this Matrix Multiplication Library.

## Example 3: RAII GPU Example
This example implements good RAII and Rule of 3/5/0 principles to help prevent memory leaks.

## Example 4: Hiding GPU Example
This example further abstracts the GPU code by implementing both a CPU and GPU matrix multiplication class, impelmenting a generic Matrix template. 
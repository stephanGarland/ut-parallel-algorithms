#include <cuda.h>
#include <cuda_runtime.h>


#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using std::cout;
using std::endl;

#define BLOCKSIZE 16
#define NUM_BLOCKS 2
#define THREADS_PER_BLOCK 512


__global__ void findMin(int *arr_in, int *arr_out) {
    unsigned int tid = threadIdx.x;
    unsigned int index = threadIdx.x + (blockDim.x * blockIdx.x);
    __syncthreads();

    // This does not yet actually find the minimum; it's a partial implementation of a reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>=1) {
        if (tid < s) {
            if (arr_in[index*s] < arr_in[index]) {
                arr_in[index] = arr_in[index*s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        arr_out[blockIdx.x] = arr_in[0];
    }
}

int main (void) {

    std::ifstream inp;
    inp.open("./inp.txt");
    std::vector<int> A;
    std::vector<int> A_from_cuda;
    int num;
    while ((inp >> num) && inp.ignore()) {
        A.push_back(num);
    }
    // inp.ignore() defaults to EOF, and since the example file doesn't include a \n, add the last number
    // But just in case the test does include a newline, don't add anything
    inp.seekg(-1, std::ios_base::end);
    char c;
    if (c != '\n') {
        A.push_back(num);
    }

    float *cuda_in_ptr;
    float *cuda_out_ptr;
    const int ARRAY_SIZE = A.size();
    const inst ARRAY_BYTES = sizeof int * ARRAY_SIZE;

    cudaMalloc((void**), &cuda_in_ptr, ARRAY_BYTES);
    cudaMemcpy(cuda_in_ptr, A, ARRAY_BYTES, cudaMemcpyHostToDevice);

    findMin<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(cuda_in_ptr);
    cudaMemcpy(A_from_cuda, cuda_in_ptr, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    cout << "Min number in array is: " << A_from_cuda[0];
    cudaFree(cuda_in_ptr);
    free(A_from_cuda);
    free(A);
}

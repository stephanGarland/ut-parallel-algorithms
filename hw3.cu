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

int main (int argc, char **argv) {
    cudaDeviceReset();
    std::ifstream inp;
    inp.open("./inp.txt");
    std::vector<int> A;
    int num;
    while ((inp >> num) && inp.ignore()) {
        A.push_back(num);
    }
    // inp.ignore() defaults to EOF, and since the example file doesn't include a \n, add the last number
    // But just in case the test does include a newline, don't add anything
    inp.seekg(-1, std::ios_base::end);
    char c = 0;
    if (c != '\n') {
        A.push_back(num);
    }

    const int ARRAY_SIZE = A.size();
    const int ARRAY_BYTES = sizeof(int) * ARRAY_SIZE;
    const int OP_BLOCK_ARR_SIZE = NUM_BLOCKS * sizeof(int);

    // CUDA doesn't have vectors, so pull the data out for an array instead
    cout << "Input array size = " << ARRAY_SIZE << endl;
    int *A_for_cuda = A.data();
    int *A_from_cuda = new int[ARRAY_SIZE];
    int *cuda_in_ptr;
    int *cuda_out_ptr;

    cudaMalloc((void**) &cuda_in_ptr, ARRAY_BYTES);
    cudaMalloc((void**) &cuda_out_ptr, OP_BLOCK_ARR_SIZE);
    cudaMemcpy(cuda_in_ptr, A_for_cuda, ARRAY_BYTES, cudaMemcpyHostToDevice);

    //Perform reduction with in each block
    findMin<<<NUM_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(cuda_in_ptr, cuda_out_ptr);
    //Perform reduction of the results across blocks
    findMin<<<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(cuda_out_ptr, cuda_out_ptr);

    int *blocks_output = new int[NUM_BLOCKS];
    cudaMemcpy(blocks_output, cuda_out_ptr, OP_BLOCK_ARR_SIZE, cudaMemcpyDeviceToHost);

    cout << "First 10 values of original array are: ";
    for (int i = 0; i < 10; i++) {
        cout << A_for_cuda[i] << ' ';
    }
    cout << endl;
    cout << "Min : " << blocks_output[0] << endl;

    cout << "Min number in array is: " << A_from_cuda[0] << endl;
    cudaFree(cuda_in_ptr);
    cudaFree(cuda_out_ptr);
    free(A_from_cuda);
    free(blocks_output);
    cudaDeviceReset();
}
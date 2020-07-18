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

    // CUDA doesn't have vectors, so pull the data out for an array instead
    int *A_for_cuda = A.data();
    int *A_from_cuda = new int[ARRAY_SIZE];
    
    int *cuda_in_ptr;
    int *cuda_out_ptr;
    cudaMalloc((void**) &cuda_in_ptr, ARRAY_BYTES);
    cudaMalloc((void**) &cuda_out_ptr, ARRAY_BYTES);
    cudaMemcpy(cuda_in_ptr, A_for_cuda, ARRAY_BYTES, cudaMemcpyHostToDevice);

    findMin<<<NUM_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(cuda_in_ptr, cuda_out_ptr);

    cudaMemcpy(A_from_cuda, cuda_out_ptr, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    cout << "First 10 values of original array are: ";
    for (int i = 0; i < 10; i++) {
        cout << A_for_cuda[i] << ' ';
    }
    cout << endl;

    cout << "First 10 values of CUDA array are: ";
    for (int i = 0; i < 10; i++) {
        cout << A_from_cuda[i] << ' ';
    }
    cout << endl;
    //cout << "Min number in array is: " << A_from_cuda[0] << endl;
 
    cudaFree(cuda_in_ptr);
    free(A_from_cuda);
    cudaDeviceReset();
}
   


/*
{
    // Copy vector to GPU
    thrust::device_vector<int> A = host_nums;
	// Find minimum element using thrust, specifying it to be done on the GPU
    thrust::device_vector<int>::iterator iter = thrust::min_element(thrust::device, A.begin(), A.end());
    
    int pos = iter - A.begin();
    int minA = *iter;

	// Since we know the range to be 0-999, check that the returned min is 0
	assert(minA == 0);
	cout << "Min value is " << minA << " at position " << pos << endl;

	thrust::device_vector<int> B(A.size());
	thrust::fill(B.begin(), B.end(), A.back());
	
	cout << "Last value of A is: " << A.back() << endl;
	// Check that all elements of vector B equal each other
	assert(std::equal(B.begin() + 1, B.end(), B.begin()));

	// And that the two arrays are equal, given that the previous assertion is true
	assert(A.back() == B.back());
	cout << "Every value of B is: " << B.back() << endl;

}
*/

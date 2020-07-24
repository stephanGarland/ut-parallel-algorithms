#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using std::cout;
using std::endl;

#define BLOCKSIZE 16

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
   
__global__ void findMin(std::vector<int> *A, unsigned int size) {
    __shared__ double min[BLOCKSIZE];
    unsigned int tid = threadIdx.x;
    unsigned int index = threadIdx.x + (blockDim.x * blockIdx.x);
    min[tid] = A[index];
    __syncthreads();
    unsigned int numThreads = size;
    for (int j = 1; j < blockDim.x; j *= 2) {
        int k = 2 * j * tid;
        if (k < blockDim.x) {
            if (min[tid + j] < min[tid]) {
                min[tid] = min[tid + s]
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        min[blockIdx.x] = min[0];
    }
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
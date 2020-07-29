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


__global__ void findMin(int n, int MAX, int *arr_in, int *arr_out) {
    extern __shared__ int shared_arr[];
    unsigned int tid = threadIdx.x;
    unsigned int index = threadIdx.x + (blockDim.x * blockIdx.x);
    shared_arr[tid] = MAX; //initialize the shared array to max value
    if (index < n) {
        shared_arr[tid] = arr_in[index]; // copy to shared memory
    }
    __syncthreads();

    // do reduction in shared memory
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s ) {
            if (shared_arr[tid + s] <  shared_arr[tid]) {
                shared_arr[tid] = shared_arr[tid + s];
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        arr_out[blockIdx.x] = shared_arr[0]; // first element of the shared array contains the reduced value with in the block
    }
}

__global__ void findLastDigit(int *arr_in, int *arr_out) {
    unsigned int tid = threadIdx.x;
    int ele = arr_in[tid];
    arr_out[tid] = ele % 10;
}

int main (int argc, char **argv) {
    cudaDeviceReset();
    std::ifstream inp;
    std::ofstream q1a("./q1a.txt");
    std::ofstream q1b("./q1b.txt");

    int BLOCKS = 4;
    int BLOCK_SIZE = 256;
    const char *INPUT_FILE= "./inp/inp.txt";
    std::vector<int> A;

    /*
    if(argc==3) {
        BLOCK_SIZE = atoi(argv[1]);
        INPUT_FILE = argv[2];
    }
    */

    inp.open(INPUT_FILE);
    int num;
    while ((inp >> num) && inp.ignore()) {
        A.push_back(num);
    }
    // inp.ignore() defaults to EOF, and since the example file doesn't include a \n, add the last number
    // But just in case the test does include a newline, don't add anything
    inp.seekg(-1, std::ios_base::end);
    char c;
    inp.get(c);
    if (c != '\n') {
        A.push_back(num);
    }
    int dev = 0;
    cudaSetDevice(dev);
    /*
    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (int)devProps.totalGlobalMem,
               (int)devProps.major, (int)devProps.minor,
               (int)devProps.clockRate);
    }
    */

    const int N = A.size();
    const int ARRAY_BYTES = sizeof(int) * N;

    // calculate the block size
    BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int OP_BLOCK_ARR_SIZE = BLOCKS * sizeof(int);

    /*
    cout << "Input array size = " << N << endl;
    cout << "Output array size = " << OP_BLOCK_ARR_SIZE << endl;
    cout << "Number of blocks for the input = " << BLOCKS << endl;
    */
    
    // CUDA doesn't have vectors, so pull the data out for an array instead
    int *in_arr = A.data();
    int *q1a_out;
    int *q1b_out;
    int *cuda_in;
    int *cuda_out;

    cudaMalloc((void**) &cuda_in, ARRAY_BYTES);
    cudaMalloc((void**) &cuda_out, OP_BLOCK_ARR_SIZE);
    cudaMemcpy(cuda_in, in_arr, ARRAY_BYTES, cudaMemcpyHostToDevice);

    int MAX = std::numeric_limits<int>::max();
    //Perform reduction with in each block
    findMin<<<BLOCKS, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(N, MAX, cuda_in, cuda_out);


    //Perform reduction of the results across blocks
    if (BLOCKS > 1){
        findMin<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(N, MAX, cuda_out, cuda_out);
    }

    q1a_out = (int *)calloc(1, sizeof(int));
    cudaMemcpy(q1a_out, cuda_out, sizeof(int), cudaMemcpyDeviceToHost);

    /*
    cout << "Last 10 in original array are: ";
    for (int i = N; i > N - 10; i--) {
        cout << A.data()[i] << ' ';
    }
    cout << endl;
    */
    q1a << *q1a_out;

    //cout << "Min number in array is: " << *q1a_out << endl;

    cudaFree(cuda_out);
    cudaMalloc((void**) &cuda_out, 10000 * sizeof(int));
    findLastDigit<<<BLOCKS, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(cuda_in, cuda_out);
    q1b_out = (int *)calloc(10000, sizeof(int));
    cudaMemcpy(q1b_out, cuda_out, sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < sizeof(q1b_out) / sizeof(q1b_out[0]); i++) {
        q1b << q1b_out[i] << ', ';
    }

    cudaFree(cuda_in);
    cudaFree(cuda_out);
    free(q1a_out);
    free(q1b_out);
    cudaDeviceReset();
}

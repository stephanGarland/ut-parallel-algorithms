#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using std::cout;
using std::endl;

typedef struct{
    int id;
    int from;
    int to;
}Bucket;

__global__ void groupElements(int n, int *arr_in, int *arr_out, int iteration, int startRange, int endRange) {
    unsigned int index = threadIdx.x + (blockDim.x * blockIdx.x);

    if (index < n) {
        if (arr_in[index] >= startRange && arr_in[index] <= endRange) {
            atomicAdd(&arr_out[iteration], 1);
        }
    }
}

__global__ void groupElementsSharedMem(int n, int *arr_in, int *arr_out, int iteration, int startRange, int endRange) {
    __shared__ int count;
    unsigned int index = threadIdx.x + (blockDim.x * blockIdx.x);
    count = 0;
    __syncthreads();

    if (index < n) {
        if (arr_in[index] >= startRange && arr_in[index] <= endRange) {
            atomicAdd(&count, 1);
        }
    }
    __syncthreads();
     if (threadIdx.x == 0){
        arr_out[blockIdx.x] = count;
     }
}

__global__ void reduceSum(int n, int iteration, int *arr_in, int *arr_out) {
    extern __shared__ int shared_arr[];
    unsigned int tid = threadIdx.x;
    unsigned int index = threadIdx.x + (blockDim.x * blockIdx.x);
    shared_arr[tid] = 0;
    if (index < n) {
        shared_arr[tid] = arr_in[index]; // copy to shared memory
    }
    __syncthreads();

    // do reduction in shared memory
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s ) {
              shared_arr[tid] += shared_arr[tid + s];
            __syncthreads();
        }
    }

    if (tid == 0) {
        arr_out[iteration] = shared_arr[0]; // first element of the shared array contains the reduced value with in the block
    }
}

int main (int argc, char **argv) {
    cudaDeviceReset();
    std::ifstream inp;
    int BLOCKS = 4;
    int BLOCK_SIZE = 256;
    const char *INPUT_FILE= "../inp/inp.txt";
    std::vector<int> A;

    if(argc == 3) {
        BLOCK_SIZE = atoi(argv[2]);
        INPUT_FILE = argv[3];
    }

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
    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (int)devProps.totalGlobalMem,
               (int)devProps.major, (int)devProps.minor,
               (int)devProps.clockRate);
    }
    
    const int N = A.size();
    const int ARRAY_BYTES = sizeof(int) * N;

    cout << "Input array size = " << N << endl;

    // CUDA doesn't have vectors, so pull the data out for an array instead
    int *in_arr = A.data();
    int *cuda_in;
    int *B;
    int *C;

    Bucket bkt_case_2[10]= { 
        {0,0,99},
        {1,100,199},
        {2,200,299},
        {3,300,399},
        {4,400,499},
        {5,500,599},
        {6,600,699},
        {7,700,799},
        {8,800,899},
        {9,900,999},
    };

      Bucket bkt_case_3[10]= { 
        {0,0,99},
        {1,0,199},
        {2,0,299},
        {3,0,399},
        {4,0,499},
        {5,0,599},
        {6,0,699},
        {7,0,799},
        {8,0,899},
        {9,0,999},
    };

    BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cout << "Blocks : " << BLOCKS << " Block size : " << BLOCK_SIZE << endl;

    cudaMalloc((void**) &cuda_in, ARRAY_BYTES);
    cudaMalloc((void**) &B, 10 * sizeof(int));
    cudaMalloc((void**) &C, 10 * sizeof(int));

    cudaMemcpy(cuda_in, in_arr, ARRAY_BYTES, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //q2 part 1 - Global memory B
    cout << "\n*** Executing part 1 of question 2 *** " << endl;
    cudaEventRecord(start, 0);
    for (int i = 0; i < 10; i++) {
        Bucket *item = &bkt_case_2[i];
        groupElements<<<BLOCKS, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(N, cuda_in, B, item->id, item->from, item->to);
    }
    cudaEventRecord(stop, 0);

    cout << "\n*** Executing part 2 of question 2 *** " << endl;
    cudaEventRecord(start, 0);
    int *cuda_interim_arr_case_2;
        for(int i = 0; i < 10; i++) {
            Bucket *item = &bkt_case_2[i];
            cudaMalloc((void**) &cuda_interim_arr_case_2, BLOCKS * sizeof(int));
            // Step 1. First do the count in per block shared variable
            groupElementsSharedMem<<<BLOCKS, BLOCK_SIZE>>>(N, cuda_in, cuda_interim_arr_case_2, item->id, item->from, item->to);

            // Step 2. Sum the counts across blocks and store in global B
            reduceSum<<<1, BLOCKS, BLOCKS * sizeof(int)>>>(BLOCKS, item->id, cuda_interim_arr_case_2, B);
            cudaFree(cuda_interim_arr_case_2);
        }
    cudaEventRecord(stop, 0);

    cout << "\n*** Executing part 3 of question 2 ***" << endl;
    cudaMemcpy(cuda_in, B, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaEventRecord(start, 0);
    int *cuda_interim_arr_case_3;
    for (int i = 0; i < 10; i++) {
        Bucket *item = &bkt_case_3[i];
        cudaMalloc((void**) &cuda_interim_arr_case_3, BLOCKS * sizeof(int));
        groupElementsSharedMem<<<BLOCKS, BLOCK_SIZE>>>(N, cuda_in, cuda_interim_arr_case_3, item->id, item->from, item->to);
        reduceSum<<<1, BLOCKS, BLOCKS * sizeof(int)>>>(BLOCKS, item->id, cuda_interim_arr_case_3, C);
        cudaFree(cuda_interim_arr_case_3);
    }

    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    int *out_arr_B = new int[10];
    cudaMemcpy(out_arr_B, B, 10*sizeof(int), cudaMemcpyDeviceToHost);
    cout << "\noutput array B : ";

    for (int i = 0; i < 10; i++) {
        cout << out_arr_B[i] << ' ';
    }
    cout << endl;

    int *out_arr_C = new int[10];
    cudaMemcpy(out_arr_C, C, 10*sizeof(int), cudaMemcpyDeviceToHost);
    cout << "\noutput array C : ";

    for (int i = 0; i < 10; i++) {
        cout << out_arr_C[i] << ' ';
    }
    cout << endl;

    cout << "average time elapsed : " << elapsedTime << endl;

    cudaFree(cuda_in);
    cudaFree(B);
    cudaFree(C);
    free(out_arr_B);
    free(out_arr_C);
    cudaDeviceReset();
}
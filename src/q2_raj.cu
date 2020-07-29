
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <fstream>

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

__global__ void scan(int *arr_out, int *arr_in, int n) {   
    int tid = threadIdx.x;  
    //copy the input array to output 
    arr_out[tid] = arr_in[tid];
    __syncthreads();

    //Perform inclusive scan on output array
    for (int d = 1; d < n; d *= 2){
        if(tid - d >= 0 ){
            arr_out[tid] = arr_out[tid] + arr_out[tid-d];
        }
        __syncthreads();
    }
} 

void createOutputFile(const int *arr, const int size, const char *filename){
    std::ofstream outputfile;
    outputfile.open(filename);

    for (int i=0; i < size; i++)
    {
        outputfile  << arr[i];
        if (i < size - 1) {
            outputfile << ", ";
        }
    }
    outputfile << endl;
    outputfile.close();
}

int main (int argc, char **argv) {
    cudaDeviceReset();
    std::ifstream inp;
    int BLOCKS = 4; // Number of blocks are calculated based on input size below
    int BLOCK_SIZE = 256;
    const char *INPUT_FILE= "./inp/inp.txt";
    std::vector<int> A;

    inp.open(INPUT_FILE);

    int num;
    while ((inp >> num) && inp.ignore()) {
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
    // CUDA doesn't have vectors, so pull the data out for an array instead

    cout << "Input array size = " << N << endl;

    int *in_arr = A.data();
    int *cuda_in;
    int *B;
    int *out_arr;

    Bucket bkt[10]= { {0,0,99},
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


    BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cout << "Blocks : " << BLOCKS << " Block size : " << BLOCK_SIZE << endl;

    cudaMalloc((void**) &cuda_in, ARRAY_BYTES);
    cudaMalloc((void**) &B, 10 * sizeof(int));

    cudaMemcpy(cuda_in, in_arr, ARRAY_BYTES, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    //q2 part a1 - Global memory 
    cout << "*** Start part a of question 2 *** " << endl;
    cudaEventRecord(start, 0);
    for(int i=0; i<10; i++){
        Bucket *item = &bkt[i];
        groupElements<<<BLOCKS, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(N, cuda_in, B, item->id, item->from, item->to);
    }
    cudaEventRecord(stop, 0);
    out_arr = new int[10];
    cudaMemcpy(out_arr, B, 10*sizeof(int), cudaMemcpyDeviceToHost);
    createOutputFile(out_arr,10,"q2a.txt");
    cudaEventRecord(stop, 0);
    out_arr = new int[10];
    cudaMemcpy(out_arr, B, 10*sizeof(int), cudaMemcpyDeviceToHost);        
    createOutputFile(out_arr,10,"q2b.txt");

    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "output array B : ";
    for (int i = 0; i < 10; i++) {
        cout << out_arr[i] << ' ';
    }
    cout << endl;
    cout << "average time elapsed : " << elapsedTime << endl;
    cudaFree(B);
    free(out_arr);
    cout << "**** End part a of question 2 **** " << endl;

    //q2 part b - Shared memory
    cudaMalloc((void**) &B, 10 * sizeof(int));
    cout << "*** Start part b of question 2 *** " << endl;
    cudaEventRecord(start, 0);
    int *cuda_interim_arr;
    for(int i=0; i<10; i++){
         Bucket *item = &bkt[i];
         cudaMalloc((void**) &cuda_interim_arr, BLOCKS * sizeof(int));
         // Step 1. First do the count in per block shared variable
         groupElementsSharedMem<<<BLOCKS, BLOCK_SIZE>>>(N, cuda_in, cuda_interim_arr, item->id, item->from, item->to);

         // Step 2. Sum the counts across blocks and store in global B
         reduceSum<<<1, BLOCKS, BLOCKS * sizeof(int)>>>(BLOCKS, item->id, cuda_interim_arr, B);
         cudaFree(cuda_interim_arr);
     }
    cudaEventRecord(stop, 0);
    out_arr = new int[10];
    cudaMemcpy(out_arr, B, 10*sizeof(int), cudaMemcpyDeviceToHost);        
    createOutputFile(out_arr,10,"q2b.txt");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "output array B : ";
    for (int i = 0; i < 10; i++) {
        cout << out_arr[i] << ' ';
    }
    cout << endl;
    cout << "average time elapsed : " << elapsedTime << endl;
    cout << "**** End part b of question 2 **** " << endl;

    //q2 part c - Inclusive scan
    cout << "*** Start part c of question 2 *** " << endl;
    int *C;
    cudaMalloc((void**) &C, 16 * sizeof(int));
    cudaEventRecord(start, 0);    
    scan<<<1, 16>>>(C, B, 16); // Perform the inclusive scan.Take array size as 16 padded with 0s so that its power of 2
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    out_arr = new int[10];
    cudaMemcpy(out_arr, C, 10*sizeof(int), cudaMemcpyDeviceToHost);
    createOutputFile(out_arr,10,"q2c.txt");
    cout << "output array C : ";
    for (int i = 0; i < 10; i++) {
        cout << out_arr[i] << ' ';
    }
    cout<<endl;
    cout << "average time elapsed : " << elapsedTime << endl;
    cout << "**** End part c of question 2 **** " << endl;
    cudaFree(cuda_in);
    cudaFree(B);
    cudaFree(C);
    free(out_arr);
    cudaDeviceReset();
}
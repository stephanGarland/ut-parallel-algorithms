CPU_CC=g++
CPU_CFLAGS=-std=c++11
GPU_CC=nvcc

cpu:
	$(CPU_CC) $(CPU_CFLAGS) -o hw3_cpu hw3_cpu.cpp

gpu:
	$(GPU_CC) -o hw3_gpu hw3_gpu.cu

clean:
	rm hw3_cpu hw3_gpu

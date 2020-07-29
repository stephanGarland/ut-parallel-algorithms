GPU_CC=nvcc

q1:
	$(GPU_CC) -o ./bin/q1 ./src/q1.cu

q2:
	$(GPU_CC) -o ./bin/q2 ./src/q2.cu
	$(GPU_CC) -o ./bin/q2_raj ./src/q2_raj.cu

clean:
	rm ./bin/*

GPU_CC=nvcc

q1:
	$(GPU_CC) -o q1 q1.cu

q2:
	$(GPU_CC) -o q2 q2.cu

clean:
	rm q1 q2

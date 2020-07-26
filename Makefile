GPU_CC=nvcc

q1:
	$(GPU_CC) -o hw3_q1 hw3_q1.cu

q2:
	$(GPU_CC) -o hw3_q2 hw3_q2.cu

clean:
	rm hw3_q1 hw3_q2

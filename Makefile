CC = g++-9
CFLAGS = -std=c++11 -Wall

parser:
	$(CC) $(CFLAGS) -o parser parser.cpp
clean:
	rm parser
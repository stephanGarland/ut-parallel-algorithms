CC = g++-9
CFLAGS = -std=c++11 -Wall

build: parser.o
	$(CC) $(CFLAGS) -o ./bin/parser parser.o
	rm *.o
parser.o:
	$(CC) $(CFLAGS) -c ./src/parser.cpp
clean:
	rm ./bin/parser
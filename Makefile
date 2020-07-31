CC = g++-9
CFLAGS = -std=c++11 -Wall

build: parser.o CSRGraphV2.o
	$(CC) $(CFLAGS) -o ./bin/parser parser.o CSRGraphV2.o
	rm *.o
parser.o:
	$(CC) $(CFLAGS) -c ./src/parser.cpp
CSRGraphV2.o:
	$(CC) $(CFLAGS) -c ./src/csr/CSRGraphV2.cpp
clean:
	rm ./bin/parser
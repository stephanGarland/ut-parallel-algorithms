CC = g++-9
CFLAGS = -std=c++11 -Wall

build: parser.o bellmanford.o
	$(CC) $(CFLAGS) -o ./bin/parser parser.o bellmanford.o
	rm *.o
parser.o:
	$(CC) $(CFLAGS) -c ./src/parser.cpp
bellmanford.o:
	$(CC) $(CFLAGS) -c ./src/bellmanford.cpp
clean:
	rm ./bin/parser
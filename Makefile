CC = g++-9
CFLAGS = -std=c++11 -Wall

parser:
	$(CC) $(CFLAGS) -o ./bin/parser ./src/parser.cpp
clean:
	rm ./bin/parser
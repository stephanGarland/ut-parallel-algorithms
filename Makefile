CC       = g++-9
CFLAGS   = -std=c++11 -Wall
PROGNAME = parser
OFILES   = $(patsubst %.cpp,%.o,$(wildcard src/*.cpp))

vpath %.cpp src

build:
	@make $(PROGNAME)

$(PROGNAME): $(OFILES)
	$(CC) $(CFLAGS) -o ./bin/$(PROGNAME) $(OFILES)

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm ./bin/parser
	rm *.o
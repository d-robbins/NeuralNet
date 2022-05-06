SRC := main.c network.c node.c layer.c

all:
	gcc-11 -g -Wall -fopenmp $(SRC) -o nn
	
clean:
	rm $(EXC_NAME)
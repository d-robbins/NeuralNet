SRC := main.c network.c
CC := gcc-11
OPTIONS := -g -Wall -fopenmp

all:
	$(CC) $(OPTIONS) $(SRC) -o nn
	
clean:
	rm $(EXC_NAME)
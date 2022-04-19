SRC := main.c network.c node.c layer.c

all:
	gcc $(SRC) -o main
	
clean:
	rm $(EXC_NAME)
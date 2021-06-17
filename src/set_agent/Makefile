INC = $(shell python3-config --includes)
LIB = $(shell python3-config --embed --ldflags)
OPTS = -g -Wall -std=c++1z

# Main: Main.o
# 	clang++ $(INC) $(LIB) -o Main Main.o

run_file: run_file.o
	clang++ $(INC) $(LIB) -o run_file run_file.o

# Main.o: main.cpp run_file.cpp
# 	clang++ $(OPTS) $(INC) -c main.cpp

run_file.o: run_file.cpp CPyObject.hpp
	clang++ $(OPTS) $(INC) -c run_file.cpp

clean:
	rm -rf run_file *.o
	rm -rf Main *.o
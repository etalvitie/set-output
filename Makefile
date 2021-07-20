INC = $(shell python3-config --includes)
LIB = $(shell python3-config --embed --ldflags)
OPTS = -g -Wall -std=c++1z

# real compilation requirements
Main: Main.o CPPWrapper.o
	clang++ $(INC) $(LIB) -o Main Main.o CPPWrapper.o

CPPWrapper: CPPWrapper.o
	clang++ $(INC) $(LIB) -o CPPWrapper CPPWrapper.o

Main.o: main.cpp CPPWrapper.hpp ObjectState.cpp
	clang++ $(OPTS) $(INC) -c main.cpp

CPPWrapper.o: CPPWrapper.cpp CPPWrapper.hpp CPyObject.hpp ObjectState.cpp
	clang++ $(OPTS) $(INC) -c CPPWrapper.cpp




# run_file: run_file.o
# 	clang++ $(INC) $(LIB) -o run_file run_file.o

# run_file.o: run_file.cpp CPyObject.hpp
# 	clang++ $(OPTS) $(INC) -c run_file.cpp

# test for state 
# Main: Main.o 
# 	clang++ $(INC) $(LIB) -o Main Main.o

# Main.o: main.cpp ObjectState.cpp
# 	clang++ $(OPTS) $(INC) -c main.cpp



clean:
	rm -rf CPPWrapper *.o
	rm -rf run_file *.o
	rm -rf Main *.o
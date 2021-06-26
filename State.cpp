#include <iostream>
#include "State.hpp"

#ifdef PATH
#define PATH_ PATH
#else
#define PATH_ "set-output/" // Default path


using namespace std;

State::State(vector<int> data) : data_(data){}

// destructor
State::~State() {
	data_.clear();
}

void print() {
   cout << "The vector elements are: ";

   for(int i=0; i < data_.size(); i++)
   {
	   cout << data_.at(i) << endl;
   }
}

#endif
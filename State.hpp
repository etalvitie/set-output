#ifndef STATES_HPP
#define STATES_HPP

#include <string>
#include <vector>

using namespace std;

class State{
  public:
    //constructor 
    Class(vector<int> data);

    //destructor
    ~Class();

    int getSize();
    void print();

  private:
    vector<int> data_;
};

#endif
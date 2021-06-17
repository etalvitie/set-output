#ifndef CLASS_HPP
#define CLASS_HPP

#include "CPyObject.hpp"
#include <string>
#include <tuple>
#include <map>
#include <vector>

using namespace std;

class MinAtarInterface{
  public:
    //constructor (take in checkpoint path to model)
    Class(string path);

    //destructor
    ~Class();

    map<string,string> predict();

  private:
    CPyObject model_;
    string path_;
};

#endif
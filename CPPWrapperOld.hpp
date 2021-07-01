#ifndef CLASS_HPP
#define CLASS_HPP

#include "CPyObject.hpp"
#include "ObjectState.cpp"
#include <string>
#include <tuple>
#include <map>
#include <vector>

using namespace std;

class CPPWrapper{
  public:
    //constructor (take in checkpoint path to model)
    CPPWrapper();

    //destructor
    ~CPPWrapper();

    tuple<vector<ObjectState>, vector<ObjectState>, vector<ObjectState>> predict(vector<ObjectState> s, vector<float> a);
    vector<ObjectState> pysetToStateVector(PyObject* incoming, vector<bool> history);
    // void updateModel(vector<ObjectState> s, vector<ObjectState> a, vector<ObjectState> sprime, vector<ObjectState> sappear, float r);

  private:
    CPyObject model_;
    bool train_exist;
    bool train_appear;
    size_t obj_in_len;
    size_t env_len;
    size_t obj_reg_len;
    size_t obj_attri_len;
    size_t new_set_size;
    char* exist_ckpt_path;
    char* appear_ckpt_path;
};

#endif
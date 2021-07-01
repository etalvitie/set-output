#ifndef CLASS_HPP
#define CLASS_HPP

#include "CPyObject.hpp"
#include <string>
#include <tuple>
#include <vector>

using namespace std;

class CPPWrapper{
  public:
    size_t obj_in_len;
    size_t env_len;
    CPyObject model_;
    //constructor (take in checkpoint path to model)
    CPPWrapper();

    //destructor
    ~CPPWrapper();

    // tuple<float**, float**, float**> predict(float** s, float* a);
    // float** pyToCppArray(PyObject* incoming);

    tuple<vector<vector<float>>, vector<vector<float>>, vector<vector<float>>> predict(const vector<vector<float>>& s, const vector<float>& a);
    vector<vector<float>> pyToVector(PyObject* incoming);

    // void updateModel(vector<ObjectState> s, vector<ObjectState> a, vector<ObjectState> sprime, vector<ObjectState> sappear, float r);

  private:
    bool train_exist;
    bool train_appear;
    size_t obj_reg_len;
    size_t obj_attri_len;
    size_t new_set_size;
    string exist_ckpt_path;
    string appear_ckpt_path;
};

#endif
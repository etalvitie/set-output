#ifndef CLASS_HPP
#define CLASS_HPP

#include "CPyObject.hpp"
#include <string>
#include <tuple>
#include <vector>

using namespace std;

class CPPWrapper{
  public:
    //constructor (take in checkpoint path to model)
    CPPWrapper();

    //destructor
    ~CPPWrapper();

    // tuple<float**, float**, float**> predict(float** s, float* a);
    // float** pyToCppArray(PyObject* incoming);

    tuple<vector<vector<float>>, vector<vector<float>>, vector<vector<float>>> predict(const vector<vector<float>>& s, const vector<float>& a);
    vector<vector<float>> pyToVector(PyObject* incoming);
    void updateModel(vector<vector<float>> s, vector<float> a, vector<vector<float>> sprime, vector<vector<float>> sappear, float r);
    PyObject* vectorToPyTuple(const vector<vector<float>>& s);
    void decref(PyObject*& doublevector);

  private:
    size_t obj_in_len;
    size_t env_len;
    bool train_exist;
    bool train_appear;
    size_t obj_reg_len;
    size_t obj_attri_len;
    size_t new_set_size;
    string exist_ckpt_path;
    string appear_ckpt_path;
    CPyObject model_;
};

#endif
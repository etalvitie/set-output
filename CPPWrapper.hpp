#ifndef CLASS_HPP
#define CLASS_HPP

#include "CPyObject.hpp"
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

    tuple<vector<int>, vector<int>, vector<int>> predict(vector<vector<int>> s, vector<int> a);
    void updateModel(vector<int> s, vector<int> a, vector<int> sprime, vector<int> sappear, float r);

  private:
    CPyObject model_;
    char[] exist_ckpt_path;
    char[] appear_ckpt_path;
    bool train_exist;
    bool train_appear;
    size_t obj_in_len;
    size_t env_len;
    size_t obj_reg_len;
    size_t obj_attri_len;
    size_t new_set_sizel;
};

#endif
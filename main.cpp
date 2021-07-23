#include <iostream>
#include "CPPWrapper.hpp"
#include <string>
#include <tuple>
#include <vector>


using namespace std;

int main(){
    // type needs to be one-hot encoding
    vector<float> o1{3,9,-1,0,1,0,0,0,0,0,1};
    vector<float> o2{0.9166666865348816,0.0,-0.0833333358168602,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0};
    vector<float> o3{2,2,0,0,1,0,0,0,0,0,1};
    vector<vector<float>> s1{o1,o2,o3};
    vector<vector<float>> s2{o1,o2};
    vector<vector<float>> s3{o3};
    vector<vector<float>> s4;
    vector<float> action{0,0,0,0,1,0};

    Py_Initialize();
    
    CPPWrapper *model = new CPPWrapper("","","",true,true,true,11,6,2,6,3,64,false,false,false,false);

    tuple<vector<vector<float>>,vector<vector<float>>,vector<vector<float>>,float> output = model->predict(s1,action);

    vector<vector<float>> sprime = get<1>(output);
    vector<vector<float>> sappear = get<2>(output);
    float r = get<3>(output);
    
    for (size_t i = 0; i < sprime.size(); i++) {
        vector<float> temp = sprime.at(i);
        for (size_t j = 0; j < temp.size(); j++) {
            if (j == 2) {
                cout << "visPrediction: " << temp.at(j) << endl;
            }
            else {
                cout << "element: " << temp.at(j) << endl;
            }
            
        }    
    }

    cout << "reward: " << r << endl;

    model->updateModel(s1,action,s1,s4,5.0);
	model->updateModel(s2,action,s2,s3,10.0);

	tuple<vector<vector<float>>,vector<vector<float>>,vector<vector<float>>,float> output2 = model->predict(s1,action);
    vector<vector<float>> sprime2 = get<1>(output2);
    vector<vector<float>> sappear2 = get<2>(output2);
    float r2 = get<3>(output2);
	for (size_t i = 0; i < sprime2.size(); i++) {
        vector<float> temp = sprime2.at(i);
        for (size_t j = 0; j < temp.size(); j++) {
            if (j == 2) {
                cout << "visPrediction: " << temp.at(j) << endl;
            }
            else {
                cout << "element: " << temp.at(j) << endl;
            }
            
        }
    }

    cout << "reward: " << r2 << endl;

    cout << "deleting model..." << endl;
    delete model;

    Py_Finalize();

    return 0;
};


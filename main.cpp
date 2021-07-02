#include <iostream>
#include "CPPWrapper.cpp"
#include <string>
#include <tuple>
#include <vector>


using namespace std;

vector<vector<float>> test(const vector<vector<float>>& s) 
{
	Py_ssize_t row = s.size();
	PyObject *set = PyTuple_New(row);
	for (Py_ssize_t i = 0; i < row; i++) {
		Py_ssize_t col = s.at(i).size();
		PyObject *object = PyTuple_New(col);
		vector<float> temp = s.at(i);
		for (Py_ssize_t j = 0; j < col; j++) {
			PyTuple_SET_ITEM(object, j, PyFloat_FromDouble(temp.at(j)));
		}
			
		PyTuple_SET_ITEM(set, i, object);
	}

	Py_ssize_t idx = 0;
	Py_ssize_t length = PyList_Size(set);
	Py_ssize_t height = PyList_Size(PyList_GetItem(set, idx));
	vector<vector<float>> output;
	
	if (PyList_Check(set)) {
		for (Py_ssize_t i = 0; i < length; i++) {
			CPyObject item = PyList_GetItem(set, i);
			vector<float> obj;
			
			for (Py_ssize_t j = 0; j < height; j++) {
				obj.push_back(PyLong_AsLong(PyList_GetItem(set, j)));
			}

			output.push_back(obj);
		}
	} else {
		throw logic_error("Passed PyObject pointer was not a list array...");
	}

    for (Py_ssize_t i = 0; i < PyList_Size(set); i++) {
		Py_DECREF(PyList_GetItem(set, i));
	}
    Py_DECREF(set);

	return output;
}

int main(){
    vector<float> o1{0,0,0,0,0,0,0,0};
    vector<float> o2{1,1,1,1,1,1,1,1};
    vector<float> o3{2,2,2,2,2,2,2,2};
    vector<vector<float>> s1{o1,o2,o3};
    vector<float> action{0,1,0,0,1,0};
    
    CPPWrapper model;

    tuple<vector<vector<float>>,vector<vector<float>>,vector<vector<float>>> output = model.predict(s1,action);
    vector<vector<float>> sprime = get<1>(output);
    vector<vector<float>> sappear = get<2>(output);
    float r = 1;
    
    for (size_t i = 0; i < sprime.size(); i++) {
        vector<float> temp = sprime.at(i);
        for (size_t j = 0; j < temp.size(); j++)
        cout << "element: " << temp.at(j) << endl;
    }

    model.updateModel(s1,action,sprime,sappear,r);

    return 0;
};


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

    //vector<vector<float>> x = test(s1);
    //cout << x.size() << endl;


    // float** s;
    // float* a;
    
    // if (s1.size() > 0)
    // {
    //     s = new float*[s1.size()];
    //     for (size_t i = 0; i < s1.size(); i++)
    //     {
    //         s[i] = new float[s1.at(0).size()];
    //     }
    // }

	// a = new float[action.size()];
    
    // for (size_t i = 0; i < s1.size(); i++)
    // {
    //     for (size_t j = 0; j < s1.at(0).size(); j++)
    //     {
    //         *(*(s+i)+j) = s1[i][j];
    //     }
    // }

    // for (size_t i = 0; i < action.size(); i++)
    // {
    //     *(a+i) = action[i];
    // }
    
    CPPWrapper model;

    cout << PyBytes_AS_STRING(PyUnicode_AsEncodedString(PyObject_Repr(model.model_), "utf-8", "~E~")) << endl;


    tuple<vector<vector<float>>,vector<vector<float>>,vector<vector<float>>> output = model.predict(s1,action);
    vector<vector<float>> set = get<0>(output);
    
    for (size_t i = 0; i < set.size(); i++) {
        vector<float> temp = set.at(i);
        for (size_t j = 0; j < temp.size(); j++)
        cout << "element: " << temp.at(j) << endl;
    }



    // vector<bool> vis1{true,false};
    // vector<bool> vis2{true,true,false,false};
    // ObjectState s1(1,1,0,0,vis1,1,1);
    // ObjectState s2(2,2,0,0,vis2,1,2);
    // cout << s1 << endl;
    // cout << s2 << endl;
    // cout << s1.getVisibility(0) << endl;
    // cout << s1.getVisibility(5) << endl;
    // cout << s1.getVisibility(62) << endl;

    // vector<float> action{0,0};

    // cout << "Test Predict and Constructor..." << endl;
    // CPPWrapper model;
    // vector<ObjectState> set;
    // set.push_back(s1);
    // set.push_back(s2);
    // tuple<vector<ObjectState>, vector<ObjectState>, vector<ObjectState>> pred1 = model.predict(set,action);


    // vector<ObjectState> vect = get<0>(pred1);
    // for (auto obj : vect) {
    //     cout << obj << endl;
    // }

    return 0;
};


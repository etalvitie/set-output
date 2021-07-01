#include <stdio.h>
#include <iostream>
#include "CPPWrapper.hpp"
#include <list>

#ifdef PATH
#define PATH_ PATH
#else
#define PATH_ "set_agent/" // Default path

// set input, action vector for predict are from the training loop

using namespace std;

CPPWrapper::CPPWrapper() : 
				exist_ckpt_path(nullptr),
                appear_ckpt_path(nullptr),
                train_exist(true),
                train_appear(true),
                obj_in_len(8),
                env_len(6),
                obj_reg_len(2),
                obj_attri_len(2),
                new_set_size(3)
{

	// Initialize Python environment
	Py_Initialize();

	// path to directory of module
	const char *scriptDirectoryName = "./";
	PyObject *sysPath = PySys_GetObject("path");
    PyObject *curPath = PyUnicode_FromString(scriptDirectoryName);
    int result = PyList_Insert(sysPath, 0, curPath);
	// module name
    PyObject *pModule = PyImport_ImportModule("src.set_agent.nn_prediction_model");


	// if the module exists
	if (pModule)
	{
		// get PredictionModel object
		CPyObject class_ = PyObject_GetAttrString(pModule, "PredictionModel");
		char arr1[1]; 
		char arr2[1];

		CPyObject args = Py_BuildValue("[c][c]bbiiiii",
									arr1,
									arr2,
									train_exist,
									train_appear,
									obj_in_len,
									env_len,
									obj_reg_len,
									obj_attri_len,
									new_set_size);
		model_ = PyObject_CallObject(class_, args);
	}
	else
	{
		cout << "ERROR: Module not imported" << endl;
	}
}

// destructor
CPPWrapper::~CPPWrapper() {
	model_.release();
	Py_Finalize();
}

// NEED TO EDIT FOR PARAMETERS s,a
tuple<vector<ObjectState>, vector<ObjectState>, vector<ObjectState>> CPPWrapper::predict(vector<ObjectState> s, vector<float> a) 
{
	vector<vector<bool>> histories;

	
	float states[s.size()][13];
	cout << "length of states is" << sizeof(states) << endl;
	float action[a.size()];
	for (size_t i = 0; i < states.size(); i++)
	{
		ObjectState s1 = s.at(i);
		vector<bool> history;

		states[i][0] = s1.getXPos();
		states[i][1] = s1.getYPos();
		states[i][2] = s1.getXVel();
		states[i][3] = s1.getYVel();
		states[i][4] = s1.getType();

		for (size_t j = 0; j < 7; j++)
		{
			if (j < s1.getVisHistorySize())
			{
				states[i][j+5] = s1.getVisibility(j);
				history.push_back(s1.getVisibility(j));
			}
			else 
			{
				states[i][j+5] = 0;
			}
		}
		states[i][12] = s1.getType();

		histories.push_back(history);
	}
	cout << "XPOS is " << states[1][0] << endl;
	cout << "YPOS is " << states[1][1] << endl;
	cout << "XVEL is " << states[1][2] << endl;
	cout << "YVEL is " << states[1][3] << endl;
	cout << "TYPE is " << states[1][4] << endl;

	for (size_t i = 0; i < action.size(); i++)
	{
		action[i] = a.at(i);
	}


	// float** states = new float*[s.size()];
	// for (int i = 0; i < s.size(); i++)
	// {
	// 	ObjectState s1 = s.at(i);
	// 	states[i] = new float[13];
	// 	vector<bool> history;
	// 	vector<float> temp;
	// 	temp.push_back(s1.getXPos());
	// 	temp.push_back(s1.getYPos());
	// 	temp.push_back(s1.getXVel());
	// 	temp.push_back(s1.getYVel());
	// 	temp.push_back(s1.getType());
	// 	for (size_t i = 0; i < 7; i++)
	// 	{
	// 		if (i < s1.getVisHistorySize())
	// 		{
	// 			temp.push_back(s1.getVisibility(i));
	// 			history.push_back(s1.getVisibility(i));
	// 		}
	// 		else 
	// 		{
	// 			temp.push_back(0);
	// 		}
	// 	}
	// 	temp.push_back(s1.getID());

	// 	histories.push_back(history);
	// 	states[i] = &temp[0];
		
	// }
	// cout << "XPOS is " << states[1][0] << endl;
	// cout << "YPOS is " << states[1][1] << endl;
	// cout << "XVEL is " << states[1][2] << endl;
	// cout << "YVEL is " << states[1][3] << endl;
	// cout << "TYPE is " << states[1][4] << endl;

	// float* action = &a[0];

	CPyObject predict = PyObject_GetAttrString(model_, "predict");

	// if class exists and can be called
	if (predict && PyCallable_Check(predict))
	{
		CPyObject pyTuple = PyObject_CallFunction(predict, "[[f]][f]", states, action);

		if (PyTuple_Check(pyTuple)) 
		{
			Py_ssize_t s_pos = 0;
			Py_ssize_t sprimepos = 1;
			Py_ssize_t sappearpos = 2;

			vector<ObjectState> s_ = pysetToStateVector(PyTuple_GetItem(pyTuple, s_pos), histories);
			vector<ObjectState> sprime = pysetToStateVector(PyTuple_GetItem(pyTuple, sprimepos), histories);
			vector<ObjectState> sappear = pysetToStateVector(PyTuple_GetItem(pyTuple, sappearpos), histories);
			return make_tuple(s_, sprime, sappear);
		}
		else
		{
			cout << "ERROR: returned a nontuple" << endl;
		}
	}
	else
	{
		cout << "ERROR: predict function" << endl;
	}
}

// PyObject (set prediction) -> set 
vector<ObjectState> CPPWrapper::pysetToStateVector(PyObject* incoming, vector<bool> history) {
	vector<ObjectState> stateVector;

	if (PyList_Check(incoming)) {
		for (Py_ssize_t i = 0; i < PyList_Size(incoming); i++) {
			CPyObject item = PyList_GetItem(incoming, i);
			vector<float> data;
			
			for (Py_ssize_t j = 0; j < PyList_Size(item); j++) {
				CPyObject value = PyList_GetItem(incoming, j);
				data.push_back( PyFloat_AsDouble(value) );
			}
			// vector<float> history(data.begin()+6,data.end()-1);
			ObjectState result(data.at(0), 
							   data.at(1), 
							   data.at(2), 
							   data.at(3),
							   history, 
							   data.at(4),
							   data.at(data.size()-1));
			stateVector.push_back(result);
		}
	} else {
		throw logic_error("Passed PyObject pointer was not a list array...");
	}

	return stateVector;
}

// void CPPWrapper::updateModel(vector<ObjectState> s, vector<ObjectState> a, 
// 								vector<ObjectState> sprime, vector<ObjectState> sappear, float r)
// {
// 	CPyObject updateModel = PyObject_GetAttrString(model_, "updateModel");

// 	if(updateModel && PyCallable_Check(updateModel))
// 	{
// 		PyObject_CallFunction(updateModel, "[i][i][i][i]f", s, a, sprime, sappear, r);
// 	}
// 	else 
// 	{
// 		cout << "ERROR: updateModel function" << endl;
// 	}
// }

#endif
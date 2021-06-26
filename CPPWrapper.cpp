#include <stdio.h>
#include <iostream>
#include "CPPWrapper.hpp"
#include <list>
#include "States.cpp"

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
                obj_in_len(nullptr),
                env_len(nullptr),
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
    PyObject *pModule = PyImport_ImportModule("src.nn_prediction_model");


	// if the module exists
	if (pModule)
	{
		// get PredictionModel object
		CPyObject class_ = PyObject_GetAttrString(pModule, "PredictionModel");
		CPyObject args = Py_BuildValue("[c][c]bb[c][c]iii",
									exist_ckpt_path,
									appear_ckpt_path,
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
tuple<vector<states>, vector<states>, vector<states>> CPPWrapper::predict(vector<states> s, vector<states> a) 
{
	CPyObject predict = PyObject_GetAttrString(model_, "predict");

	// if class exists and can be called
	if(predict && PyCallable_Check(predict))
	{	
		CPyObject pyTuple = PyObject_CallFunction(predict, "[[i]][i]", s, a);

		if (PyTuple_Check(pyTuple)) 
		{
			Py_ssize_t s_pos = 0;
			Py_ssize_t sprimepos = 1;
			Py_ssize_t sappearpos = 2;
			vector<int> s_ = PyTuple_GetItem(pyTuple, s_pos);
			vector<int> sprime = PyTuple_GetItem(pyTuple, sprimepos);
			vector<int> sappear = PyTuple_GetItem(pyTuple, sappearpos);

			return make_tuple(s_, sprime, sappear)
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

void CPPWrapper::updateModel(vector<int> s, vector<int> a, vector<int> sprime, vector<int> sappear, float r)
{
	CPyObject updateModel = PyObject_GetAttrString(model_, "updateModel");

	if(updateModel && PyCallable_Check(updateModel))
	{
		PyObject_CallFunction(updateModel, "[i][i][i][i]f", s, a, sprime, sappear, r);
	}
	else 
	{
		cout << "ERROR: updateModel function" << endl;
	}
}

#endif
#include <stdio.h>
#include <iostream>
#include "CPPWrapper.hpp"
#include <list>

#ifdef PATH
#define PATH_ PATH
#else
#define PATH_ "set_agent/" // Default path


using namespace std;

CPPWrapper::CPPWrapper(string exist_ckpt_path,
						string appear_ckpt_path,
						bool train_exist,
						bool train_appear,
						size_t obj_in_len,
						size_t env_len,
						size_t obj_reg_len,
						size_t obj_attri_len,
						size_t new_set_size) : 
				exist_ckpt_path(exist_ckpt_path),
                appear_ckpt_path(appear_ckpt_path),
                train_exist(train_exist),
                train_appear(train_appear),
                obj_in_len(obj_in_len),
                env_len(env_len),
                obj_reg_len(obj_reg_len),
                obj_attri_len(obj_attri_len),
                new_set_size(new_set_size)
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

		const char* exist_path; 
		const char* appear_path;
		if (exist_ckpt_path.empty())
		{
			exist_path = nullptr;
		}
		else
		{
			exist_path = exist_ckpt_path.c_str();
		}
		if (appear_ckpt_path.empty())
		{
			appear_path = nullptr;
		}
		else
		{
			appear_path = appear_ckpt_path.c_str();
		}

		CPyObject args = Py_BuildValue("(ssOOiiiii)",
									exist_path,
									appear_path,
									train_exist ? Py_True : Py_False,
									train_appear? Py_True : Py_False,
									obj_in_len,
									env_len,
									obj_reg_len,
									obj_attri_len,
									new_set_size);
		model_ = PyObject_CallObject(class_, args);
		if (model_ == NULL) 
		{
			PyErr_Print();
		}
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

// Calls the predict function in Python model
tuple<vector<vector<float>>, vector<vector<float>>, vector<vector<float>>> CPPWrapper::predict(const vector<vector<float>>& s, const vector<float>& a)
{
	CPyObject predict = PyObject_GetAttrString(model_, "predict");
	if (predict == NULL)
	{
		PyErr_Print();
	}
	// CPyObject update = PyObject_GetAttrString(model_, "updateModel");

	// TEST FOR ATTR STRING
	// string x = PyBytes_AS_STRING(PyUnicode_AsEncodedString(PyObject_Repr(predict), "utf-8", "~E~"));
	// string y = PyBytes_AS_STRING(PyUnicode_AsEncodedString(PyObject_Repr(update), "utf-8", "~E~"));
	// cout << x << endl;
	// cout << y << endl;

	// if class exists and can be called
	if (predict && PyCallable_Check(predict))
	{
		PyObject *set = vectorToPyTuple(s);

		Py_ssize_t actionSize = a.size();
		PyObject *action = PyTuple_New(actionSize);
		for (Py_ssize_t i = 0; i < actionSize; i++) {
			PyTuple_SET_ITEM(action, i, PyFloat_FromDouble(a.at(i)));
		}

		CPyObject pyTuple = PyObject_CallFunction(predict, "(OO)", set, action);

		cout << "called predict function in python..." << endl;

		if (pyTuple == NULL) 
		{
			PyErr_Print();
		}
		else if (PyTuple_Check(pyTuple)) 
		{
			Py_ssize_t s_pos = 0;
			Py_ssize_t sprimepos = 1;
			Py_ssize_t sappearpos = 2;

			vector<vector<float>> s_ = pyToVector(PyTuple_GetItem(pyTuple, s_pos));
			vector<vector<float>> sprime = pyToVector(PyTuple_GetItem(pyTuple, sprimepos));
			vector<vector<float>> sappear = pyToVector(PyTuple_GetItem(pyTuple, sappearpos));
			return make_tuple(s_, sprime, sappear);
		}
		else
		{
			cout << "ERROR: returned a nontuple" << endl;
		}

		decref(set);
		Py_DECREF(action);
	}
	else
	{
		cout << "ERROR: predict function" << endl;
	}
}

// converts a Python tuple object to 2D vector
vector<vector<float>> CPPWrapper::pyToVector(PyObject* incoming) {
	Py_ssize_t idx = 0;
	Py_ssize_t length = PyList_Size(incoming);
	Py_ssize_t height = PyList_Size(PyList_GetItem(incoming, idx));
	vector<vector<float>> output;
	
	if (PyList_Check(incoming)) {
		for (Py_ssize_t i = 0; i < length; i++) {
			CPyObject item = PyList_GetItem(incoming, i);
			vector<float> obj;
			
			for (Py_ssize_t j = 0; j < height; j++) {
				obj.push_back(PyFloat_AsDouble(PyList_GetItem(item, j)));
			}

			output.push_back(obj);
		}
	} else {
		throw logic_error("Passed PyObject pointer was not a list array...");
	}

	//Py_DECREF(incoming);
	return output;
}

// Calls the updateModel function in Python model
void CPPWrapper::updateModel(vector<vector<float>> s_, vector<float> a_, 
								vector<vector<float>> sprime_, vector<vector<float>> sappear_, float r)
{
	CPyObject updateModel = PyObject_GetAttrString(model_, "updateModel");

	if (updateModel && PyCallable_Check(updateModel))
	{
		string y = PyBytes_AS_STRING(PyUnicode_AsEncodedString(PyObject_Repr(updateModel), "utf-8", "~E~"));
		cout << y << endl;

		PyObject *s = vectorToPyTuple(s_);
		PyObject *sprime = vectorToPyTuple(sprime_);
		PyObject *sappear = vectorToPyTuple(sappear_);

		Py_ssize_t actionSize = a_.size();
		PyObject *action = PyTuple_New(actionSize);
		for (Py_ssize_t i = 0; i < actionSize; i++) {
			PyTuple_SET_ITEM(action, i, PyFloat_FromDouble(a_.at(i)));
		}

		PyObject_CallFunction(updateModel, "(OOOOf)", s, action, sprime, sappear, r);
		
		// deallocate
		// Py_DECREF(action);
		// decref(s);
		// decref(sprime);
		// decref(sappear);

		cout << "Updated model successfully!" << endl;
	}
	else 
	{
		cout << "ERROR: updateModel function" << endl;
	}
}

// Converts a given 2D vector to Python tuple object
PyObject* CPPWrapper::vectorToPyTuple(const vector<vector<float>>& s)
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

	return set;
}

// Deallocates a given 2D Python list object
void CPPWrapper::decref(PyObject*& doublevector)
{
	for (Py_ssize_t i = 0; i < PyList_Size(doublevector); i++) {
		Py_DECREF(PyList_GetItem(doublevector, i));
	}
	Py_DECREF(doublevector);
}

#endif
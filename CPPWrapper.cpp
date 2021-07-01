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
				exist_ckpt_path(""),
                appear_ckpt_path(""),
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

tuple<vector<vector<float>>, vector<vector<float>>, vector<vector<float>>> CPPWrapper::predict(const vector<vector<float>>& s, const vector<float>& a)
{
	cout << "into the wrapper predict function..." << endl;
	CPyObject predict = PyObject_GetAttrString(model_, "predict");

	// test
	// string x = PyBytes_AS_STRING(PyUnicode_AsEncodedString(PyObject_Repr(predict), "utf-8", "~E~"));
	cout << 3 << endl;

	// if class exists and can be called
	if (predict && PyCallable_Check(predict))
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

		Py_ssize_t actionSize = a.size();
		PyObject *action = PyTuple_New(actionSize);
		for (Py_ssize_t i = 0; i < row; i++) {
			PyTuple_SET_ITEM(action, i, PyFloat_FromDouble(a.at(i)));
		}

		CPyObject pyTuple = PyObject_CallFunction(predict, "(OO)", set, action);

		cout << "called predict function in python..." << endl;

		if (PyTuple_Check(pyTuple)) 
		{
			Py_ssize_t s_pos = 0;
			Py_ssize_t sprimepos = 1;
			Py_ssize_t sappearpos = 2;

			// vector<vector<float>> s_ = pyToCppArray(PyTuple_GetItem(pyTuple, s_pos));
			// vector<vector<float>> sprime = pyToCppArray(PyTuple_GetItem(pyTuple, sprimepos));
			// vector<vector<float>> sappear = pyToCppArray(PyTuple_GetItem(pyTuple, sappearpos));
			vector<vector<float>> s_ = pyToVector(PyTuple_GetItem(pyTuple, s_pos));
			vector<vector<float>> sprime = pyToVector(PyTuple_GetItem(pyTuple, sprimepos));
			vector<vector<float>> sappear = pyToVector(PyTuple_GetItem(pyTuple, sappearpos));
			Py_DECREF(set);
			Py_DECREF(action);
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
				obj.push_back(PyLong_AsLong(PyList_GetItem(item, j)));
			}

			output.push_back(obj);
		}
	} else {
		throw logic_error("Passed PyObject pointer was not a list array...");
	}

	return output;
}

//////////////////////////////////////////////////////////////////////////////////////

// tuple<float**, float**, float**> CPPWrapper::predict(float** s, float* a)
// {
// 	cout << "into the wrapper predict function..." << endl;
// 	CPyObject predict = PyObject_GetAttrString(model_, "predict");

// 	// test
// 	string x = PyBytes_AS_STRING(PyUnicode_AsEncodedString(PyObject_Repr(predict), "utf-8", "~E~"));
// 	cout << "Object name is " << x << endl;

// 	// if class exists and can be called
// 	if (predict && PyCallable_Check(predict))
// 	{
// 		CPyObject pyTuple = PyObject_CallFunction(predict, "[[f]][f]", s, a);

// 		cout << "called predict function in python..." << endl;

// 		if (PyTuple_Check(pyTuple)) 
// 		{
// 			Py_ssize_t s_pos = 0;
// 			Py_ssize_t sprimepos = 1;
// 			Py_ssize_t sappearpos = 2;

// 			float** s_ = pyToCppArray(PyTuple_GetItem(pyTuple, s_pos));
// 			float** sprime = pyToCppArray(PyTuple_GetItem(pyTuple, sprimepos));
// 			float** sappear = pyToCppArray(PyTuple_GetItem(pyTuple, sappearpos));
// 			return make_tuple(s_, sprime, sappear);
// 		}
// 		else
// 		{
// 			cout << "ERROR: returned a nontuple" << endl;
// 		}
// 	}
// 	else
// 	{
// 		cout << "ERROR: predict function" << endl;
// 	}
// }

// PyObject (set prediction) -> set 
// float** CPPWrapper::pyToCppArray(PyObject* incoming) {
// 	Py_ssize_t idx = 0;
// 	int length = PyList_Size(incoming);
// 	int height = PyList_Size(PyList_GetItem(incoming, idx));
// 	float** output = new float*[length];
// 	for (size_t i = 0; i < 10; i++)
// 	{
// 		output[i] = new float[height];
// 	}
	
// 	size_t x = 0;
// 	if (PyList_Check(incoming)) {
// 		for (Py_ssize_t i = 0; i < length; i++) {
// 			CPyObject item = PyList_GetItem(incoming, i);

// 			size_t y = 0;
			
// 			for (Py_ssize_t j = 0; j < height; j++) {
// 				CPyObject value = PyList_GetItem(incoming, j);
// 				*(*(output+i)+y) = PyFloat_AsDouble(value);
// 				y++;
// 			}

// 			x++;
// 		}
// 	} else {
// 		throw logic_error("Passed PyObject pointer was not a list array...");
// 	}

// 	float** result = output;

// 	return result;
// }

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




// static PyArrayObject* vector_to_nparray(const vector< vector<T> >& vec, int type_num = PyArray_FLOAT){

//    // rows not empty
//    if( !vec.empty() ){

//       // column not empty
//       if( !vec[0].empty() ){

//         size_t nRows = vec.size();
//         size_t nCols = vec[0].size();
//         npy_intp dims[2] = {nRows, nCols};
//         PyArrayObject* vec_array = (PyArrayObject *) PyArray_SimpleNew(2, dims, type_num);

//         T *vec_array_pointer = (T*) PyArray_DATA(vec_array);

//         // copy vector line by line ... maybe could be done at one
//         for (size_t iRow=0; iRow < vec.size(); ++iRow){

//           if( vec[iRow].size() != nCols){
//              Py_DECREF(vec_array); // delete
//              throw(string("Can not convert vector<vector<T>> to np.array, since c++ matrix shape is not uniform."));
//           }

//           copy(vec[iRow].begin(),vec[iRow].end(),vec_array_pointer+iRow*nCols);
//         }

//         return vec_array;

//      // Empty columns
//      } else {
//         npy_intp dims[2] = {vec.size(), 0};
//         return (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_FLOAT, 0);
//      }


//    // no data at all
//    } else {
//       npy_intp dims[2] = {0, 0};
//       return (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_FLOAT, 0);
//    }

// }

#endif
#include <stdio.h>
#include <iostream>
#include "MinAtarInterface.hpp"
#include <list>

#ifdef PATH
#define PATH_ PATH
#else
#define PATH_ "set_agent/" // Default path

// set input, action vector for predict are from the training loop

using namespace std;

MinAtarInterface::MinAtarInterface(string path) : path_(path)
{

	// Initialize Python environment
	Py_Initialize();

	// path to directory of module
	const char *scriptDirectoryName = "../dspn";
	PyObject *sysPath = PySys_GetObject("path");
    PyObject *path = PyUnicode_FromString(scriptDirectoryName);
    int result = PyList_Insert(sysPath, 0, path);
	// module name
    PyObject *pModule = PyImport_ImportModule("SetDSPN");


	// if the module exists
	if (pModule)
	{
		// get class object
		CPyObject class_ = PyObject_GetAttrString(pModule, "SetDSPN");
		int* list_ = &data_[0];
		CPyObject args = Py_BuildValue("[i]i", list_, size_);
		model_ = PyObject_CallObject(class_, args);

		// get model size
		CPyObject model_size = PyObject_CallMethod(model_, "getSize", nullptr);
		int size = PyLong_AsLong(model_size);
		cout << "Size of model is " << size << endl;

		CPyObject add_data = PyObject_GetAttrString(model_, "addData");
		

		// if class exists and can be called
		if(add_data && PyCallable_Check(add_data))
		{	
			PyObject_CallFunction(add_data, "i", 2);
			cout << "Added data value 2 to Model" << endl;
		}
		else
		{
			cout << "ERROR: addData function" << endl;
		}

	}
	else
	{
		cout << "ERROR: Module not imported" << endl;
	}
}

// destructor
MinAtarInterface::~MinAtarInterface() {
	model_.release();
	Py_Finalize();
}

// NEED TO EDIT FOR PARAMETERS s,a
void MinAtarInterface::predict() {
	CPyObject add_data = PyObject_GetAttrString(model_, "addData");

	// if class exists and can be called
	if(add_data && PyCallable_Check(add_data))
	{	
		PyObject_CallFunction(add_data, "i", n);
		cout << "Added data value " << n << " to Model" << endl;

		CPyObject pyStr = PyObject_CallMethod(model_, "print", nullptr);
		CPyObject pyBytes = PyUnicode_AsEncodedString(pyStr.getObject(), "UTF-8", "strict");
		string printStr(PyBytes_AsString(pyBytes));
		cout << printStr << endl;
	}
	else
	{
		cout << "ERROR: addData or print function" << endl;
	}
}

// copy constructor
// Class::Class(const Class& other) {
//     *this = other;
// }

#endif
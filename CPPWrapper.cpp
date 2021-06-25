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

CPPWrapper::CPPWrapper(string path) : path_(path)
{

	// Initialize Python environment
	Py_Initialize();

	// path to directory of module
	const char *scriptDirectoryName = "./";
	PyObject *sysPath = PySys_GetObject("path");
    PyObject *path = PyUnicode_FromString(scriptDirectoryName);
    int result = PyList_Insert(sysPath, 0, path);
	// module name
    PyObject *pModule = PyImport_ImportModule("src.SetDSPN");


	// if the module exists
	if (pModule)
	{
		// get class object
		CPyObject class_ = PyObject_GetAttrString(pModule, "SetDSPN");
		model_ = PyObject_CallMethod(class_, "load_from_checkpoint", path_);
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
	CPyObject predict = PyObject_GetAttrString(model_, "forward");

	// if class exists and can be called
	if(add_data && PyCallable_Check(predict))
	{	
		PyObject_CallFunction(predict, "i", n);
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
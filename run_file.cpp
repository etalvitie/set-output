#include <stdio.h>
#include <iostream>
#include "CPyObject.hpp"

#ifdef PATH
#define PATH_ PATH
#else
#define PATH_ "set_agent/" // Default path
#endif

// parameters: take in set input and an action vector to produce prediction (forward function in setDSPN.py)

using namespace std;

int main()
{

	// Initialize Python environment
	Py_Initialize();


	const char *scriptDirectoryName = "./";
	PyObject *sysPath = PySys_GetObject("path");
    PyObject *path = PyUnicode_FromString(scriptDirectoryName);
    int result = PyList_Insert(sysPath, 0, path);
	// module name
    PyObject *pModule = PyImport_ImportModule("src.set_agent.minatar_dspn_model");


	// if the module exists
	if (pModule)
	{
		// get function attribute object 
		// first argument = module being used
		// second argument = name of function being called
		CPyObject pFunc = PyObject_GetAttrString(pModule, "evaluate");

		// if function exists and can be called
		if(pFunc && PyCallable_Check(pFunc))
		{	
			// call Python function
			// first argument = function object in use
			// second argument = arguments passed to the Python method (NULL if none)
			CPyObject status = PyObject_CallObject(pFunc, NULL);
			cout << PyLong_AsLong(status) << endl;
		}
		else
		{
			cout << "ERROR: function evaluate" << endl;
		}

	}
	else
	{
		cout << "ERROR: Module not imported" << endl;
		PyErr_Print();
	}
	
	Py_Finalize();

	return 0;
}
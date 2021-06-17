#include <stdio.h>
#include <iostream>
#include "CPyObject.hpp"

#ifdef PATH
#define PATH_ PATH
#else
#define PATH_ "set_agent/" // Default path
#endif

// parameters: take in set input and an action vector to produce prediction (forward function in setDSPN.py)

int main()
{
	char filename[] = "minatar_dspn_model.py";
	FILE* fp;

	// Initialize Python environment
	Py_Initialize();

	// attempt to open file then run it
	fp = _Py_fopen(filename, "r");
	PyRun_SimpleFile(fp, filename);

    return 0;
}
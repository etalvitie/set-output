#ifndef PYINTERFACE_HPP
#define PYINTERFACE_HPP

// Credit: https://www.codeproject.com/Articles/820116/Embedding-Python-program-in-a-C-Cplusplus-code
// License: https://www.codeproject.com/info/cpol10.aspx

#include <Python.h>

#include <string>
#include <tuple>
#include <vector>
// #include <curses.h>

class CPyObject {
  private:
   PyObject* p_;
   
  public:
   CPyObject() : p_{NULL} {}
   
   CPyObject(PyObject* p) : p_{p} {}

   ~CPyObject() {
      release();
   }
   
   PyObject* getObject() {
      return p_;
   }
   
   PyObject* setObject(PyObject* p) {
      return (p_ = p);
   }
   
   PyObject* addRef() {
      if(p_) {
	 Py_INCREF(p_);
      }
      return p_;
   }
   
   void release() {
      if(p_) {
	 Py_DECREF(p_);
      }

      p_ = NULL;
   }

   PyObject* operator ->() {
      return p_;
   }

   bool is() {
      return p_ ? true : false;
   }

   operator PyObject*() {
      return p_;
   }

   PyObject* operator =(PyObject* pp) {
      p_ = pp;
      return p_;
   }

   operator bool() {
      return p_ ? true : false;
   }
};

#endif
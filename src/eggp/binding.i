%module binding
%{
#include "EGGP_stub.h"

char * unsafe_hs_eggp_version() {
  return hs_eggp_version();
}

int unsafe_hs_eggp_main() {
  return hs_eggp_main();
}

char * unsafe_hs_eggp_run( char *dataset, int gens, int nPop, int maxSize, int nTournament, double pc, double pm,  char *nonterminals,  char *loss, int optIter, int optRepeat, int nParams, int split, int max_time, int simplify, int trace, int generational, char *dumpTo,  char *loadFrom, char *varnames ) {
  return hs_eggp_run(dataset, gens, nPop, maxSize, nTournament, pc, pm, nonterminals, loss, optIter, optRepeat, nParams, split, max_time, simplify, trace, generational, dumpTo, loadFrom, varnames);
}

void unsafe_hs_eggp_init(int argc, char **argv) {
  hs_init(&argc, &argv);
}

void unsafe_hs_eggp_exit() {
  hs_exit();
}

void unsafe_py_write_stdout( char * str) {
  PySys_FormatStdout("%s", str);
}

void unsafe_py_write_stderr( char * str) {
  PySys_FormatStderr("%s", str);
}
%}

%typemap(in) (int argc, char **argv) {
  /* Check if is a list */
  if (PyList_Check($input)) {
    int i;
    $1 = PyList_Size($input);
    $2 = (char **) malloc(($1+1)*sizeof(char *));
    for (i = 0; i < $1; i++) {
      PyObject *o = PyList_GetItem($input, i);
      if (PyUnicode_Check(o)) {
        $2[i] = (char *) PyUnicode_AsUTF8AndSize(o, 0);
      } else {
        PyErr_SetString(PyExc_TypeError, "list must contain strings");
        SWIG_fail;
      }
    }
    $2[i] = 0;
  } else {
    PyErr_SetString(PyExc_TypeError, "not a list");
    SWIG_fail;
  }
}

%typemap(freearg) (int argc, char **argv) {
  free((char *) $2);
}

char * unsafe_hs_eggp_version();
int unsafe_hs_eggp_main();
char * unsafe_hs_eggp_run( char *dataset, int gens, int nPop, int maxSize, int nTournament, double p, double pm,  char *nonterminals,  char *loss, int optIter, int optRepeat, int nParams, int split, int max_time, int simplify,  int trace, int generational, char *dumpTo,  char *loadFrom, char *varnames);
void unsafe_hs_eggp_init(int argc, char **argv);
void unsafe_hs_eggp_exit();

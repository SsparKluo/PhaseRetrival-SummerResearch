//
// MATLAB Compiler: 7.0.1 (R2019a)
// Date: Tue Jun 25 22:27:26 2019
// Arguments:
// "-B""macro_default""-B""cpplib:myUnwrap""-W""cpplib:myUnwrap""-T""link:lib""m
// yUnwrap.m"
//

#ifndef __myUnwrap_h
#define __myUnwrap_h 1

#if defined(__cplusplus) && !defined(mclmcrrt_h) && defined(__linux__)
#  pragma implementation "mclmcrrt.h"
#endif
#include "mclmcrrt.h"
#include "mclcppclass.h"
#ifdef __cplusplus
extern "C" {
#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_myUnwrap_C_API 
#define LIB_myUnwrap_C_API /* No special import/export declaration */
#endif

/* GENERAL LIBRARY FUNCTIONS -- START */

extern LIB_myUnwrap_C_API 
bool MW_CALL_CONV myUnwrapInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_myUnwrap_C_API 
bool MW_CALL_CONV myUnwrapInitialize(void);

extern LIB_myUnwrap_C_API 
void MW_CALL_CONV myUnwrapTerminate(void);

extern LIB_myUnwrap_C_API 
void MW_CALL_CONV myUnwrapPrintStackTrace(void);

/* GENERAL LIBRARY FUNCTIONS -- END */

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_myUnwrap_C_API 
bool MW_CALL_CONV mlxMyUnwrap(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

#ifdef __cplusplus
}
#endif


/* C++ INTERFACE -- WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

#ifdef __cplusplus

/* On Windows, use __declspec to control the exported API */
#if defined(_MSC_VER) || defined(__MINGW64__)

#ifdef EXPORTING_myUnwrap
#define PUBLIC_myUnwrap_CPP_API __declspec(dllexport)
#else
#define PUBLIC_myUnwrap_CPP_API __declspec(dllimport)
#endif

#define LIB_myUnwrap_CPP_API PUBLIC_myUnwrap_CPP_API

#else

#if !defined(LIB_myUnwrap_CPP_API)
#if defined(LIB_myUnwrap_C_API)
#define LIB_myUnwrap_CPP_API LIB_myUnwrap_C_API
#else
#define LIB_myUnwrap_CPP_API /* empty! */ 
#endif
#endif

#endif

extern LIB_myUnwrap_CPP_API void MW_CALL_CONV myUnwrap(int nargout, mwArray& phaseImage, const mwArray& input);

/* C++ INTERFACE -- WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */
#endif

#endif

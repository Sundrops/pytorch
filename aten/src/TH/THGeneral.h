#ifndef TH_GENERAL_INC
#define TH_GENERAL_INC

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <time.h>
#include <string.h>
#include <stddef.h>
#include <inttypes.h>

#cmakedefine USE_BLAS
#cmakedefine USE_LAPACK
#cmakedefine BLAS_F2C
#cmakedefine BLAS_USE_CBLAS_DOT

#ifdef __cplusplus
# define TH_EXTERNC extern "C"
#else
# define TH_EXTERNC extern
#endif

#ifdef _WIN32
# ifdef TH_EXPORTS
// 关于 __declsec(dllexport) 的解释 http://www.cppblog.com/Dutyboy/archive/2010/11/15/133699.html
#  define TH_API TH_EXTERNC __declspec(dllexport)
# else
#  define TH_API TH_EXTERNC __declspec(dllimport)
# endif
#else
# define TH_API TH_EXTERNC
#endif

#ifndef M_PI
# define M_PI 3.14159265358979323846
#endif

#ifndef TH_INDEX_BASE
#define TH_INDEX_BASE 1
#endif

typedef void (*THErrorHandlerFunction)(const char *msg, void *data);
typedef void (*THArgErrorHandlerFunction)(int argNumber, const char *msg, void *data);

#define TH_DESC_BUFF_LEN 64
typedef struct {
    char str[TH_DESC_BUFF_LEN];
} THDescBuff;

// TH_API 声明导出函数， extern
TH_API double THLog1p(const double x);
TH_API THDescBuff _THSizeDesc(const int64_t *size, const int64_t ndim);
TH_API void _THError(const char *file, const int line, const char *fmt, ...);
TH_API void _THAssertionFailed(const char *file, const int line, const char *exp, const char *fmt, ...);
TH_API void THSetErrorHandler(THErrorHandlerFunction new_handler, void *data);
TH_API void THSetDefaultErrorHandler(THErrorHandlerFunction new_handler, void *data);
TH_API void _THArgCheck(const char *file, int line, int condition, int argNumber, const char *fmt, ...);
TH_API void THSetArgErrorHandler(THArgErrorHandlerFunction new_handler, void *data);
TH_API void THSetDefaultArgErrorHandler(THArgErrorHandlerFunction new_handler, void *data);
TH_API void* THAlloc(ptrdiff_t size);
TH_API void* THRealloc(void *ptr, ptrdiff_t size);
TH_API void THFree(void *ptr);
TH_API void THSetGCHandler( void (*torchGCHandlerFunction)(void *data), void *data );
// this hook should only be called by custom allocator functions
TH_API void THHeapUpdate(ptrdiff_t size);
TH_API void THSetNumThreads(int num_threads);
TH_API int THGetNumThreads(void);
TH_API int THGetNumCores(void);
TH_API void THInferNumThreads(void);

#define THError(...) _THError(__FILE__, __LINE__, __VA_ARGS__)

#define THCleanup(...) __VA_ARGS__

// THArgCheck(condition, argNumber, char*) 
// condition：条件表达式， argNumber ？？？。 char* 报错信息。
#define THArgCheck(...)                                               \
do {                                                                  \
  _THArgCheck(__FILE__, __LINE__, __VA_ARGS__);                       \
} while(0)

#define THArgCheckWithCleanup(condition, cleanup, ...)                \
do if (!(condition)) {                                                \
  cleanup                                                             \
  _THArgCheck(__FILE__, __LINE__, 0, __VA_ARGS__);                    \
} while(0)

// 断言，如果 exp 为 True， 则通过，否则 报错。 
// _THAssertionFailed 是用来报错 和 打印错误信息的。
#define THAssert(exp)                                                 \
do {                                                                  \
  if (!(exp)) {                                                       \
    _THAssertionFailed(__FILE__, __LINE__, #exp, "");                 \
  }                                                                   \
} while(0)

#define THAssertMsg(exp, ...)                                         \
do {                                                                  \
  if (!(exp)) {                                                       \
    _THAssertionFailed(__FILE__, __LINE__, #exp, __VA_ARGS__);        \
  }                                                                   \
} while(0)

#define TH_CONCAT_STRING_2(x,y) TH_CONCAT_STRING_2_EXPAND(x,y)
#define TH_CONCAT_STRING_2_EXPAND(x,y) #x #y

#define TH_CONCAT_STRING_3(x,y,z) TH_CONCAT_STRING_3_EXPAND(x,y,z)
#define TH_CONCAT_STRING_3_EXPAND(x,y,z) #x #y #z

#define TH_CONCAT_STRING_4(x,y,z,w) TH_CONCAT_STRING_4_EXPAND(x,y,z,w)
#define TH_CONCAT_STRING_4_EXPAND(x,y,z,w) #x #y #z #w

#define TH_CONCAT_2(x,y) TH_CONCAT_2_EXPAND(x,y)
#define TH_CONCAT_2_EXPAND(x,y) x ## y

// CONCAT 生成 TOKEN 的

#define TH_CONCAT_3(x,y,z) TH_CONCAT_3_EXPAND(x,y,z)
#define TH_CONCAT_3_EXPAND(x,y,z) x ## y ## z

#define TH_CONCAT_4_EXPAND(x,y,z,w) x ## y ## z ## w
#define TH_CONCAT_4(x,y,z,w) TH_CONCAT_4_EXPAND(x,y,z,w)

#define THMin(X, Y)  ((X) < (Y) ? (X) : (Y))
#define THMax(X, Y)  ((X) > (Y) ? (X) : (Y))

#if (defined(_MSC_VER) || defined(__MINGW32__))
#if defined(_MSC_VER)
__inline double log1p(double x) { return THLog1p(x); }
#else
inline double log1p(double x) { return THLog1p(x); }
#endif

#define snprintf _snprintf
#define popen _popen
#define pclose _pclose
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#endif

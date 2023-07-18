#include "common.h"

LIBRARY_API
void freeIntArray(int* ptr)
{
    if (ptr != nullptr) {
        delete [] ptr;
    }
}

LIBRARY_API
void freeDoubleArray(double* ptr)
{
    if (ptr != nullptr) {
        delete [] ptr;
    }
}

LIBRARY_API
void freeBoolArray(bool* ptr)
{
    if (ptr != nullptr) {
        delete [] ptr;
    }
}

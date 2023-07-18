#include "common.h"


struct DoubleResult
{
	char* error = nullptr;
	double value = 0.0;
};


LIBRARY_API
DoubleResult get_suggested_range(int n, int dimensions, int divisions, int k = 3);

#include "get_suggested_range.h"


#include <cmath>

#include <algorithm>

using std::pow;
using std::min;
using std::max;


const double REASONABLE_RANGE = 0.25;


LIBRARY_API
DoubleResult get_suggested_range(int n, int dimensions, int divisions, int k) {
    DoubleResult result;

    double ksi = pow((double)k / n, 1.0 / dimensions);
    double suggested_range = (1.0 - ksi * (1 + divisions)) / (1.0 - ksi * (1 - divisions));
    result.value = max(0.0, min(suggested_range, 1.0));

    if (result.value == 0) {
        result.error = (char*) "Too small sample for the test";
    } else if (result.value < REASONABLE_RANGE) {
        result.error = (char*) "Too small sample for multiple discretizations";
    }

    return result;
}

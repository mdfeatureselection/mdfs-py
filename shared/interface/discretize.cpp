#include "common.h"

#include "../src/cpu/discretize.h"

#include <algorithm>
#include <memory>


struct D_Result
{
	char* error = nullptr;
	int n_objs = 0;
	int* discretized_var = nullptr;
};


LIBRARY_API
D_Result discretize(
    int obj_count,
    double* variable,
    int variable_idx,
    int divisions,
    int discretization_nr,
    int seed,
    double range
)
{
    D_Result result;

    if(divisions < 1 || divisions > 15)
    {
        result.error = (char*) "Divisions has to be an integer between 1 and 15 (inclusive).";
        return result;
    }

    if(discretization_nr < 1)
    {
        result.error = (char*) "discretization_nr has to be a positive integer.";
        return result;
    }

    if(range < 0.0 || range > 1.0)
    {
        result.error = (char*) "range has to be a number between 0.0 and 1.0 (inclusive)";
        return result;
    }

    if(seed < 0 || seed > 2147483647)
    {
        result.error = (char*) "Only integer seeds from 0 to 2^31-1 are portable. Using non-portable seed may make result harder to reproduce.";
        return result;
    }

    std::vector<double> sorted_variable(variable, variable + obj_count);
    std::sort(sorted_variable.begin(), sorted_variable.end());

    result.discretized_var = new int[obj_count];
    std::unique_ptr<uint8_t> discretized_variable(new uint8_t[obj_count]);
    discretize(seed, discretization_nr, variable_idx, divisions, obj_count, variable, sorted_variable, discretized_variable.get(), range);
    std::copy(discretized_variable.get(), discretized_variable.get() + obj_count, result.discretized_var);

    result.n_objs = obj_count;

    return result;
}

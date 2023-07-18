#include "common.h"

#include "get_suggested_range.h"

#include "../src/cpu/mdfs.h"

#ifdef WITH_CUDA
#include "gpu/cucubes.h"
#endif

#include <cmath>
#include <chrono>
#include <random>

using std::pow;
using std::min;
using std::max;


struct CMIG_Result
{
	char* error = nullptr;
	int dimensions = 0;
	int n_max_igs = 0;
	double* max_igs = nullptr;
	int n_max_igs_contrast = 0;
	double* max_igs_contrast = nullptr;
	int* tuples = nullptr;
	int* dids = nullptr;
};


LIBRARY_API
CMIG_Result compute_max_ig(
    int obj_count,
    int var_count,
    double* data,
    int n_contrast_variables,
    double* contrast_data,
    int* decision,
    int dimensions,
    int divisions,
    int discretizations,
    int seed,
    double range,
    double pseudocount,
    bool return_tuples,
    int int_var_count,
    int* interesting_vars,
    bool require_all_vars,
    bool use_CUDA
)
{
    CMIG_Result result;

    int true_count = 0;
    int false_count = 0;

    for(auto i = decision; i < decision + obj_count; i++)
    {
        if(*i == 0)
        {
            false_count++;
        }
        else if(*i == 1)
        {
            true_count++;
        }
        else
        {
            result.error = (char*) "Decision must be binary.";
            return result;
        }
    }

    if(!true_count || !false_count)
    {
        result.error = (char*) "Both classes have to be represented.";
        return result;
    }

    if(dimensions < 1)
    {
        result.error = (char*) "Dimensions has to be a positive integer.";
        return result;
    }

    if(dimensions > 5)
    {
        result.error = (char*) "Dimensions cannot exceed 5.";
        return result;
    }

    if(divisions < 1 || divisions > 15)
    {
        result.error = (char*) "Divisions has to be an integer between 1 and 15 (inclusive).";
        return result;
    }

    if(discretizations < 1)
    {
        result.error = (char*) "Discretizations has to be a positive integer.";
        return result;
    }

    if(pseudocount <= 0)
    {
        result.error = (char*) "pc_xi has to be a real number strictly greater than 0.";
        return result;
    }

    if(range == -1)
    {
        DoubleResult range_result = get_suggested_range(obj_count, dimensions, divisions);
        if(range_result.error != nullptr)
        {
            result.error = range_result.error;
            return result;
        }
        range = range_result.value;
    }
    else if(range < 0.0 || range > 1.0)
    {
        result.error = (char*) "Range has to be a number between 0.0 and 1.0 (inclusive)";
        return result;
    }

    if (range == 0.0 && discretizations > 1)
    {
        result.error = (char*) "Zero range does not make sense with more than one discretization. All will always be equal.";
        return result;
    }

    if(seed == -1)
    {
        std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());
        seed = generator() / 2; // halving the result to keep it (exactly) in desired range
    } else if(seed < 0 || seed > 2147483647)
    {
        result.error = (char*) "Only integer seeds from 0 to 2^31-1 are portable. Using non-portable seed may make result harder to reproduce.";
        return result;
    }

    if (dimensions == 1 && return_tuples)
    {
        result.error = (char*) "return_tuples does not make sense in 1D.";
        return result;
    }

    if(use_CUDA)
    {
        if(dimensions == 1)
        {
            result.error = (char*) "CUDA acceleration does not support 1 dimension";
            return result;
        }

        if(pow(divisions + 1, dimensions) > 256)
        {
            result.error = (char*) "CUDA acceleration does not support more than 256 cubes = (divisions+1)^dimensions";
            return result;
        }

        if(return_tuples)
        {
            result.error = (char*) "CUDA acceleration does not support return_tuples parameter (for now)";
            return result;
        }

        if(int_var_count > 0)
        {
            result.error = (char*) "CUDA acceleration does not support interesting_vars parameter (for now)";
            return result;
        }
    }

    #ifdef WITH_CUDA
    if (use_CUDA)
    {
        try {
            run_cucubes(
                    obj_count,
                    variable_count,
                    dimensions,
                    divisions,
                    discretizations,
                    seed,
                    range,
                    pseudocount,
                    data,
                    decision,
                    max_igs);
        } catch (const cudaException& e) {
            // TODO: properly handle CUDA exceptions
            return (char*) "CUDA exception";
            // error("CUDA exception: %s (in %s:%d)", cudaGetErrorString(e.code), e.file, e.line);
        } catch (const NotImplementedException& e) {
            // TODO: properly handle Not-implemented exception
            return (char*) "Not-implemented exception";
            // error("Not-implemented exception: %s", e.msg.c_str());
        }

        return NULL;
    }
    #endif

    RawData rawdata(RawDataInfo(obj_count, var_count), data, decision);
    std::unique_ptr<RawData> contrast_rawdata;
    if (contrast_data != nullptr) {
        contrast_rawdata.reset(new RawData(RawDataInfo(obj_count, n_contrast_variables), contrast_data, nullptr));
    }

    std::unique_ptr<const DiscretizationInfo> dfi(new DiscretizationInfo(
        seed,
        discretizations,
        divisions,
        range
    ));

    MDFSInfo mdfs_info(
        dimensions,
        divisions,
        discretizations,
        pseudocount,
        0.0f,
        interesting_vars,
        int_var_count,
        require_all_vars,
        nullptr,
        false
    );

    result.n_max_igs = var_count;
    result.max_igs = new double[var_count];
    if (n_contrast_variables > 0) {
        result.n_max_igs_contrast = n_contrast_variables;
        result.max_igs_contrast = new double[n_contrast_variables];
    }

    MDFSOutput mdfs_output(MDFSOutputType::MaxIGs, dimensions, var_count, n_contrast_variables);
    if(return_tuples)
    {
        result.tuples = new int[dimensions * var_count];
        result.dids = new int[var_count];
        mdfs_output.setMaxIGsTuples(result.tuples, result.dids);
    }

    mdfs[dimensions - 1](mdfs_info, &rawdata, contrast_rawdata.get(), std::move(dfi), mdfs_output);

    mdfs_output.copyMaxIGsAsDouble(result.max_igs);
    if (n_contrast_variables > 0) {
        mdfs_output.copyContrastMaxIGsAsDouble(result.max_igs_contrast);
    }

    result.dimensions = dimensions;

    return result;
}


LIBRARY_API
CMIG_Result compute_max_ig_discrete(
    int obj_count,
    int var_count,
    double* data,
    int n_contrast_variables,
    double* contrast_data,
    int* decision,
    int dimensions,
    int divisions,
    double pseudocount,
    bool return_tuples,
    int int_var_count,
    int* interesting_vars,
    bool require_all_vars
)
{
    CMIG_Result result;

    int true_count = 0;
    int false_count = 0;

    for(auto i = decision; i < decision + obj_count; i++)
    {
        if(*i == 0)
        {
            false_count++;
        }
        else if(*i == 1)
        {
            true_count++;
        }
        else
        {
            result.error = (char*) "Decision must be binary.";
            return result;
        }
    }

    if(!true_count || !false_count)
    {
        result.error = (char*) "Both classes have to be represented.";
        return result;
    }

    if(dimensions < 1)
    {
        result.error = (char*) "Dimensions has to be a positive integer.";
        return result;
    }

    if(dimensions > 5)
    {
        result.error = (char*) "Dimensions cannot exceed 5.";
        return result;
    }

    if(divisions < 1 || divisions > 15)
    {
        result.error = (char*) "Divisions has to be an integer between 1 and 15 (inclusive).";
        return result;
    }

    if(pseudocount <= 0)
    {
        result.error = (char*) "pc_xi has to be a real number strictly greater than 0.";
        return result;
    }

    if (dimensions == 1 && return_tuples)
    {
        result.error = (char*) "return_tuples does not make sense in 1D.";
        return result;
    }

    RawData rawdata(RawDataInfo(obj_count, var_count), data, decision);
    std::unique_ptr<RawData> contrast_rawdata;
    if (contrast_data != nullptr) {
        contrast_rawdata.reset(new RawData(RawDataInfo(obj_count, n_contrast_variables), contrast_data, nullptr));
    }

    MDFSInfo mdfs_info(
        dimensions,
        divisions,
        1,
        pseudocount,
        0.0f,
        interesting_vars,
        int_var_count,
        require_all_vars,
        nullptr,
        false
    );

    result.n_max_igs = var_count;
    result.max_igs = new double[var_count];
    if (n_contrast_variables > 0) {
        result.n_max_igs_contrast = n_contrast_variables;
        result.max_igs_contrast = new double[n_contrast_variables];
    }

    MDFSOutput mdfs_output(MDFSOutputType::MaxIGs, dimensions, var_count, n_contrast_variables);
    if(return_tuples)
    {
        result.tuples = new int[dimensions * var_count];
        result.dids = new int[var_count];
        mdfs_output.setMaxIGsTuples(result.tuples, result.dids);
    }

    mdfs[dimensions - 1](mdfs_info, &rawdata, contrast_rawdata.get(), nullptr, mdfs_output);

    mdfs_output.copyMaxIGsAsDouble(result.max_igs);
    if (n_contrast_variables > 0) {
        mdfs_output.copyContrastMaxIGsAsDouble(result.max_igs_contrast);
    }

    result.dimensions = dimensions;

    return result;
}

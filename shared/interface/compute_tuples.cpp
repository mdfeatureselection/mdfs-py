#include "common.h"

#include "get_suggested_range.h"

#include "../src/cpu/mdfs.h"


#include <cmath>
#include <chrono>
#include <random>

using std::pow;
using std::min;
using std::max;


struct CT_Result
{
	char* error = nullptr;
	int data_count = 0;
	int n_dims = 0;
	double* igs = nullptr;
	int* tuples = nullptr;
	int* vars = nullptr;
};


LIBRARY_API
CT_Result compute_tuples(
    int obj_count,
    int var_count,
    double* data,
    int* decision,
    int dimensions,
    int divisions,
    int discretizations,
    int seed,
    double range,
    double pseudocount,
    int int_vars_count,
    int* interesting_vars,
    bool require_all_vars,
    double ig_thr,
    const double* I_lower,
    bool return_matrix,
    StatMode stat_mode,
    bool average
)
{
    CT_Result result;

    if (decision != nullptr) {
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
    }

    if(dimensions < 2)
    {
        result.error = (char*) "Dimensions has to be at least 2 for this function to make any sense.";
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
    } else if(range < 0.0 || range > 1.0)
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
        seed = generator()/2; // halving the result to keep it (exactly) in desired range
    } else if(seed < 0 || seed > 2147483647)
    {
        result.error = (char*) "Only integer seeds from 0 to 2^31-1 are portable. Using non-portable seed may make result harder to reproduce.";
        return result;
    }

    RawData rawdata(RawDataInfo(obj_count, var_count), data, decision);

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
        ig_thr,
        interesting_vars,
        int_vars_count,
        require_all_vars,
        I_lower,
        average
    );

    MDFSOutputType out_type = mdfs_info.dimensions == 2 && ig_thr <= 0.0 && int_vars_count == 0 ? MDFSOutputType::AllTuples : MDFSOutputType::MatchingTuples;

    MDFSOutput mdfs_output(out_type, dimensions, var_count, 0);

    if (decision == nullptr) {
        switch (stat_mode) {
            case StatMode::Entropy: mdfsEntropy[dimensions - 1](mdfs_info, &rawdata, nullptr, std::move(dfi), mdfs_output);
            break;
            case StatMode::MutualInformation: mdfsMutualInformation[dimensions - 1](mdfs_info, &rawdata, nullptr, std::move(dfi), mdfs_output);
            break;
            case StatMode::VariationOfInformation: mdfsVariationOfInformation[dimensions - 1](mdfs_info, &rawdata, nullptr, std::move(dfi), mdfs_output);
            break;
            default:
                result.error = (char*) "Unknown statistic";
                return result;
        }
    } else {
        switch (stat_mode) {
            case StatMode::Entropy: mdfsDecisionConditionalEntropy[dimensions - 1](mdfs_info, &rawdata, nullptr, std::move(dfi), mdfs_output);
            break;
            case StatMode::MutualInformation: mdfs[dimensions - 1](mdfs_info, &rawdata, nullptr, std::move(dfi), mdfs_output);
            break;
            case StatMode::VariationOfInformation: mdfsDecisionConditionalVariationOfInformation[dimensions - 1](mdfs_info, &rawdata, nullptr, std::move(dfi), mdfs_output);
            break;
            default:
                result.error = (char*) "Unknown statistic";
                return result;
        }
    }

    if (out_type == MDFSOutputType::AllTuples && return_matrix) {
        result.data_count = var_count;

        result.igs = new double[var_count * var_count];

        // TODO: perhaps we could avoid copying here at all and fill in this matrix already from the mdfs?
        mdfs_output.copyAllTuplesMatrix(result.igs);
    } else {
        // 2D only now
        result.data_count = out_type == MDFSOutputType::AllTuples ? var_count * (var_count - 1) : mdfs_output.getMatchingTuplesCount();;

        result.igs = new double[result.data_count];
        result.tuples = new int[result.data_count * dimensions];
        result.vars = new int[result.data_count];

        if (out_type == MDFSOutputType::AllTuples) {
            mdfs_output.copyAllTuples(result.vars, result.igs, result.tuples);
        } else {
            mdfs_output.copyMatchingTuples(result.vars, result.igs, result.tuples);
        }
    }

    result.n_dims = dimensions;

    return result;
}

LIBRARY_API
CT_Result compute_tuples_discrete(
    int obj_count,
    int var_count,
    double* data,
    int* decision,
    int dimensions,
    int divisions,
    double pseudocount,
    int int_vars_count,
    int* interesting_vars,
    bool require_all_vars,
    double ig_thr,
    const double* I_lower,
    bool return_matrix,
    StatMode stat_mode
)
{
    CT_Result result;

    if (decision != nullptr) {
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
    }

    if(dimensions < 2)
    {
        result.error = (char*) "Dimensions has to be at least 2 for this function to make any sense.";
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

    RawData rawdata(RawDataInfo(obj_count, var_count), data, decision);

    MDFSInfo mdfs_info(
        dimensions,
        divisions,
        1,
        pseudocount,
        ig_thr,
        interesting_vars,
        int_vars_count,
        require_all_vars,
        I_lower,
        false
    );

    MDFSOutputType out_type = mdfs_info.dimensions == 2 && ig_thr <= 0.0 && int_vars_count == 0 ? MDFSOutputType::AllTuples : MDFSOutputType::MatchingTuples;

    MDFSOutput mdfs_output(out_type, dimensions, var_count, 0);

    if (decision == nullptr) {
        switch (stat_mode) {
            case StatMode::Entropy: mdfsEntropy[dimensions - 1](mdfs_info, &rawdata, nullptr, nullptr, mdfs_output);
            break;
            case StatMode::MutualInformation: mdfsMutualInformation[dimensions - 1](mdfs_info, &rawdata, nullptr, nullptr, mdfs_output);
            break;
            case StatMode::VariationOfInformation: mdfsVariationOfInformation[dimensions - 1](mdfs_info, &rawdata, nullptr, nullptr, mdfs_output);
            break;
            default:
                result.error = (char*) "Unknown statistic";
                return result;
        }
    } else {
        switch (stat_mode) {
            case StatMode::Entropy: mdfsDecisionConditionalEntropy[dimensions - 1](mdfs_info, &rawdata, nullptr, nullptr, mdfs_output);
            break;
            case StatMode::MutualInformation: mdfs[dimensions - 1](mdfs_info, &rawdata, nullptr, nullptr, mdfs_output);
            break;
            case StatMode::VariationOfInformation: mdfsDecisionConditionalVariationOfInformation[dimensions - 1](mdfs_info, &rawdata, nullptr, nullptr, mdfs_output);
            break;
            default:
                result.error = (char*) "Unknown statistic";
                return result;
        }
    }

    if (out_type == MDFSOutputType::AllTuples && return_matrix) {
        result.data_count = var_count;

        result.igs = new double[var_count * var_count];

        // TODO: perhaps we could avoid copying here at all and fill in this matrix already from the mdfs?
        mdfs_output.copyAllTuplesMatrix(result.igs);
    } else {
        // 2D only now
        result.data_count = out_type == MDFSOutputType::AllTuples ? var_count * (var_count - 1) : mdfs_output.getMatchingTuplesCount();;

        result.igs = new double[result.data_count];
        result.tuples = new int[result.data_count * dimensions];
        result.vars = new int[result.data_count];

        if (out_type == MDFSOutputType::AllTuples) {
            mdfs_output.copyAllTuples(result.vars, result.igs, result.tuples);
        } else {
            mdfs_output.copyMatchingTuples(result.vars, result.igs, result.tuples);
        }
    }

    result.n_dims = dimensions;

    return result;
}

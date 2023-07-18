#include "common.h"

#include <cstddef>
#include <set>
#include <chrono>
#include <random>

// NOTE: We are not using the std::shuffle on purpose - it gives no guarantees
// on the reproducibility of its results.
// NOTE: We are also not using the std::uniform_int_distribution for a similar
// reason - its behaviour differs between implementations.

template<typename T>
void mdfs_sample_to(T* out_indices, T space_size, size_t length, std::mt19937& gen) {
    std::set<T> used_indices;
    for (size_t i = 0; i < length; i++) {
        T source_i = gen() % space_size;
        while (used_indices.count(source_i) > 0) {
            source_i = gen() % space_size;
        }
        used_indices.insert(source_i);
        out_indices[i] = source_i;
    }
}

template<typename T>
void mdfs_shuffle_to(T* to, T* from, size_t length, std::mt19937& gen) {
    std::set<size_t> used_indices;
    for (size_t i = 0; i < length; i++) {
        size_t source_i = gen() % length;
        while (used_indices.count(source_i) > 0) {
            source_i = gen() % length;
        }
        used_indices.insert(source_i);
        to[i] = from[source_i];
    }
}


struct GCV_Result
{
	char* error = nullptr;
	int n_objects = 0;
	int n_contrast_vars = 0;
	double* contrast_data = nullptr;
	int* indices = nullptr;
};


LIBRARY_API
GCV_Result gen_contrast_variables(
    int obj_count,
    int var_count,
    double* data,
    int n_contrast,
    int seed
)
{
    GCV_Result result;

    if(n_contrast == -1)
    {
        n_contrast = std::max(var_count, 30);
    }
    else if(n_contrast < 1)
    {
        result.error = (char*) "n_contrast has to be a positive integer";
        return result;
    }

    result.indices = new int[n_contrast];

    if(seed == -1)
    {
        std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());
        seed = generator() / 2; // halving the result to keep it (exactly) in desired range
    } else if(seed < 0 || seed > 2147483647)
    {
        result.error = (char*) "Only integer seeds from 0 to 2^31-1 are portable. Using non-portable seed may make result harder to reproduce.";
        return result;
    }

    std::mt19937 generator(seed);

    if(n_contrast > var_count) // pick random indices in range [0, var_count)
    {
        std::uniform_int_distribution<> distribution(0, var_count - 1);

        for(int i = 0; i < n_contrast; i++)
        {
            result.indices[i] = distribution(generator);
        }
    }
    else // sample indices from a vector with sequence of integers in range [0, var_count)
    {
        mdfs_sample_to(result.indices, var_count, n_contrast, generator);
    }

    const int contrast_data_count = n_contrast * obj_count;

    result.contrast_data = new double[contrast_data_count];

    for(int i = 0; i < n_contrast; i++)
    {
        mdfs_shuffle_to(
            result.contrast_data + i * obj_count,
            data + result.indices[i] * obj_count,
            obj_count, generator);
    }

    result.n_objects = obj_count;
    result.n_contrast_vars = n_contrast;

    return result;
}

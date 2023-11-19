#include "common.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>

struct PVFit_Result
{
    char* error = nullptr;
    int n_vars = 0;
    double* p_value = nullptr;
    double sq_dev = 0.0;
    double dist_param = 0.0;
    double err_param = 0.0;
};

struct calcS_params
{
    int data_count;
    double* S = nullptr;
    double* alpha = nullptr;
    double* d_alpha = nullptr;

    ~calcS_params()
    {
        delete[] this->S;
        delete[] this->alpha;
        delete[] this->d_alpha;
    }
};

calcS_params calc_SSSS(int n0i, int n_var, const std::vector<int>& K, const std::vector<double>& L,
    const std::vector<double>& Sw, const std::vector<double>& Sv,
    const std::vector<double>& Swv, const std::vector<double>& Swv2,
    const std::vector<double>& Swl, const std::vector<double>& Swl2,
    const std::vector<double>& Swlv, const std::vector<double>& Srt,
    const bool one_dim_mode) 
{

    int npt = n_var - n0i ;
    // Create sub-vectors 'k' and 'l' by taking slices of 'K' and 'L'
    std::vector<int> k(K.begin(), K.begin() + npt);
    std::vector<double> l(L.begin(), L.begin() + npt);

    // Create sub-vectors 'sw', 'sv', 'swv', 'swv2', 'swl', 'swl2', 'swlv', and 'srt'
    std::vector<double> sw(Sw.begin() + n0i, Sw.end());
    std::vector<double> sv(Sv.begin() + n0i, Sv.end());
    std::vector<double> swv(Swv.begin() + n0i, Swv.end());
    std::vector<double> swv2(Swv2.begin() + n0i, Swv2.end());
    std::vector<double> swl(Swl.begin() + n0i, Swl.end());
    std::vector<double> swl2(Swl2.begin() + n0i, Swl2.end());
    std::vector<double> swlv(Swlv.begin() + n0i, Swlv.end());
    std::vector<double> srt(Srt.begin() + n0i, Srt.end());

    for(int i = 0; i < npt; i++)
    {
        sw[i]   = sw[i]-Sw[n0i-1];
        sv[i]   = sv[i]-Sv[n0i-1];
        swv[i]  = swv[i]-Swv[n0i-1];
        swv2[i] = swv2[i]-Swv2[n0i-1];
        swl[i]  = swl[i]-Swl[n0i-1];
        swl2[i] = swl2[i]-Swl2[n0i-1];
        swlv[i] = swlv[i]-Swlv[n0i-1];
    }

    // Initialize variables for S, alpha, d_alpha as double* pointers
    double* S = new double[npt];
    double* alpha = new double[npt];
    double* d_alpha = new double[npt];

    if (one_dim_mode) {
        for (int i = 0; i < npt; ++i) {
            alpha[i] = sv[i] / k[i];
            S[i] = (
                swv2[i]
                + 2 * alpha[i] * swlv[i]
                - 2 * alpha[i] * l[i] * swv[i]
                + alpha[i] * alpha[i] * swl2[i]
                - 2 * alpha[i] * alpha[i] * l[i] * swl[i]
                + ((alpha[i] * l[i]) * (alpha[i] * l[i])) * sw[i]
                );
            d_alpha[i] = S[i] / (
                swl2[i]
                - 2 * l[i] * swl[i]
                + l[i] * l[i] * sw[i]
                );
        }
    } else {
        for (int i = 0; i < npt; ++i) {
            if (!one_dim_mode) {
                alpha[i] = 1.0;
            } else {
                alpha[i] = (swv[i] - swlv[i] / l[i]) / (sw[i] - 2 * swl[i] / l[i] + swl2[i] / (l[i] * l[i]));
            }
            S[i] = (swv2[i] - 2 * alpha[i] * swv[i] + 2 * alpha[i] / l[i] * swlv[i] + alpha[i] * alpha[i] * sw[i] - 2 * alpha[i] * alpha[i] / l[i] * swl[i] + alpha[i] * alpha[i] / (l[i] * l[i]) * swl2[i]);
            d_alpha[i] = S[i] / (sw[i] - 2 * swl[i] / l[i] + swl2[i] / (l[i] * l[i]));
        }
    }

    double* padded_S = new double[npt + 1];
    padded_S[0] = 0;
    for (int i = 0; i < npt; ++i) {
        padded_S[i+1] = (S[i]);
    }

    double* padded_srt = new double[npt + 1];
    for (int i = 0; i < npt; ++i) {
        padded_srt[i] = (srt[i]);
    }
    padded_srt[npt+1] = 0;

    for (int i = 0; i < npt; ++i) {
        S[i] = (padded_S[i] + padded_srt[i])/sw[npt-1];
    }

    calcS_params result;
    result.S = S;
    result.alpha = alpha;
    result.d_alpha = d_alpha;
    result.data_count = npt;

    delete[] padded_S;
    return result;
}


int calculate_nvi(int irr_vars_num, int min_irr_vars_num, const double* S, int s_size) 
{
    int nvi;
    
    if (irr_vars_num > 0) {
        nvi = irr_vars_num;
    } else {
        auto min_it = std::min_element(S + min_irr_vars_num - 1, S + s_size);
        if (min_it != S + s_size) {
            nvi = std::distance(S, min_it) + 1;
        } else {
            nvi = -1;
        }
    }
    
    return nvi;
}

LIBRARY_API
PVFit_Result fit_p_value_(
    int var_count,
    double* chisq,
    int contrast_count,
    double* chisq_contrast,
    bool one_dim_mode,
    int irr_vars_num,
    int ign_low_ig_vars_num,
    int min_irr_vars_num,
    int max_ign_low_ig_vars_num,
    int search_points)
{

    PVFit_Result result;
    std::vector<double> IGc;

    if(var_count < 4)
    {
        result.error = (char*) "IG needs at least 4 values";
        return result;
    }

    if(contrast_count < 4){
        result.error = (char*) "Contrast data needs at least 4 values";
        return result;
    }

    if(min_irr_vars_num == -1)
    {
        min_irr_vars_num = std::min(contrast_count / 3, 30);
    }
    else if(min_irr_vars_num < 2)
    {
        result.error = (char*) "min_irr_vars_num has to be an integer bigger than 2";
        return result;
    }

    if(max_ign_low_ig_vars_num == -1)
    {
        max_ign_low_ig_vars_num = std::min(contrast_count - min_irr_vars_num, contrast_count / 3);
    }
    else if(max_ign_low_ig_vars_num < 0)
    {
        result.error = (char*) "max_ign_low_ig_vars_num has to be an non-negative integer";
        return result;
    }

    if(min_irr_vars_num + max_ign_low_ig_vars_num > contrast_count)
    {
        result.error = (char*) "Sum of min_irr_low_ig_vars_num and max_ign_vars_num has to be lower than the IG vector length";
        return result;
    }

    if(irr_vars_num != -1 && irr_vars_num < 3)
    {
        result.error = (char*) "irr_vars_num has to be an integer bigger than 2";
        return result;
    }

    if(irr_vars_num != -1)
    {
        if(irr_vars_num < 3)
        {
            result.error = (char*) "irr_vars_num has to be an integer bigger than 2";
            return result;
        }
        if(irr_vars_num < min_irr_vars_num)
        {
            result.error = (char*) "irr_vars_num cannot be smaller than min_irr_vars_num";
            return result;
        }
    }

    if(ign_low_ig_vars_num != -1)
    {
        if(ign_low_ig_vars_num < 0)
        {
            result.error = (char*) "ign_low_ig_vars_num has to be a non-negative integer";
            return result;
        }
        if(ign_low_ig_vars_num > max_ign_low_ig_vars_num)
        {
            result.error = (char*) "ign_low_ig_vars_num cannot be bigger than max_ign_low_ig_vars_num";
            return result;
        }
        if(search_points < 2)
        {
            result.error = (char*) "search_points has to be an integer bigger than 1";
            return result;
        }
    }

    if((ign_low_ig_vars_num == -1 ? max_ign_low_ig_vars_num : ign_low_ig_vars_num) + (irr_vars_num == -1 ? min_irr_vars_num : irr_vars_num) > contrast_count)
    {
        result.error = (char*) "Sum of ign_low_ig_vars_num and irr_vars_num cannot be bigger than the IG vector length";
        return result;
    }

    std::vector<int> K(contrast_count);
    std::vector<double> V(contrast_count);
    std::vector<double> L(contrast_count);
    std::vector<double> W(contrast_count);

    std::sort(chisq_contrast, chisq_contrast + contrast_count);
    std::reverse(chisq_contrast, chisq_contrast + contrast_count);

    std::vector<double> chisq_log(var_count);
    for(int i = 0; i < var_count; i++)
    {
        chisq_log[i] = log(1.0 - chisq[i]);
    }

    for (int i = 0; i < contrast_count; ++i) {
        K[i] = i + 1;
    }

    for (int i = 0; i < contrast_count; ++i) {
        V[i] = chisq_contrast[K[i] - 1];
    }

    // Check conditions and calculate L and W accordingly
    if (one_dim_mode) {
        for (int i = 0; i < contrast_count; ++i) {
            L[i] = std::log(K[i]);
        }
        for (int i = 0; i < contrast_count; ++i) {
            W[i] = static_cast<double>(K[i]) / (contrast_count - K[i] + 1);
        }
    } else {
        std::copy(K.begin(), K.end(), L.begin());
        for (int i = 0; i < contrast_count; ++i) {
            W[i] = 1.0 / (K[i] * (contrast_count - K[i] + 1));
        }
    }

    std::vector<double> Sw(contrast_count);
    std::vector<double> Sv(contrast_count);
    std::vector<double> Swv(contrast_count);
    std::vector<double> Swv2(contrast_count);
    std::vector<double> Swl(contrast_count);
    std::vector<double> Swl2(contrast_count);
    std::vector<double> Swlv(contrast_count);
    std::vector<double> Srt(contrast_count);

    double cum_sum_W = 0.0;
    double cum_sum_V = 0.0;
    double cum_sum_WV = 0.0;
    double cum_sum_WV2 = 0.0;
    double cum_sum_WL = 0.0;
    double cum_sum_WL2 = 0.0;
    double cum_sum_WLV = 0.0;
    double cum_sum_W_chisq = 0.0;

    for (int i = 0; i < contrast_count; ++i) {
        cum_sum_W += W[i];
        cum_sum_V += V[i];
        cum_sum_WV += W[i] * V[i];
        cum_sum_WV2 += W[i] * V[i] * V[i];
        cum_sum_WL += W[i] * L[i];
        cum_sum_WL2 += W[i] * L[i] * L[i];
        cum_sum_WLV += W[i] * L[i] * V[i];

        Sw[i] = cum_sum_W;
        Sv[i] = cum_sum_V;
        Swv[i] = cum_sum_WV;
        Swv2[i] = cum_sum_WV2;
        Swl[i] = cum_sum_WL;
        Swl2[i] = cum_sum_WL2;
        Swlv[i] = cum_sum_WLV;
    }

    for (int i = contrast_count - 1; i >= 0; --i) {
        double cumulative_sum = 0.0;
        for (int j = contrast_count - 1; j >= contrast_count - i; --j) {
            cumulative_sum += W[j] * chisq_contrast[j] * chisq_contrast[j];
        }
        Srt[i] = cumulative_sum;
    }
    std::reverse(Srt.begin(), Srt.end());
    
    int n0i = 0;
    int n0;
    int nv;
    double alpha = 0.0;

    if(ign_low_ig_vars_num == -1)
    {
        int n0_min = 0;
        int n0_max = max_ign_low_ig_vars_num - 1;

        double Smin = std::numeric_limits<double>::infinity();
        
        n0 = -1;
        nv = -1;
        int step;

        do{
            step = std::max(1, (int)round((n0_max - n0_min) / (double)search_points));
            int n0_i = n0_min;
            while (n0_i<n0_max) {
                calcS_params params = calc_SSSS(n0_i, contrast_count, K, L, Sw, Sv, Swv, Swv2, Swl, Swl2, Swlv, Srt, one_dim_mode);
                int nv_i = calculate_nvi(irr_vars_num,min_irr_vars_num,params.S,params.data_count);
                nv_i--; // bringing index base to 0
                double Si = params.S[nv_i];
                if(Si < Smin)
                {
                    n0 = n0_i;
                    nv = nv_i;
                    Smin = Si;

                    alpha = params.alpha[nv];
                }
                n0_i += step;
            }
            if(n0 > -1)
            {
                n0_min = std::max(1, n0 - step);
                n0_max = std::min(max_ign_low_ig_vars_num, n0 + step);
            }
        }while(step>1);
    }
    else
    {
        
    }

    // print("final alpha:",alpha);

    ign_low_ig_vars_num = std::min(n0 - 1, max_ign_low_ig_vars_num);
    irr_vars_num = std::max(min_irr_vars_num, nv);

    if(ign_low_ig_vars_num >= max_ign_low_ig_vars_num)
    {
        result.error = (char*) "Border value reached for ignored variable number";
        return result;
    }

    if(irr_vars_num <= min_irr_vars_num)
    {
        result.error = (char*) "Border value reached for irrelevant variable number";
        return result;
    }

    result.p_value = new double[var_count];

    if(one_dim_mode)
    {
        for(int i = 0; i < var_count; i++)
        {
            result.p_value[i] = -expm1(chisq_log[i]/alpha);
        }
    }
    else
    {
        for(int i = 0; i < var_count; i++)
        {
            result.p_value[i] = chisq[i] / alpha;
        }
    }

    result.n_vars = var_count;

    return result;
}

LIBRARY_API
std::vector<int> adjust(double* p, int n, double* adjusted_p, std::string& p_method,
    double threshold) 
{
    std::vector<size_t> indices(n);
    std::vector<int>relevant_variables;
    for (size_t i = 0; i < n; ++i) {
        indices[i] = i;
    }

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return p[a] < p[b];
    });

    for (size_t i = 0; i < n; ++i) {
        double adjusted_value = 0;
        int index = 0;
        if (strcasecmp(p_method.c_str(), "holm") == 0) {
            index = indices[i];
            adjusted_value = std::min(1.0, (n - i) * p[index]);
            adjusted_p[index] = adjusted_value;
        } else if (strcasecmp(p_method.c_str(), "hochberg") == 0) { //not working
            int j = n - i - 1;
            index = indices[j];
            adjusted_value = std::min(p[index] * (j + 1), 1.0);
            adjusted_p[index] = adjusted_value;            
        } else if (strcasecmp(p_method.c_str(), "bh") == 0) { //accuracy is not perfect
            int j = n - i - 1;
            index = indices[j];
            adjusted_value = std::min(p[index] * (n / (j + 1.0)), 1.0);
            adjusted_p[index] = adjusted_value;
        } else if (strcasecmp(p_method.c_str(), "by") == 0) {//accuracy is not perfect
            int j = n - i - 1;
            index = indices[j];
            adjusted_value = std::min(p[index] * (n / (j + 1.0)) / (1.0 / (1.0 + (1.0 / (j + 1.0)))), 1.0);
            adjusted_p[index] = adjusted_value;
        }
        if(adjusted_value < threshold){
            relevant_variables.push_back(index+1);
        }
    }
    return relevant_variables;
}

#include "common.h"

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>

using std::vector;


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
	    delete [] this->S;
	    delete [] this->alpha;
	    delete [] this->d_alpha;
	}
};

calcS_params calcS(
    int ix,
    int n,
	bool exponential_fit,
    double* K,
    double* L,
    double* Sw,
    double* Sv,
    double* Swv,
    double* Swv2,
    double* Swl,
    double* Swl2,
    double* Swlv,
    double* Srt
)
{
	calcS_params result;

    int npt = n - ix;

    result.data_count = npt;

	vector<double> k(npt);
	vector<double> l(npt);
	vector<double> sw(npt);
	vector<double> sv(npt);
	vector<double> swv(npt);
    vector<double> swv2(npt);
    vector<double> swl(npt);
    vector<double> swl2(npt);
    vector<double> swlv(npt);
    vector<double> srt(npt);

    std::copy(K, K + npt, k.begin());
    std::copy(L, L + npt, l.begin());
	std::copy(Srt + ix, Srt + n, srt.begin());

	for(int i = 0; i < npt; i++)
	{
		sw[i]   = Sw[ix + i]   - (ix ?   Sw[ix-1] : 0);
		sv[i]   = Sv[ix + i]   - (ix ?   Sv[ix-1] : 0);
		swv[i]  = Swv[ix + i]  - (ix ?  Swv[ix-1] : 0);
		swv2[i] = Swv2[ix + i] - (ix ? Swv2[ix-1] : 0);
		swl[i]  = Swl[ix + i]  - (ix ?  Swl[ix-1] : 0);
		swl2[i] = Swl2[ix + i] - (ix ? Swl2[ix-1] : 0);
		swlv[i] = Swlv[ix + i] - (ix ? Swlv[ix-1] : 0);
	}

	result.alpha = new double[npt];
	result.S = new double[npt];
	result.d_alpha = new double[npt];

    if(exponential_fit)
    {
		for(int i = 0; i < npt; i++)
		{
			result.alpha[i] = sv[i] / k[i];

			result.S[i] = swv2[i]
				+ 2.0 * result.alpha[i] * swlv[i]
				- 2.0 * result.alpha[i] * l[i] * swv[i]
				+ result.alpha[i] * result.alpha[i] * swl2[i]
				- 2.0 * result.alpha[i] * result.alpha[i] * l[i] * swl[i]
				+ (result.alpha[i] * l[i]) * (result.alpha[i] * l[i]) * sw[i];

			result.d_alpha[i] = result.S[i] / (swl2[i] - 2.0 * l[i] * swl[i] + l[i] * l[i] * sw[i]);
		}
    }
    else
    {
		for(int i = 0; i < npt; i++)
		{
			result.alpha[i] = (swv[i] - swlv[i]/l[i]) / (sw[i] - 2.0*swl[i]/l[i] + swl2[i]/(l[i]*l[i]));

			result.S[i] = swv2[i]
				- 2.0 * result.alpha[i] * swv[i]
				+ 2.0 * result.alpha[i] / l[i] * swlv[i]
				+ result.alpha[i] * result.alpha[i] * sw[i]
				- 2.0 * result.alpha[i] * result.alpha[i] / l[i] * swl[i]
				+ result.alpha[i] * result.alpha[i] / (l[i] * l[i]) * swl2[i];

			result.d_alpha[i] = result.S[i] / (sw[i] - 2.0*swl[i]/l[i] + swl2[i]/(l[i]*l[i]));
		}
    }

    for(int i = npt-1; i > 0; i--)
    {
		result.S[i] = (result.S[i-1] + srt[i]) / sw[npt-1];
    }
    result.S[0] = srt[0] / sw[npt-1];

    return result;
}

LIBRARY_API
PVFit_Result fit_p_value(
    int var_count,
    double* chisq,
    int contrast_count,
    double* chisq_contrast,
    bool exponential_fit,
    int irr_vars_num,
    int ign_low_ig_vars_num,
    int min_irr_vars_num,
    int max_ign_low_ig_vars_num,
    int search_points
)
{
    PVFit_Result result;

    vector<double> IGc;

    if(var_count < 4)
    {
        result.error = (char*) "IG needs at least 4 values";
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

    vector<double> chisq_log(var_count);

    for(int i = 0; i < var_count; i++)
    {
        chisq_log[i] = log(1.0 - chisq[i]);
    }

    // | Weights and index
    vector<double> K(contrast_count);
    vector<double> V(contrast_count);

    std::iota(K.begin(), K.end(), 1.0);
    std::copy(chisq_contrast, chisq_contrast + contrast_count, V.begin());
    std::sort(V.begin(), V.end());
    std::reverse(V.begin(), V.end());

    vector<double> L(contrast_count);
    vector<double> W(contrast_count);
    vector<double> wv(contrast_count);
    vector<double> wv2(contrast_count);
    vector<double> wl(contrast_count);
    vector<double> wl2(contrast_count);
    vector<double> wlv(contrast_count);
    vector<double> rt(contrast_count);

    vector<double> sw(contrast_count);
    vector<double> sv(contrast_count);
    vector<double> swv(contrast_count);
    vector<double> swv2(contrast_count);
    vector<double> swl(contrast_count);
    vector<double> swl2(contrast_count);
    vector<double> swlv(contrast_count);
    vector<double> srt(contrast_count);

    for(int i = 0; i < contrast_count; i++)
    {
        L[i] = (exponential_fit) ? log(K[i]) : K[i];
        W[i] = (exponential_fit)
            ? K[i] / (contrast_count - K[i]+1)
            : 1.0 / K[i] / (contrast_count - K[i]+1);

        wv[i] = W[i] * V[i];
        wv2[i] = W[i] * V[i] * V[i];
        wl[i] = W[i] * L[i];
        wl2[i] = W[i] * L[i] * L[i];
        wlv[i] = W[i] * L[i] * V[i];
        rt[i] = W[i] * V[i] * V[i];
    }

    std::reverse(rt.begin(), rt.end());

    std::partial_sum(W.begin(), W.end(), sw.begin());
    std::partial_sum(V.begin(), V.end(), sv.begin());
    std::partial_sum(wv.begin(), wv.end(), swv.begin());
    std::partial_sum(wv2.begin(), wv2.end(), swv2.begin());
    std::partial_sum(wl.begin(), wl.end(), swl.begin());
    std::partial_sum(wl2.begin(), wl2.end(), swl2.begin());
    std::partial_sum(wlv.begin(), wlv.end(), swlv.begin());
    std::partial_sum(rt.begin(), rt.end(), srt.begin());
    std::reverse(srt.begin(), srt.end());

    // | search for the optimal number of ignored variables
    int n0;
    int nv;
    // Setting alpha upfront to silence the compiler's warnings.
    // It is always set properly because of Smin being infinity on start.
    double alpha = 0.0;

    if(ign_low_ig_vars_num == -1)
    {
        int n0_min = 0;
        int n0_max = max_ign_low_ig_vars_num - 1;

        double Smin = std::numeric_limits<double>::infinity();

        n0 = -1;
        nv = -1;
        int step;

        do
        {
            step = std::max(1, (int)round((n0_max - n0_min) / (double)search_points));
            int n0_i = n0_min;

            while(n0_i < n0_max)
            {
                calcS_params params = calcS(
                            n0_i, contrast_count, exponential_fit, &K[0], &L[0],
                            &sw[0], &sv[0], &swv[0], &swv2[0], &swl[0], &swl2[0], &swlv[0], &srt[0] );

                int nv_i = (irr_vars_num == -1)
                    ? params.data_count
                    : irr_vars_num;

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
        }
        while(step > 1);
    }
    else
    {
        n0 = ign_low_ig_vars_num + 1;

        calcS_params params = calcS(
                    n0, contrast_count, exponential_fit, &K[0], &L[0],
                    &sw[0], &sv[0], &swv[0], &swv2[0], &swl[0], &swl2[0], &swlv[0], &srt[0] );

        nv = (irr_vars_num == -1)
            ? params.data_count
            : irr_vars_num;

        nv--; // bringing index base to 0

        alpha = params.alpha[nv];
    }

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

    if(exponential_fit)
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

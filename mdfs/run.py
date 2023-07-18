import math
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

from . import gen_contrast_variables, compute_max_ig, fit_p_value


def run(data, decision, *, n_contrast=None, dimensions=1, divisions=1, discretizations=1,
        seed=None, range_=None, pc_xi=0.25, p_adjust_method='holm', level=0.05, use_CUDA=False):

    # n_objs not used directly
    _, n_vars = data.shape

    if dimensions == 1 and discretizations == 1:
        fit_mode = "raw"
    elif dimensions == 1 and discretizations * divisions < 12:
        fit_mode = "lin"
    else:
        fit_mode = "exp"

    if n_contrast is None:
        if fit_mode != "raw":
            n_contrast = max(30, n_vars)
        else:
            n_contrast = 0

    if n_contrast > 0:
        contrast_variables = gen_contrast_variables(data, n_contrast=n_contrast, seed=seed)
        contrast_data = contrast_variables.data
        contrast_indices = contrast_variables.indices
    else:
        contrast_data = None
        contrast_indices = None

    ig_result = compute_max_ig(
        data,
        decision,
        contrast_data=contrast_data,
        dimensions=dimensions,
        divisions=divisions,
        discretizations=discretizations,
        seed=seed,
        range_=range_,
        pc_xi=pc_xi,
        use_CUDA=use_CUDA)

    common_df = divisions * math.pow(divisions+1, dimensions-1)

    mdfs_result = {
        "chi_squared": chi2.sf(ig_result.max_igs * math.log(2) * 2, common_df),
    }

    if fit_mode == "raw":
        mdfs_result["p_value"] = mdfs_result["chi_squared"]
    else:  # lin or exp
        contrast_chi_squared = chi2.sf(ig_result.max_igs_contrast * math.log(2) * 2, common_df)
        fpv_result = fit_p_value(mdfs_result["chi_squared"], contrast_chi_squared,
                                 exponential_fit=(fit_mode=="exp"))
        mdfs_result["p_value"] = fpv_result.p_values.copy()

    (rejections, adjusted_p_value, _, _) = multipletests(mdfs_result["p_value"], alpha=level, method=p_adjust_method)

    result = {
        "contrast_indices": contrast_indices.copy(),
        "statistic": ig_result.max_igs.copy(),
        "statistic_contrast": ig_result.max_igs_contrast.copy(),
        "chi_squared": mdfs_result["chi_squared"],
        "p_value": mdfs_result["p_value"],
        "adjusted_p_value": adjusted_p_value,
        "relevant_variables": rejections.nonzero()[0],
    }

    return result

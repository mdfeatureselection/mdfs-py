import mdfs
import numpy as np
import pandas as pd
import json
import math
from scipy.stats import chi2

#| Reading data
madelon = pd.read_csv("../data/all_madelon_dat_py.csv", dtype=float)
dim = madelon.shape
data = madelon[[col for col in madelon.columns[:dim[1]-1]]][:dim[0]].to_numpy()
decision = madelon['Class'][:dim[0]].astype(np.intc).to_numpy()
contrast_ = pd.read_csv("../data/contrast_dat.csv", dtype=float)
contrast_data = contrast_.iloc[:dim[0], :dim[1]].to_numpy()

dimensions = 2
discretizations = 1
divisions=1
seed = 0
range_ = 0
pc_xi = 0.25
use_CUDA = False

if dimensions == 1 and discretizations == 1:
    fit_mode = "raw"
elif dimensions == 1 and discretizations * divisions < 12:
    fit_mode = "lin"
else:
    fit_mode = "exp"

ig_result = mdfs.compute_max_ig(
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
    "chi_squared": chi2.sf(ig_result.max_igs * math.log(2) * 2, common_df)
}

result_ = {
        "max_ig": ig_result.max_igs.tolist(),
        "max_ig_contrast": ig_result.max_igs_contrast.tolist(),
        "chi_squared_max_ig":mdfs_result["chi_squared"].tolist()
}

with open("results/example_mdfs_compute_max_ig_results.json", "w") as json_file:
    json.dump(result_, json_file)
import mdfs
import numpy as np
import pandas as pd
import json

#| Reading data
madelon = pd.read_csv("../data/all_madelon_dat_py.csv", dtype=float)
dim = madelon.shape
data = madelon[[col for col in madelon.columns[:dim[1]-1]]][:dim[0]].to_numpy()
decision = madelon['Class'][:dim[0]].astype(np.intc).to_numpy()

dimensions = 1

result = mdfs.run(data, decision, dimensions = 2, divisions = 1,
     seed = 0, range_ = 0)

result_ = {
        "contrast_indices": result["contrast_indices"].tolist(),
        "statistic": result["statistic"].tolist(),
        "statistic_contrast": result["statistic_contrast"].tolist(),
        "chi_squared": result["chi_squared"].tolist(),
        "p_value": result["p_value"].tolist(),
        "adjusted_p_value": result["adjusted_p_value"].tolist(),
        "relevant_variables_indices": result["relevant_variables"].tolist(),
}

with open("results/example_mdfs_run_results.json", "w") as json_file:
    json.dump(result_, json_file)
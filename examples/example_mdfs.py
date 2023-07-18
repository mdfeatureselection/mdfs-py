import mdfs
import numpy as np
import pandas as pd
import json

#| Reading data
madelon = pd.read_csv("data/madelon.50.csv", dtype=float)
dim = madelon.shape
data = madelon[[col for col in madelon.columns[:dim[1]-1]]][:dim[0]].to_numpy()
decision = madelon['Class'][:dim[0]].astype(np.intc).to_numpy() - 1

dimensions = 1

result = mdfs.run(data, decision, dimensions=dimensions, seed=0, discretizations=30, n_contrast=30)

contrast_variables = mdfs.gen_contrast_variables(data, n_contrast=30, seed=0)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


with open("data/MDFS_Result_python_" + str(dimensions) + "D.json", "w") as f:
    f.write(json.dumps(result, indent=4, cls=NumpyEncoder))

# for contrast variable debugging:
# np.savetxt("data/test_contrast_variables.csv", contrast_variables.data, delimiter="\t")

pairs = mdfs.compute_tuples(data=data, decision=decision, seed=0, discretizations=30, return_matrix=True)

np.savetxt("data/pairs.csv", pairs.stats, delimiter="\t")

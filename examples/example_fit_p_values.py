import mdfs
import numpy as np
import pandas as pd
import json


chisq = pd.read_csv("../data/chisq_log.txt", header=None, dtype=float)  # No header specified
chisq_dat = chisq.iloc[:,0].to_numpy()

chisq_contrast = pd.read_csv("../data/chisq_contrast.txt", header=None, dtype=float)  # No header specified
chisq_contrast_Dat = chisq_contrast.iloc[:,0].to_numpy()

dimensions = 2
exponential_fit="exp"

ddd = (mdfs.fit_p_value(chisq=chisq_dat,chisq_contrast=chisq_contrast_Dat,exponential_fit= (exponential_fit == "exp")))
print(ddd.p_values)
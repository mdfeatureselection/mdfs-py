from ctypes import CDLL, POINTER, c_bool, c_double, c_int
import numpy as np
from pathlib import Path


funs = CDLL(str(Path(__file__).parent / "libmdfs.so"))


funs.freeIntArray.argtypes = [POINTER(c_int)]
funs.freeIntArray.restype = None

funs.freeDoubleArray.argtypes = [POINTER(c_double)]
funs.freeDoubleArray.restype = None

funs.freeBoolArray.argtypes = [POINTER(c_bool)]
funs.freeBoolArray.restype = None


data_type = np.ctypeslib.ndpointer(float, ndim=2, flags='ALIGNED, F_CONTIGUOUS')
decision_type = np.ctypeslib.ndpointer(np.intc, ndim=1, flags='ALIGNED, F_CONTIGUOUS')


def _from_param_data(cls, obj):
    if obj is None:
        return obj
    return data_type.from_param(obj)


def _from_param_decision(cls, obj):
    if obj is None:
        return obj
    return decision_type.from_param(obj)


optional_data_type = type(
    'optional_data_type',
    (data_type,),
    {'from_param': classmethod(_from_param_data)}
)

optional_decision_type = type(
    'optional_decision_type',
    (decision_type,),
    {'from_param': classmethod(_from_param_decision)}
)

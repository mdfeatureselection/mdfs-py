from ctypes import c_int, c_double, c_char_p, POINTER, Structure

import numpy as np

from .c import funs, data_type
from .internal_common import handle_error


class DoubleResult(Structure):
    _fields_ = [('_error', c_char_p),
                ('_value', c_double)]


funs.get_suggested_range.argtypes = [
    c_int,  # n
    c_int,  # dimensions
    c_int,  # divisions
    c_int,  # k
]
funs.get_suggested_range.restype = DoubleResult


def get_suggested_range(n, dimensions, divisions, k = 3):
    result = funs.get_suggested_range(c_int(n), c_int(dimensions), c_int(divisions), c_int(k))

    return handle_error(result)._value


column_type = np.ctypeslib.ndpointer(float, ndim=1, flags='ALIGNED, F_CONTIGUOUS')


class DiscretizeResult(Structure):
    _fields_ = [('_error', c_char_p),
                ('_n_objs', c_int),
                ('_discretized_var', POINTER(c_int))]

    @property
    def discretized_var(self):
        if self._discretized_var:
            return np.ctypeslib.as_array(self._discretized_var, (self._n_objs,))

    def __del__(self):
        funs.freeIntArray(self.discretized_var)


funs.discretize.argtypes = [c_int, column_type, c_int, c_int, c_int, c_int, c_double]
funs.discretize.restype = DiscretizeResult


def discretize(data, *, variable_idx, divisions, discretization_nr, seed, range_):

    variable = data[:, variable_idx]
    obj_count = len(variable)

    result = funs.discretize(c_int(obj_count), variable, c_int(variable_idx), c_int(divisions),
                                c_int(discretization_nr), c_int(seed), c_double(range_))

    return handle_error(result)


class GenContrastVariablesResult(Structure):
    _fields_ = [('_error', c_char_p),
                ('_n_objects', c_int),
                ('_n_contrast_vars', c_int),
                ('_data', POINTER(c_double)),
                ('_indices', POINTER(c_int))]

    @property
    def data(self):
        if self._data:
            # NOTE: transpose to give the expected view on the data (F_CONTIGUOUS)
            return np.ctypeslib.as_array(self._data, shape=(self._n_contrast_vars, self._n_objects)).transpose()

    @property
    def indices(self):
        if self._indices:
            return np.ctypeslib.as_array(self._indices, (self._n_contrast_vars,))

    def __del__(self):
        funs.freeDoubleArray(self._data)
        funs.freeIntArray(self._indices)


funs.gen_contrast_variables.argtypes = [c_int, c_int, data_type, c_int, c_int]
funs.gen_contrast_variables.restype = GenContrastVariablesResult


def gen_contrast_variables(data, n_contrast, seed=None):

    obj_count, var_count = data.shape
    seed = -1 if seed is None else seed
    result = funs.gen_contrast_variables(obj_count, var_count, data, n_contrast, seed)
    return handle_error(result)

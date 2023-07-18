from ctypes import c_int, c_double, c_bool, c_char_p, POINTER, Structure
import numpy as np

from .c import funs, data_type, decision_type, optional_data_type
from .internal_common import handle_error


class ComputeMaxIGResult(Structure):
    _fields_ = [('_error', c_char_p),
                ('_n_dimensions', c_int),
                ('_n_max_igs', c_int),
                ('_max_igs', POINTER(c_double)),
                ('_n_max_igs_contrast', c_int),
                ('_max_igs_contrast', POINTER(c_double)),
                ('_tuples', POINTER(c_int)),
                ('_dids', POINTER(c_int))]

    @property
    def max_igs(self):
        if self._max_igs:
            return np.ctypeslib.as_array(self._max_igs, (self._n_max_igs,))

    @property
    def max_igs_contrast(self):
        if self._max_igs_contrast:
            return np.ctypeslib.as_array(self._max_igs_contrast, (self._n_max_igs_contrast,))

    @property
    def tuples(self):
        if self._tuples:
            # NOTE: this is C_CONTIGUOUS; R has to fix it for itself by transposing
            return np.ctypeslib.as_array(self._tuples, (self._n_max_igs, self._n_dimensions))

    @property
    def discretization_nrs(self):
        if self._dids:
            return np.ctypeslib.as_array(self._dids, (self._n_max_igs,))

    def __del__(self):
        funs.freeDoubleArray(self._max_igs)
        funs.freeDoubleArray(self._max_igs_contrast)
        funs.freeIntArray(self._tuples)
        funs.freeIntArray(self._dids)


funs.compute_max_ig.argtypes = [
    c_int,  # obj_count
    c_int,  # var_count
    data_type,  # data
    c_int,  # n_contrast_variables
    optional_data_type,  # contrast_data
    decision_type,  # decision
    c_int,  # dimensions
    c_int,  # divisions
    c_int,  # discretizations
    c_int,  # seed
    c_double,  # range
    c_double,  # pseudocount
    c_bool,  # return_tuples
    c_int,  # int_vars_count
    POINTER(c_int),  # interesting_vars
    c_bool,  # require_all_vars
    c_bool,  # use_CUDA
]
funs.compute_max_ig.restype = ComputeMaxIGResult

funs.compute_max_ig_discrete.argtypes = [
    c_int,  # obj_count
    c_int,  # var_count
    data_type,  # data
    c_int,  # n_contrast_variables
    optional_data_type,  # contrast_data
    decision_type,  # decision
    c_int,  # dimensions
    c_int,  # divisions
    c_double,  # pseudocount
    c_bool,  # return_tuples
    c_int,  # int_vars_count
    POINTER(c_int),  # interesting_vars
    c_bool,  # require_all_vars
]
funs.compute_max_ig_discrete.restype = ComputeMaxIGResult


def compute_max_ig(data, decision, contrast_data=None, dimensions=1, divisions=1, discretizations=1,
                   seed=None, range_=None, pc_xi=0.25, return_tuples=False,
                   interesting_vars=None, require_all_vars=False, use_CUDA=False):

    obj_count, var_count = data.shape

    if len(decision) != obj_count:
        raise Exception("Length of decision is not equal to the number of rows in data.")

    seed = -1 if seed is None else seed
    range_ = -1 if range_ is None else range_

    iv_count = 0 if interesting_vars is None else len(interesting_vars)
    in_interesting = (c_int * iv_count)(*interesting_vars) if interesting_vars else (c_int * iv_count)()

    if contrast_data is not None:
        if contrast_data.shape[0] != data.shape[0]:
            raise Exception("contrast_data must have the same object count as data")
        n_contrast_variables = contrast_data.shape[1]
    else:
        n_contrast_variables = 0

    result = funs.compute_max_ig(obj_count, var_count, data, n_contrast_variables, contrast_data, decision, c_int(dimensions),
                                    c_int(divisions), c_int(discretizations), c_int(seed),
                                    c_double(range_), c_double(pc_xi), c_bool(return_tuples), iv_count,
                                    in_interesting, c_bool(require_all_vars), c_bool(use_CUDA))

    return handle_error(result)


def compute_max_ig_discrete(data, decision, contrast_data=None, dimensions=1, pc_xi=0.25, return_tuples=False,
                   interesting_vars=None, require_all_vars=False):

    obj_count, var_count = data.shape

    if len(decision) != obj_count:
        raise Exception("Length of decision is not equal to the number of rows in data.")

    seed = -1 if seed is None else seed
    range_ = -1 if range_ is None else range_

    iv_count = 0 if interesting_vars is None else len(interesting_vars)
    in_interesting = (c_int * iv_count)(*interesting_vars) if interesting_vars else (c_int * iv_count)()

    if contrast_data is not None:
        if contrast_data.shape[0] != data.shape[0]:
            raise Exception("contrast_data must have the same object count as data")
        n_contrast_variables = contrast_data.shape[1]
    else:
        n_contrast_variables = 0

    divisions = len(set(data)) - 1
    if contrast_data is not None:
        contrast_divisions = len(set(contrast_data)) - 1
        if contrast_divisions != divisions:
            raise Exception("Contrast data has a different number of classes.")

    result = funs.compute_max_ig_discrete(obj_count, var_count, data, n_contrast_variables, contrast_data, decision, c_int(dimensions),
                                    c_int(divisions), c_double(pc_xi), c_bool(return_tuples), iv_count,
                                    in_interesting, c_bool(require_all_vars))

    return handle_error(result)

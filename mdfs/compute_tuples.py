from ctypes import c_int, c_double, c_bool, c_char_p, POINTER, Structure
from enum import Enum
import numpy as np

from .c import funs, data_type, optional_decision_type
from .internal_common import handle_error


class StatMode(int, Enum):
    H = 0
    MI = 1
    VI = 2


class ComputeTuplesResult(Structure):
    _fields_ = [('_error', c_char_p),
                ('_count', c_int),
                ('_n_dims', c_int),
                ('_igs', POINTER(c_double)),
                ('_tuples', POINTER(c_int)),
                ('_vars', POINTER(c_int))]

    @property
    def vars(self):
        if self._vars:
            return np.ctypeslib.as_array(self._vars, (self._count,))

    @property
    def tuples(self):
        if self._tuples:
            # NOTE: transpose to give the expected view on the data (F_CONTIGUOUS)
            return np.ctypeslib.as_array(self._tuples, shape=(self._n_dims, self._count)).transpose()

    @property
    def stats(self):
        if self._igs:
            if self._tuples:
                # per-tuple mode
                return np.ctypeslib.as_array(self._igs, (self._count,))
            else:
                # matrix mode
                n = self._count
                # NOTE: transpose to give the expected view on the data (F_CONTIGUOUS)
                return np.ctypeslib.as_array(self._igs, (n, n)).transpose()

    def __del__(self):
        funs.freeDoubleArray(self._igs)
        funs.freeIntArray(self._tuples)
        funs.freeIntArray(self._vars)


i_type = np.ctypeslib.ndpointer(float, ndim=1, flags='ALIGNED, F_CONTIGUOUS')


def _from_param_i(cls, obj):
    if obj is None:
        return obj
    return i_type.from_param(obj)


optional_i_type = type(
    'optional_i_type',
    (i_type,),
    {'from_param': classmethod(_from_param_i)}
)

funs.compute_tuples.argtypes = [
    c_int,  # obj_count
    c_int,  # var_count
    data_type,  # data
    optional_decision_type,  # decision
    c_int,  # dimensions
    c_int,  # divisions
    c_int,  # discretizations
    c_int,  # seed
    c_double,  # range
    c_double,  # pseudocount
    c_int,  # int_vars_count
    POINTER(c_int),  # interesting_vars
    c_bool,  # require_all_vars
    c_double,  # ig_thr
    optional_i_type,  # I_lower
    c_bool,  # return_matrix
    c_int,  # stat_mode
    c_bool,  # average
]
funs.compute_tuples.restype = ComputeTuplesResult

funs.compute_tuples_discrete.argtypes = [
    c_int,  # obj_count
    c_int,  # var_count
    data_type,  # data
    optional_decision_type,  # decision
    c_int,  # dimensions
    c_int,  # divisions
    c_double,  # pseudocount
    c_int,  # int_vars_count
    POINTER(c_int),  # interesting_vars
    c_bool,  # require_all_vars
    c_double,  # ig_thr
    optional_i_type,  # I_lower
    c_bool,  # return_matrix
    c_int,  # stat_mode
]
funs.compute_tuples_discrete.restype = ComputeTuplesResult


def compute_tuples(data, decision=None, *, dimensions=2, divisions=1,
                   discretizations=1, seed=None, range_=None, pc_xi=0.25,
                   ig_thr=0.0, i_lower=None,
                   interesting_vars=None, require_all_vars=False, return_matrix=False,
                   stat_mode=StatMode.MI,
                   average=False) -> ComputeTuplesResult:

    # ensure stat_mode is a proper StatMode
    stat_mode = StatMode(stat_mode)

    if dimensions > 2:
        if decision is not None and stat_mode == StatMode.VI:
            raise Exception("Unable to compute target information difference in higher than 2 dimensions (it is not well defined)")
        if decision is None and stat_mode != StatMode.H:
            raise Exception("Unable to compute decisionless non-entropy statistics in higher than 2 dimensions (they are not defined)")

    obj_count, var_count = data.shape
    iv_count = 0 if interesting_vars is None else len(interesting_vars)

    if decision is not None:
        if len(decision) != obj_count:
            raise Exception("Length of decision is not equal to the number of rows in data.")

    seed = -1 if seed is None else seed
    range_ = -1 if range_ is None else range_

    if i_lower is not None:
        if dimensions != 2:
            raise Exception("Currently, i_lower is supported only in 2D.")
        if len(i_lower) != var_count:
            raise Exception("i_lower should have the length of variables count.")

    result = funs.compute_tuples(obj_count, var_count, data, decision,
                                 c_int(dimensions), c_int(divisions), c_int(discretizations),
                                 c_int(seed), c_double(range_), c_double(pc_xi), iv_count,
                                 interesting_vars, c_bool(require_all_vars), c_double(ig_thr),
                                 i_lower, return_matrix, stat_mode, average)
    return handle_error(result)


def compute_tuples_discrete(data, decision=None, *, dimensions=2, pc_xi=0.25,
                   ig_thr=0.0, i_lower=None,
                   interesting_vars=None, require_all_vars=False, return_matrix=False,
                   stat_mode=StatMode.MI) -> ComputeTuplesResult:

    # ensure stat_mode is a proper StatMode
    stat_mode = StatMode(stat_mode)

    if dimensions > 2:
        if decision is not None and stat_mode == StatMode.VI:
            raise Exception("Unable to compute target information difference in higher than 2 dimensions (it is not well defined)")
        if decision is None and stat_mode != StatMode.H:
            raise Exception("Unable to compute decisionless non-entropy statistics in higher than 2 dimensions (they are not defined)")

    obj_count, var_count = data.shape
    iv_count = 0 if interesting_vars is None else len(interesting_vars)

    if decision is not None:
        if len(decision) != obj_count:
            raise Exception("Length of decision is not equal to the number of rows in data.")

    if i_lower is not None:
        if dimensions != 2:
            raise Exception("Currently, i_lower is supported only in 2D.")
        if len(i_lower) != var_count:
            raise Exception("i_lower should have the length of variables count.")

    divisions = len(set(data)) - 1

    result = funs.compute_tuples_discrete(obj_count, var_count, data, decision,
                                 c_int(dimensions), c_int(divisions), c_double(pc_xi), iv_count,
                                 interesting_vars, c_bool(require_all_vars), c_double(ig_thr),
                                 i_lower, return_matrix, stat_mode)
    return handle_error(result)

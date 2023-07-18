from ctypes import c_int, c_double, c_bool, c_char_p, POINTER, Structure
import numpy as np

from .c import funs
from .internal_common import handle_error


p_value_type = np.ctypeslib.ndpointer(float, ndim=1, flags='ALIGNED, F_CONTIGUOUS')


class FitPValueResult(Structure):
    _fields_ = [
        ('_error', c_char_p),
        ('_n_vars', c_int),
        ('_p_values', POINTER(c_double)),
        ('sq_dev', c_double),
        ('dist_param', c_double),
        ('err_param', c_double),
    ]

    @property
    def p_values(self):
        if self._p_values:
            return np.ctypeslib.as_array(self._p_values, (self._n_vars,))

    def __del__(self):
        funs.freeDoubleArray(self._p_values)


funs.fit_p_value.argtypes = [
    c_int,  # var_count
    p_value_type,  # chisq
    c_int,  # contrast_count
    p_value_type,  # chisq_contrast
    c_bool,  # exponential fit
    c_int,  # irr_vars_num
    c_int,  # ign_low_ig_vars_num
    c_int,  # min_irr_vars_num
    c_int,  # max_ign_low_ig_vars_num
    c_int,  # search_points
]
funs.fit_p_value.restype = FitPValueResult


def fit_p_value(chisq, chisq_contrast, *,
                exponential_fit,
                irr_vars_num=None, ign_low_ig_vars_num=None,
                min_irr_vars_num=None, max_ign_low_ig_vars_num=None,
                search_points=8):

    n_vars = len(chisq)
    n_contrasts = len(chisq_contrast)

    irr_vars_num = -1 if irr_vars_num is None else irr_vars_num
    ign_low_ig_vars_num = -1 if ign_low_ig_vars_num is None else ign_low_ig_vars_num
    min_irr_vars_num = -1 if min_irr_vars_num is None else min_irr_vars_num
    max_ign_low_ig_vars_num = -1 if max_ign_low_ig_vars_num is None else max_ign_low_ig_vars_num

    result = funs.fit_p_value(c_int(n_vars), chisq, c_int(n_contrasts), chisq_contrast,
                                 c_bool(exponential_fit),
                                 c_int(irr_vars_num), c_int(ign_low_ig_vars_num),
                                 c_int(min_irr_vars_num), c_int(max_ign_low_ig_vars_num),
                                 c_int(search_points))

    return handle_error(result)

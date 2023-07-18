# MDFS (MultiDimensional Feature Selection) for Python

MDFS is a library to assist in MultiDimensional Feature Selection (MDFS),
i.e. feature selection that accounts for multidimensional interactions
in the dataset. To learn more about MDFS, please visit the
[MDFS website][mdfs-web].

This project is the implementation of the MDFS library for Python.
Functionality-wise, it is aligned with the
[R version of the MDFS library][mdfs-r], but the interface differs to
make it more native to the Python ecosystem (i.e. _pythonic_) and to
free it from early assumptions carried on for backward compatibility in R.

## License

This software is released the same as the R MDFS library: under the
[GNU General Public License (GPL) v3][gpl-3].

## Copyright

The copyrights are held by Radosław Piliszek (the package maintainer
and author), Abraham Kaczmarski (major contributor to the new interface),
Krzysztof Mnich and Witold Rudnicki (authors of the MDFS method).

## Changelog

See [the common changelog][changelog].

## Library structure

The library consists of a single package module: `mdfs`, which exports
all the user-facing functionality.

## Introduction for beginners

The `mdfs` package module needs to be imported. Then, the main function
to run is, aptly named, `run`. It accepts a numpy data matrix data and
its corresponding decision, and returns a dictionary with the details of
analysis, including the entry for `relevant_variables` which gives the
indices of variables deemed relevant under chosen conditions.

## Interface differences between R and Python

### Function names

The following list gives the translation between R functions and their
Python counterparts.

- `MDFS` = `run`
- `ComputeMaxInfoGains` = `compute_max_ig`
- `ComputeInterestingTuples` = `compute_tuples`
- `ComputePValue` = `fit_p_value`
- `Discretize` = `discretize`
- `GetRange` = `get_suggested_range`
- `GenContrastVariables` = `gen_contrast_variables`

### Function parameter names

Function parameter names have been adjusted to avoid the dot (`.`),
replacing it with an underscore (`_`).

### No global seed in Python

There is no global seed in use. All functions depending on PRNG take
a `seed` parameter.

### Quirks

Due to the way the Python-C interface is implemented in this library with
`numpy` views, there is one quirk to be aware of. Functions returning
a `Structure` subclass object do so without incurring a copy. Properties
present on such objects return views, not copies. These views do not protect
the result from being garbage collected (i.e., think of them as weak
references to the underlying data). Thus, to avoid freed memory reads,
keep the original structures around when using these views or copy
data elsewhere as necessary.
This quirk might be lifted in the future.


[mdfs-web]: https://www.mdfs.it/
[mdfs-r]: https://cran.r-project.org/package=MDFS
[gpl-3]: https://www.gnu.org/licenses/gpl-3.0.en.html
[changelog]: https://www.mdfs.it/CHANGELOG

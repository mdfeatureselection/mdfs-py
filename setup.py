from setuptools import setup, Extension

from distutils.command.build_ext import build_ext as _build_ext
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


# inspired by https://github.com/joerick/python-ctypes-package-sample


class CTypesExtension(Extension):
    pass


class build_ext(_build_ext):
    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypesExtension)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            return ext_name + ".so"
        return super().get_ext_filename(ext_name)


class bdist_wheel_abi_none(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = _bdist_wheel.get_tag(self)
        return "py3", "none", plat


setup(
    packages=["mdfs"],
    ext_modules=[
        CTypesExtension(
            "mdfs.libmdfs",
            [
                "shared/interface/compute_max_ig.cpp",
                "shared/interface/compute_tuples.cpp",
                "shared/interface/get_suggested_range.cpp",
                "shared/interface/free_helpers.cpp",
                "shared/interface/discretize.cpp",
                "shared/interface/fit_p_value.cpp",
                "shared/interface/gen_contrast_variables.cpp",
                "shared/src/cpu/common.cpp",
                "shared/src/cpu/discretize.cpp",
            ],
        ),
    ],
    cmdclass={"build_ext": build_ext, "bdist_wheel": bdist_wheel_abi_none},
)

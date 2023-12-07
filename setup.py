#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

## usage: python setup.py build_ext --inplace

import os
import subprocess
import sys

from setuptools import Extension, setup, find_packages
from torch.utils import cpp_extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

extra_compile_args = ["-std=c++11", "-O3"]


class NumpyExtension(Extension):
    """Source: https://stackoverflow.com/a/54128391"""

    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy

        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs


extensions = [
    NumpyExtension(
        "src.dataload.data_utils_fast",
        sources=["src/dataload/data_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        "src.dataload.token_block_utils_fast",
        sources=["src/dataload/token_block_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]



if __name__ == "__main__":
    setup(
        name="wav2vec-rf",
        ext_modules=cythonize(extensions),
        cmdclass= {"build_ext": build_ext},
    )
import petsc4py
from setuptools import Extension, setup

import petsctools

extension = Extension(
    name="fast",
    language="c",
    sources=["fast.pyx"],
    include_dirs=[
        petsc4py.get_include(),
        *petsctools.get_petsc_dirs(subdir="include"),
    ],
    library_dirs=petsctools.get_petsc_dirs(subdir="lib"),
    runtime_library_dirs=petsctools.get_petsc_dirs(subdir="lib"),
    libraries=["petsc", "mpi"],
)

setup(ext_modules=[extension])

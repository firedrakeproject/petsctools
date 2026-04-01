from setuptools import setup, Extension

import petsctools


extension = Extension(
    name="fast",
    language="c",
    sources=["fast.pyx"],
    include_dirs=petsctools.get_petsc_dirs(subdir="include"),
    library_dirs=petsctools.get_petsc_dirs(subdir="lib"),
    runtime_library_dirs=petsctools.get_petsc_dirs(subdir="lib"),
    libraries=["petsc"],
)

setup(ext_modules=[extension])

from Cython.Build import cythonize
from setuptools import setup, Extension

import petsctools
import os
import petsc4py


from dataclasses import dataclass, field


@dataclass
class ExternalDependency:
    ''' This dataclass stores the relevant information for the compiler as fields
    that correspond to the keyword arguments of `Extension`. For convenience it
    also implements addition and `**` unpacking.
    '''
    include_dirs: list[str] = field(default_factory=list, init=True)
    extra_compile_args: list[str] = field(default_factory=list, init=True)
    libraries: list[str] = field(default_factory=list, init=True)
    library_dirs: list[str] = field(default_factory=list, init=True)
    extra_link_args: list[str] = field(default_factory=list, init=True)
    runtime_library_dirs: list[str] = field(default_factory=list, init=True)

    def __add__(self, other):
        combined = {}
        for f in self.__dataclass_fields__.keys():
            combined[f] = getattr(self, f) + getattr(other, f)
        return self.__class__(**combined)

    def keys(self):
        return self.__dataclass_fields__.keys()

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"Key {key} not present")



petsc_dir = petsctools.get_petsc_dir()
petsc_arch = petsctools.get_petsc_arch()
petsc_dirs = [petsc_dir, os.path.join(petsc_dir, petsc_arch)]
petsc_ = ExternalDependency(
    libraries=["petsc"],
    include_dirs=[petsc4py.get_include()] + [os.path.join(d, "include") for d in petsc_dirs],
    library_dirs=[os.path.join(petsc_dirs[-1], "lib")],
    runtime_library_dirs=[os.path.join(petsc_dirs[-1], "lib")],
)

mods = [
        Extension(
        name="cython_demo",
        language="c",
        sources=[os.path.join("cython_demo.pyx")],
        **(petsc_),
        annotate=True,
    )
]


setup(ext_modules=mods)

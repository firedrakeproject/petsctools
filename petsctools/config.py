import functools
import os
import subprocess

from petsctools.exceptions import PetscToolsException


class MissingPetscException(PetscToolsException):
    pass


def get_config():
    try:
        import petsc4py

        return petsc4py.get_config()
    except ImportError:
        pass

    if "PETSC_DIR" in os.environ:
        petsc_dir = os.environ["PETSC_DIR"]
        petsc_arch = os.getenv("PETSC_ARCH")  # can be empty
        return {"PETSC_DIR": petsc_dir, "PETSC_ARCH": petsc_arch}
    else:
        raise MissingPetscException(
            "PETSc cannot be found, please set PETSC_DIR (and maybe "
            "PETSC_ARCH)"
        )


def get_petsc_dir():
    return get_config()["PETSC_DIR"]


def get_petsc_arch():
    return get_config()["PETSC_ARCH"]


@functools.lru_cache()
def get_petscvariables():
    """Return PETSc's configuration information."""
    path = os.path.join(
        get_petsc_dir(),
        get_petsc_arch() or "",
        "lib/petsc/conf/petscvariables",
    )
    with open(path) as f:
        pairs = [line.split("=", maxsplit=1) for line in f.readlines()]
    return {k.strip(): v.strip() for k, v in pairs}


@functools.lru_cache()
def get_petscconf_h():
    """Get dict of PETSc include variables from the file:
    $PETSC_DIR/$PETSC_ARCH/include/petscconf.h

    The ``#define`` and ``PETSC_`` prefix are dropped in the dictionary key.

    The result is memoized to avoid constantly reading the file.
    """
    path = os.path.join(
        get_petsc_dir(), get_petsc_arch() or "", "include/petscconf.h"
    )
    with open(path) as f:
        splitlines = (
            line.removeprefix("#define PETSC_").split(" ", maxsplit=1)
            for line in filter(
                lambda x: x.startswith("#define PETSC_"), f.readlines()
            )
        )
    return {k: v.strip() for k, v in splitlines}


@functools.lru_cache()
def get_external_packages():
    """Return a list of PETSc external packages that are installed."""
    # The HAVE_PACKAGES variable uses delimiters at both ends
    # so we drop the empty first and last items
    return get_petscconf_h()["HAVE_PACKAGES"].split(":")[1:-1]


def _get_so_dependencies(filename):
    """Get all the dependencies of a shared object library."""
    # Linux uses `ldd` to look at shared library linkage, MacOS uses `otool`
    try:
        program = ["ldd"]
        cmd = subprocess.run([*program, filename], stdout=subprocess.PIPE)
        # Filter out the VDSO and the ELF interpreter on Linux
        results = [
            line
            for line in cmd.stdout.decode("utf-8").split("\n")
            if "=>" in line
        ]
        return [line.split()[2] for line in results]
    except FileNotFoundError:
        program = ["otool", "-L"]
        cmd = subprocess.run([*program, filename], stdout=subprocess.PIPE)
        # MacOS puts garbage at the beginning and end of `otool` output
        return [
            line.split()[0]
            for line in cmd.stdout.decode("utf-8").split("\n")[1:-1]
        ]


@functools.lru_cache()
def get_blas_library():
    """Get the path to the BLAS library that PETSc links to."""
    from petsc4py import PETSc

    petsc_py_dependencies = _get_so_dependencies(PETSc.__file__)
    library_names = ["blas", "libmkl"]
    for filename in petsc_py_dependencies:
        if any(name in filename for name in library_names):
            return filename

    # On newer MacOS versions, the PETSc Python extension library doesn't link
    # to BLAS or MKL directly, so we check the PETSc C library.
    petsc_c_library = [f for f in petsc_py_dependencies if "libpetsc" in f][0]
    petsc_c_dependencies = _get_so_dependencies(petsc_c_library)
    for filename in petsc_c_dependencies:
        if any(name in filename for name in library_names):
            return filename

    return None

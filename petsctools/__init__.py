import os
import sys
import warnings

import petsc4py
from packaging.specifiers import SpecifierSet
from packaging.version import Version


class PetscToolsException(Exception):
    pass


class InvalidEnvironmentException(PetscToolsException):
    pass


class InvalidPetscVersionException(PetscToolsException):
    pass


def init(argv=None, *, version_spec=""):
    """Initialise PETSc."""
    if argv is None:
        argv = sys.argv

    petsc4py.init(argv)
    _check_environment()
    _check_petsc_version(version_spec)


def _check_environment():
    config = get_config()
    petsc_dir = config["PETSC_DIR"]
    petsc_arch = config["PETSC_ARCH"]
    if (
        os.environ.get("PETSC_DIR", petsc_dir) != petsc_dir
        or os.environ.get("PETSC_ARCH", petsc_arch) != petsc_arch
    ):
        raise InvalidEnvironmentException(
            "PETSC_DIR and/or PETSC_ARCH are set but do not match the expected values "
            f"of '{petsc_dir}' and '{petsc_arch}'"
        )


def _check_petsc_version(version_spec):
    import petsc4py.PETSc

    version_spec = SpecifierSet(version_spec)

    petsc_version = Version("{}.{}.{}".format(*petsc4py.PETSc.Sys.getVersion()))
    petsc4py_version = Version(petsc4py.__version__)

    if petsc_version != petsc4py_version:
        warnings.warn(
            f"The PETSc version ({petsc_version}) does not match the petsc4py version "
            f"({petsc4py_version}), this may cause unexpected behaviour")

    if petsc_version not in version_spec:
        raise InvalidPetscVersionException(
            f"PETSc version ({petsc_version}) does not obey the provided constraints "
            f"({version_spec}). You probably need to rebuild PETSc or upgrade your package."
        )
    if petsc4py_version not in version_spec:
        raise InvalidPetscVersionException(
            f"petsc4py version ({petsc4py_version}) does not obey the provided constraints "
            f"({version_spec}). You probably need to rebuild petsc4py or upgrade your package."
        )


get_config = petsc4py.get_config


def get_petsc_dir():
    return get_config()["PETSC_DIR"]


def get_petsc_arch():
    return get_config()["PETSC_ARCH"]


def get_petscvariables():
    """Return PETSc's configuration information."""
    path = os.path.join(get_petsc_dir(), get_petsc_arch(), "lib/petsc/conf/petscvariables")
    with open(path) as f:
        pairs = [line.split("=", maxsplit=1) for line in f.readlines()]
    return {k.strip(): v.strip() for k, v in pairs}

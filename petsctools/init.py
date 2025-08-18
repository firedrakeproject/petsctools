import os
import sys
import warnings
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version

import petsctools.options
from petsctools.exceptions import PetscToolsException


class InvalidEnvironmentException(PetscToolsException):
    pass


class InvalidPetscVersionException(PetscToolsException):
    pass


def init(argv=None, *, version_spec=""):
    """Initialise PETSc."""
    import petsc4py

    if argv is None:
        argv = sys.argv

    petsc4py.init(argv)
    check_environment_matches_petsc4py_config()
    check_petsc_version(version_spec)

    # Save the command line options so they may be inspected later
    from petsc4py import PETSc

    petsctools.options._commandline_options = frozenset(
        PETSc.Options().getAll()
    )

    return PETSc


def check_environment_matches_petsc4py_config():
    import petsc4py

    config = petsc4py.get_config()
    petsc_dir = config["PETSC_DIR"]
    petsc_arch = config["PETSC_ARCH"]
    if (
        Path(os.environ.get("PETSC_DIR", petsc_dir)) != Path(petsc_dir)
        or os.environ.get("PETSC_ARCH", petsc_arch) != petsc_arch
    ):
        raise InvalidEnvironmentException(
            "PETSC_DIR and/or PETSC_ARCH are set but do not match the "
            f"expected values of '{petsc_dir}' and '{petsc_arch}' from "
            "petsc4py"
        )


def check_petsc_version(version_spec) -> None:
    import petsc4py.PETSc

    version_spec = SpecifierSet(version_spec)

    petsc_version = Version(
        "{}.{}.{}".format(*petsc4py.PETSc.Sys.getVersion())
    )
    petsc4py_version = Version(petsc4py.__version__)

    if petsc_version != petsc4py_version:
        warnings.warn(
            f"The PETSc version ({petsc_version}) does not match the petsc4py "
            f"version ({petsc4py_version}), this may cause unexpected "
            "behaviour"
        )

    if petsc_version not in version_spec:
        raise InvalidPetscVersionException(
            f"PETSc version ({petsc_version}) does not obey the provided "
            f"constraints ({version_spec}). You probably need to rebuild "
            "PETSc or upgrade your package."
        )
    if petsc4py_version not in version_spec:
        raise InvalidPetscVersionException(
            f"petsc4py version ({petsc4py_version}) does not obey the "
            f"provided constraints ({version_spec}). You probably need to "
            "rebuild petsc4py or upgrade your package."
        )

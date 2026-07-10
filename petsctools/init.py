import os
import sys
import types
import warnings
from collections.abc import Sequence
from pathlib import Path

import petsc4py
import petsc4py.lib
from packaging.specifiers import SpecifierSet
from packaging.version import Version

import petsctools.options
from petsctools.exceptions import (
    InvalidEnvironmentException, InvalidPetscVersionException
)


def init(
    argv: Sequence[str] | None = None,
    *,
    version_spec: SpecifierSet | str = "",
) -> types.ModuleType:
    """Initialise PETSc.

    Parameters
    ----------
    argv
        Command line options to be passed to PETSc at initialisation. If
        unspecified then `sys.argv` is used.
    version_spec
        String describing PETSc version constraints. For example
        '>=3.25.2,<3.26'.

    Returns
    -------
    types.ModuleType
        The `petsc4py.PETSc` module. This is convenient for avoiding
        boilerplate.

    """
    if argv is None:
        argv = sys.argv

    # We have to do this dance because we need to access petsc4py.PETSc without
    # initialising PETSc. This is what happens in
    # https://gitlab.com/petsc/petsc/-/blob/main/src/binding/petsc4py/src/petsc4py/PETSc.py
    PETSc = petsc4py.lib.ImportPETSc()
    if PETSc.Sys.isInitialized():
        warnings.warn(
            "Calling petsctools.init but PETSc has already been initialised, "
            "any command line options will be ignored.",
            stacklevel=2,
        )
    else:
        PETSc._initialize(argv)

    check_environment_matches_petsc4py_config()
    check_petsc_version(version_spec)

    # Save the command line options so they may be inspected later
    petsctools.options._commandline_options = frozenset(
        PETSc.Options().getAll()
    )

    return PETSc


def check_environment_matches_petsc4py_config():
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

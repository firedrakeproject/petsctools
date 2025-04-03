import os
import sys
import warnings

from packaging.specifiers import SpecifierSet
from packaging.version import Version


class PetscToolsException(Exception):
    pass


class InvalidEnvironmentException(PetscToolsException):
    pass


class InvalidPetscVersionException(PetscToolsException):
    pass


class MissingPetscException(PetscToolsException):
    pass


class MissingPetsc4pyException(PetscToolsException):
    pass


def needs_petsc4py(func):
    """Decorator indicating that a function cannot run without petsc4py installed."""
    try:
        import petsc4py
        return func
    except ImportError:
        raise MissingPetsc4pyException(
            f"petsc4py is needed for {func.__name__} but it cannot be found"
        )


@needs_petsc4py
def init(argv=None, *, version_spec=""):
    """Initialise PETSc."""
    import petsc4py

    if argv is None:
        argv = sys.argv

    petsc4py.init(argv)
    _check_environment_matches_petsc4py_config()
    _check_petsc_version(version_spec)


@needs_petsc4py
def _check_environment_matches_petsc4py_config():
    import petsc4py

    config = petsc4py.get_config()
    petsc_dir = config["PETSC_DIR"]
    petsc_arch = config["PETSC_ARCH"]
    if (
        os.environ.get("PETSC_DIR", petsc_dir) != petsc_dir
        or os.environ.get("PETSC_ARCH", petsc_arch) != petsc_arch
    ):
        raise InvalidEnvironmentException(
            "PETSC_DIR and/or PETSC_ARCH are set but do not match the expected values "
            f"of '{petsc_dir}' and '{petsc_arch}' from petsc4py"
        )


# NOTE: This doesn't strictly need petsc4py (could use PETSC_DIR etc to detect the
# version) but does for the moment
@needs_petsc4py
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
            "PETSc cannot be found, please set PETSC_DIR (and maybe PETSC_ARCH)"
        )


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

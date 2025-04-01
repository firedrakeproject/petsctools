import configparser
import functools
import os
import pathlib
import sys
import warnings

from packaging.version import Version


class PetscToolsException(Exception):
    pass


class PetscDetectionException(PetscToolsException):
    pass


class InvalidPetscException(PetscToolsException):
    pass


class PetscUninitialisedException(PetscToolsException):
    pass


_petsc_dir = None
_petsc_arch = None


def init(argv=None, bootstrap=False):
    global _petsc_dir, _petsc_arch

    if not bootstrap:
        config = configparser.ConfigParser()
        dir = pathlib.Path(__file__).parent
        with open(dir / "config.ini", "r") as f:
            config.read_file(f)

        petsc_dir = config["settings"]["petsc_dir"]
        petsc_arch = config["settings"]["petsc_arch"]

        _check_environment(petsc_dir, petsc_arch)
    else:
        # currently installing petsctools, config file does not yet exist
        try:
            import petsc4py

            petsc_config = petsc4py.get_config()
            petsc_dir = petsc_config["PETSC_DIR"]
            petsc_arch = petsc_config["PETSC_ARCH"]
            _check_environment(petsc_dir, petsc_arch)
            del petsc4py
        except ImportError:
            # last chance to detect PETSc, look for PETSC_DIR and PETSC_ARCH
            if "PETSC_DIR" not in os.environ:
                raise PetscDetectionException("PETSC_DIR is not defined so PETSc cannot be found")

            petsc_dir = os.environ["PETSC_DIR"]
            petsc_arch = os.environ.get("PETSC_ARCH", "")  # can be empty

    _check_petsc_exists(petsc_dir, petsc_arch)

    # Set the globals
    _petsc_dir = petsc_dir
    _petsc_arch = petsc_arch

    # Make petsc4py importable
    try:
        import petsc4py  # noqa: F401, F811
    except ImportError:
        # Assume petsc4py installed alongside PETSc, this may need extending in future
        sys.path.insert(0, os.path.join(petsc_dir, petsc_arch, "lib"))
        import petsc4py

    # Now initialise PETSc
    if argv is None:
        argv = sys.argv
    petsc4py.init(argv)
    import petsc4py.PETSc

    # Now check petsc4py compatibility
    petsc_config = petsc4py.get_config()
    if petsc_config["PETSC_DIR"] != petsc_dir or petsc_config["PETSC_ARCH"] != petsc_arch:
        raise InvalidPetscException("PETSC_DIR and/or PETSC_ARCH do not match with the imported petsc4py, this should not happen")

    petsc_version = Version("{}.{}.{}".format(*petsc4py.PETSc.Sys.getVersion()))
    petsc4py_version = Version(petsc4py.__version__)

    if petsc_version != petsc4py_version:
        warnings.warn(f"The PETSc version ({petsc_version}) does not match the petsc4py version ({petsc4py_version}). This may cause unexpected behaviour")


def _check_environment(petsc_dir, petsc_arch):
    if (
        os.environ.get("PETSC_DIR", petsc_dir) != petsc_dir
        or os.environ.get("PETSC_ARCH", petsc_arch) != petsc_arch
    ):
        raise InvalidPetscException(f"PETSC_DIR and/or PETSC_ARCH are set but do not match the expected values of '{petsc_dir}' and '{petsc_arch}'")


def _check_petsc_exists(petsc_dir, petsc_arch):
    return os.path.exists(os.path.join(petsc_dir, petsc_arch, "include", "petscconf.h"))


def _check_initialized(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if _petsc_dir is None or _petsc_arch is None:
            raise PetscUninitialisedException("petsctools has not been initialised, please call petsctools.init() first")
        return func(*args, **kwargs)
    return wrapper


@_check_initialized
def get_config():
    return _petsc_dir, _petsc_arch


@_check_initialized
def get_petsc_variables():
    """Attempts obtain a dictionary of PETSc configuration settings
    """
    path = os.path.join(_petsc_dir, _petsc_arch, "lib/petsc/conf/petscvariables")
    with open(path) as fh:
        # Split lines on first '=' (assignment)
        splitlines = (line.split("=", maxsplit=1) for line in fh.readlines())
    return {k.strip(): v.strip() for k, v in splitlines}

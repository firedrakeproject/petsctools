from .exceptions import PetscToolsException  # noqa: F401


class MissingPetscException(PetscToolsException):
    pass


def get_config():
    import os

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
    import os

    path = os.path.join(get_petsc_dir(), get_petsc_arch(), "lib/petsc/conf/petscvariables")
    with open(path) as f:
        pairs = [line.split("=", maxsplit=1) for line in f.readlines()]
    return {k.strip(): v.strip() for k, v in pairs}


try:
    import petsc4py  # noqa: F401
    petsc4py_found = True
    del petsc4py
except ImportError:
    petsc4py_found = False

if petsc4py_found:
    from .init import (  # noqa: F401
        InvalidEnvironmentException,
        InvalidPetscVersionException,
        init,
    )
    from .monitor import AbstractKSPMonitorFunction  # noqa: F401

del petsc4py_found

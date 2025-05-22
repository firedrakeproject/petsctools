from .config import *  # noqa: F401
from .exceptions import PetscToolsException  # noqa: F401
from .options import flatten_parameters

try:
    import petsc4py  # noqa: F401
    petsc4py_found = True
except ImportError:
    petsc4py_found = False


if petsc4py_found:
    from petsctools.init import (  # noqa: F401
        InvalidEnvironmentException,
        InvalidPetscVersionException,
        init,
    )
    from .monitor import AbstractKSPMonitorFunction  # noqa: F401
    from .options import OptionsManager

del petsc4py_found

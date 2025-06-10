from .config import (  # noqa: F401
    MissingPetscException,
    get_config,
    get_petsc_dir,
    get_petsc_arch,
    get_petscvariables,
    get_petscconf_h,
    get_external_packages,
)
from .exceptions import PetscToolsException  # noqa: F401
from .options import flatten_parameters  # noqa: F401

# Now conditionally import the functions that depend on petsc4py
from .utils import PETSC4PY_INSTALLED

if PETSC4PY_INSTALLED:
    from .config import get_blas_library  # noqa: F401
    from .init import (  # noqa: F401
        InvalidEnvironmentException,
        InvalidPetscVersionException,
        init,
    )
    from .options import OptionsManager  # noqa: F401

del PETSC4PY_INSTALLED

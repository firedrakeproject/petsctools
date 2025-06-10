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

# Now conditionally import the functions that depend on petsc4py
try:
    import petsc4py

    petsc4py_installed = True
    del petsc4py
except ImportError:
    petsc4py_installed = False

if petsc4py_installed:
    from .config import get_blas_library  # noqa: F401
    from .init import (  # noqa: F401
        InvalidEnvironmentException,
        InvalidPetscVersionException,
        init,
    )
    from .options import OptionsManager, flatten_parameters  # noqa: F401

del petsc4py_installed

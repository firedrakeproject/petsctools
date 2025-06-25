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
from .utils import PETSC4PY_INSTALLED

# Now conditionally import the functions that depend on petsc4py. If petsc4py
# is not available then attempting to access these attributes will raise an
# informative error.
if PETSC4PY_INSTALLED:
    from .config import get_blas_library  # noqa: F401
    from .init import (  # noqa: F401
        InvalidEnvironmentException,
        InvalidPetscVersionException,
        init,
    )
    from .options import OptionsManager  # noqa: F401
else:
    def __getattr__(name):
        petsc4py_attrs = {
            "get_blas_library",
            "InvalidEnvironmentException",
            "InvalidPetscVersionException",
            "init",
            "OptionsManager",
        }
        if name in petsc4py_attrs:
            raise ImportError(
                f"Cannot load '{name}' from module '{__name__}' because petsc4py "
                "is not available"
            )
        raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")

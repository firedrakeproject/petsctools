from .config import (  # noqa: F401
    get_config,
    get_external_packages,
    get_petsc_arch,
    get_petsc_dir,
    get_petsc_dirs,
    get_petscconf_h,
    get_petscvariables,
)
__all__ = [
    "get_config",
    "get_external_packages",
    "get_petsc_arch",
    "get_petsc_dir",
    "get_petsc_dirs",
    "get_petscconf_h",
    "get_petscvariables",
]
from .exceptions import (  # noqa: F401
    InvalidEnvironmentException,
    InvalidPetscVersionException,
    MissingPetscException,
    PetscToolsException,
)
__all__ += [
    "InvalidEnvironmentException",
    "InvalidPetscVersionException",
    "MissingPetscException",
    "PetscToolsException",
]
from .utils import PETSC4PY_INSTALLED
__all__ += ["PETSC4PY_INSTALLED"]

# Now conditionally import the functions that depend on petsc4py. If petsc4py
# is not available then attempting to access these attributes will raise an
# informative error.
if PETSC4PY_INSTALLED:
    from .citation import (  # noqa: F401
        add_citation,
        cite,
        print_citations_at_exit,
    )
    __all__ += [
        "add_citation",
        "cite",
        "print_citations_at_exit",
    ]
    from .config import get_blas_library  # noqa: F401
    __all__ += ["get_blas_library"]
    from .init import init  # noqa: F401
    __all__ += ["init"]
    from .options import (  # noqa: F401
        DefaultOptionSet,
        Options,
        OptionsManager,
        attach_options,
        flatten_parameters,
        get_commandline_options,
        get_options,
        has_options,
        inserted_options,
        is_set_from_options,
        petscobj2str,
        set_default_parameter,
        set_from_options,
    )
    __all__ += [
        "DefaultOptionSet",
        "Options",
        "OptionsManager",
        "attach_options",
        "flatten_parameters",
        "get_commandline_options",
        "get_options",
        "has_options",
        "inserted_options",
        "is_set_from_options",
        "petscobj2str",
        "set_default_parameter",
        "set_from_options",
    ]
    from .pc import PCBase  # noqa: F401
    __all__ += ["PCBase"]
else:

    def __getattr__(name):
        petsc4py_attrs = {
            "add_citation",
            "cite",
            "print_citations_at_exit",
            "get_blas_library",
            "init",
            "flatten_parameters",
            "get_commandline_options",
            "Options",
            "OptionsManager",
            "AppContextManager",
            "petscobj2str",
            "attach_options",
            "has_options",
            "get_options",
            "set_from_options",
            "is_set_from_options",
            "inserted_options",
            "set_default_parameter",
            "DefaultOptionSet",
            "PCBase",
            "PetscToolsAppctxException",
        }
        if name in petsc4py_attrs:
            raise ImportError(
                f"Cannot load '{name}' from module '{__name__}' because "
                "petsc4py is not available.\n"
                "If this error appears during pip install then you may have "
                "forgotten to pass --no-build-isolation"
            )
        else:
            raise AttributeError(
                f"Module '{__name__}' has no attribute '{name}'"
            )

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
from .utils import PETSC4PY_INSTALLED

# Now conditionally import the functions that depend on petsc4py. If petsc4py
# is not available then attempting to access these attributes will raise an
# informative error.
if PETSC4PY_INSTALLED:
    from .citation import (  # noqa: F401
        add_citation,
        cite,
        print_citations_at_exit,
    )
    from .config import get_blas_library  # noqa: F401
    from .init import (  # noqa: F401
        InvalidEnvironmentException,
        InvalidPetscVersionException,
        init,
    )
    from .options import (  # noqa: F401
        flatten_parameters,
        get_commandline_options,
        OptionsManager,
        petscobj2str,
        attach_options,
        has_options,
        get_options,
        set_from_options,
        is_set_from_options,
        inserted_options,
        set_default_parameter,
    )
    from .pc import PCBase  # noqa: F401
else:

    def __getattr__(name):
        petsc4py_attrs = {
            "add_citation",
            "cite",
            "print_citations_at_exit",
            "get_blas_library",
            "InvalidEnvironmentException",
            "InvalidPetscVersionException",
            "init",
            "flatten_parameters",
            "get_commandline_options",
            "OptionsManager",
            "petscobj2str",
            "attach_options",
            "has_options",
            "get_options",
            "set_from_options",
            "is_set_from_options",
            "inserted_options",
            "set_default_parameter",
            "PCBase",
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

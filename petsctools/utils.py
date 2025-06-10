try:
    import petsc4py  # noqa: F401

    PETSC4PY_INSTALLED = True
except ImportError:
    PETSC4PY_INSTALLED = False

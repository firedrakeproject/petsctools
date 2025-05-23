def petsc4py_is_installed() -> bool:
    try:
        import petsc4py
        return True
    except ImportError:
        return False

import petsctools
import pytest


@pytest.mark.skippetsc4py
def test_import_error_raised_when_petsc4py_unavailable():
    with pytest.raises(ImportError):
        petsctools.init()

    with pytest.raises(ImportError):
        petsctools.OptionsManager({}, "prefix")

    with pytest.raises(ImportError):
        petsctools.get_blas_library()

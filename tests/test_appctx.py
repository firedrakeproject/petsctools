import pytest
import petsctools
from petsctools.exceptions import PetscToolsAppctxException


@pytest.mark.skipnopetsc4py
def test_appctx():
    PETSc = petsctools.init()

    appctx = petsctools.AppContext()

    param = 10
    options = PETSc.Options()
    options['solver_param'] = appctx.add(param)

    # Can we get the key string back?
    assert str(appctx.getKey('solver_param')) == options['solver_param']

    # Can we access param via the prefixed option?
    prm = appctx.get('solver_param')
    assert prm is param

    prm = appctx['solver_param']
    assert prm is param

    # Can we set a default value?
    default = 20
    prm = appctx.get('param', default)
    assert prm is default

    # Will an invalid key raise an error
    with pytest.raises(PetscToolsAppctxException):
        appctx['param']

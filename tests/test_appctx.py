import pytest
import petsctools


@pytest.mark.skipnopetsc4py
def test_appctx():
    PETSc = petsctools.init()

    appctx = petsctools.AppContext()

    param = 10
    key = appctx.add(param)
    PETSc.Options()['solver_param'] = key

    # Can we access param via the prefixed option?
    prm = appctx.get('solver_param')
    assert prm is param

    prm = appctx['solver_param']
    assert prm is param

    # Can we access param via the key?
    prm = appctx.get(key, 20)
    assert prm is param

    prm = appctx[key]
    assert prm is param

    # Can we set a default value?
    default = 20
    prm = appctx.get('param', default)
    assert prm is default

    # Will an invalid key raise an error
    from petsctools.appctx import PetscToolsAppctxException
    with pytest.raises(PetscToolsAppctxException):
        appctx['param']

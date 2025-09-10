import pytest
import petsctools
from petsctools.exceptions import PetscToolsAppctxException


class JacobiTestPC:
    prefix = "jacobi_"
    def setFromOptions(self, pc):
        appctx = petsctools.get_appctx()
        prefix = (pc.getOptionsPrefix() or "") + self.prefix
        self.scale = appctx[prefix + "scale"]

    def apply(self, pc, x, y):
        y.pointwiseMult(x, self.scale)


@pytest.mark.skipnopetsc4py
def test_get_appctx():
    from numpy import allclose
    PETSc = petsctools.init()
    n = 4
    sizes = (n, n)

    appctx = petsctools.AppContext()

    diag = PETSc.Vec().createSeq(sizes)
    diag.setSizes((n, n))
    diag.array[:] = [1, 2, 3, 4]

    mat = PETSc.Mat().createConstantDiagonal((sizes, sizes), 1.0)

    ksp = PETSc.KSP().create()
    ksp.setOperators(mat, mat)
    petsctools.set_from_options(
        ksp,
        parameters={
            'ksp_type': 'preonly',
            'pc_type': 'python',
            'pc_python_type': f'{__name__}.JacobiTestPC',
            'jacobi_scale': appctx.add(diag)
        },
        options_prefix="myksp",
        appctx=appctx,
    )

    x, b = mat.createVecs()
    b.setRandom()

    xcheck = x.duplicate()
    xcheck.pointwiseMult(b, diag)

    with petsctools.inserted_options(ksp), petsctools.push_appctx(appctx):
        ksp.solve(b, x)

    assert allclose(x.array_r, xcheck.array_r)


@pytest.mark.skipnopetsc4py
def test_appctx_key():
    PETSc = petsctools.init()

    appctx = petsctools.AppContext()

    param = 10
    options = PETSc.Options()
    options['solver_param'] = appctx.add(param)

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

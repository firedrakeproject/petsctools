import pytest
import petsctools
from petsctools.exceptions import PetscToolsAppctxException


class JacobiTestPC:
    prefix = "jacobi_"

    def setFromOptions(self, pc):
        from petsc4py import PETSc
        prefix = (pc.getOptionsPrefix() or "") + self.prefix

        use_prefixed_appctx = PETSc.Options().getBool(
            prefix + "use_prefixed_appctx")

        if use_prefixed_appctx:
            appctx = petsctools.AppContext(prefix)
            self.scale = appctx["scale"]
        else:
            appctx = petsctools.AppContext()
            self.scale = appctx[prefix + "scale"]

    def apply(self, pc, x, y):
        y.pointwiseMult(x, self.scale)


@pytest.mark.skipnopetsc4py
@pytest.mark.parametrize("use_prefix", ["with_prefix", "without_prefix"])
@pytest.mark.parametrize("implicit_appmngr", [False, True])
def test_appctx_context_manager(use_prefix, implicit_appmngr):
    PETSc = petsctools.init()
    n = 4
    sizes = (n, n)

    diag = PETSc.Vec().createSeq(sizes)
    diag.setSizes((n, n))
    diag.array[:] = [1, 2, 3, 4]

    mat = PETSc.Mat().createConstantDiagonal((sizes, sizes), 1.0)

    ksp = PETSc.KSP().create()
    ksp.setOperators(mat, mat)

    parameters = {
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': f'{__name__}.JacobiTestPC',
        'jacobi_use_prefixed_appctx': use_prefix == "with_prefix",
    }
    if implicit_appmngr:
        appmngr = None
        parameters['jacobi_scale'] = diag
    else:
        appmngr = petsctools.AppContextManager()
        parameters['jacobi_scale'] = appmngr.add(diag)

    petsctools.set_from_options(
        ksp, parameters=parameters, options_prefix="myksp", appmngr=appmngr
    )

    x, b = mat.createVecs()
    b.setRandom()

    xcheck = x.duplicate()
    xcheck.pointwiseMult(b, diag)

    with petsctools.inserted_options(ksp):
        ksp.solve(b, x)

    assert (x - xcheck).norm() < 1e-14


@pytest.mark.skipnopetsc4py
def test_appctx_key():
    PETSc = petsctools.init()

    manager = petsctools.AppContextManager()

    prefix0_param = 10
    options = PETSc.Options()
    options['prefix0_param'] = manager.add(prefix0_param)

    appctx = petsctools.AppContext()

    # The param shouldn't be in the global dictionary yet
    with pytest.raises(PetscToolsAppctxException):
        appctx['param']

    # Can we access param via the prefixed option?
    with manager.inserted_appctx():
        prm = appctx.get('prefix0_param')
        assert prm is prefix0_param

        prm = appctx['prefix0_param']
        assert prm is prefix0_param

        # Can we set a default value?
        default = 20
        prm = appctx.get('param', default)
        assert prm is default

        # Will an invalid key raise an error
        with pytest.raises(PetscToolsAppctxException):
            appctx['param']

    # Now try with a prefixed AppContext

    # First add a param option with a different prefix
    prefix1_param = 20
    options['prefix1_param'] = manager.add(prefix1_param)

    appctx0 = petsctools.AppContext('prefix0')
    appctx1 = petsctools.AppContext('prefix1')

    with manager.inserted_appctx():
        # This should only see prefix0 entries
        prm = appctx0.get('param')
        assert prm is prefix0_param

        prm = appctx0['param']
        assert prm is prefix0_param

        # This should only see prefix1 entries
        prm = appctx1.get('param')
        assert prm is prefix1_param

        prm = appctx1['param']
        assert prm is prefix1_param


@pytest.mark.skipnopetsc4py
def test_appctx_get_all():
    item1 = object()
    item2 = object()

    optsmngr = petsctools.OptionsManager(
        {"item1": item1, "item2": item2, "other_option": 666},
        options_prefix="myprefix",
    )
    with optsmngr.inserted_options():
        appctx = petsctools.AppContext("myprefix")
        assert appctx.getAll() == {"item1": item1, "item2": item2}

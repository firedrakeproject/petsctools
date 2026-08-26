import warnings

import pytest

import petsctools


@pytest.fixture(autouse=True, scope="module")
def temporarily_remove_options():
    """Remove all options when the module is entered and reinsert them at exit.
    This ensures that options in e.g. petscrc files will not pollute the tests.
    """
    if petsctools.PETSC4PY_INSTALLED:
        PETSc = petsctools.init()
        options = PETSc.Options()
        previous_options = {
            k: v for k, v in options.getAll().items()
        }
        options.clear()
    yield
    if petsctools.PETSC4PY_INSTALLED:
        for k, v in previous_options.items():
            options[k] = v


@pytest.fixture(autouse=True)
def clear_options():
    """Clear any options from the database at the end of each test.
    """
    yield
    # PETSc already initialised by module scope fixture
    from petsc4py import PETSc
    PETSc.Options().clear()


@pytest.mark.skipnopetsc4py
@pytest.mark.parametrize("options_left", (-1, 0, 1),
                         ids=("no_options_left",
                              "options_left=0",
                              "options_left=1"))
def test_unused_options(options_left):
    """Check that unused solver options result in a warning in the log."""
    # PETSc already initialised by module scope fixture
    from petsc4py import PETSc

    if options_left >= 0:
        PETSc.Options()["options_left"] = options_left

    parameters = {
        "used": 1,
        "not_used": 2,
    }
    options = petsctools.OptionsManager(parameters, options_prefix="optobj")

    with options.inserted_options():
        _ = PETSc.Options().getInt(options.options_prefix + "used")

    # No warnings should be raised in this case.
    if options_left <= 0:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            del options
        return

    # Destroying the object will trigger the unused options warning
    with pytest.warns() as records:
        del options

    # Exactly one option is both unused and not ignored
    assert len(records) == 1
    message = str(records[0].message)

    # Does the warning include the options prefix?
    assert "optobj" in message

    # Do we only raise a warning for the unused option?
    assert "optobj_not_used" in message
    assert "optobj_used" not in message


@pytest.mark.skipnopetsc4py
def test_options_prefix():
    """Check that the OptionsManager sets the options prefix correctly.
    """
    # Generic default prefix
    options = petsctools.OptionsManager({})
    assert options.options_prefix.startswith("petsctools_")

    # User defined empty prefix
    options = petsctools.OptionsManager({}, options_prefix='')
    assert options.options_prefix == ""

    # User defined default prefix
    options = petsctools.OptionsManager({}, default_prefix="firedrake")
    assert options.options_prefix.startswith("firedrake_")

    # Explicit prefix overrides default prefix
    options = petsctools.OptionsManager({}, options_prefix="myobj")
    assert options.options_prefix == "myobj_"

    # Explicit prefix overrides default prefix
    options = petsctools.OptionsManager({}, options_prefix="myobj",
                                        default_prefix="firedrake")
    assert options.options_prefix == "myobj_"


@pytest.mark.skipnopetsc4py
def test_default_options():
    from petsc4py import PETSc

    parameters = {
        'options_left': 0,
        'unrelated_option': 0,
        'base_opt1': 1,
        'base_opt2': 2,
        'base_opt3': 3,
        'base_0_opt3': 4,
        'base_1_opt3': 5,
        'base_2_opt4': 6,
    }
    options = PETSc.Options()
    for k, v in parameters.items():
        options[k] = v

    # default_options = {"opt1": 1, "opt2": 2, "opt3": 3}
    default_option_set = petsctools.DefaultOptionSet(
        base_prefix="base", custom_prefix_endings=("0", "1", "2"))

    # test default is overriden by command line
    options0 = petsctools.OptionsManager(
        parameters={},
        options_prefix="base_0",
        default_options_set=default_option_set)

    # test defaults is overriden by command line and source-code
    options1 = petsctools.OptionsManager(
        parameters={"opt2": "7"},
        options_prefix="base_1",
        default_options_set=default_option_set)

    # test both defaults and non-defaults are picked up
    options2 = petsctools.OptionsManager(
        parameters={},
        options_prefix="base_2",
        default_options_set=default_option_set)

    assert len(options0.parameters) == 3
    assert options0.parameters["opt1"] == "1"
    assert options0.parameters["opt2"] == "2"
    assert options0.parameters["opt3"] == "4"

    assert len(options1.parameters) == 3
    assert options1.parameters["opt1"] == "1"
    assert options1.parameters["opt2"] == "7"
    assert options1.parameters["opt3"] == "5"

    assert len(options2.parameters) == 4
    assert options2.parameters["opt1"] == "1"
    assert options2.parameters["opt2"] == "2"
    assert options2.parameters["opt3"] == "3"
    assert options2.parameters["opt4"] == "6"


@pytest.mark.skipnopetsc4py
def test_python_options():
    petsctools.init()

    prefix0_param = object()
    prefix1_param = object()
    opts_manager = petsctools.OptionsManager(
        {
            "prefix0_param1": prefix0_param,
            "prefix0_param2": "some_value",
            "prefix1_param1": prefix1_param,
            "prefix1_param2": 666,
        },
        options_prefix="",
    )

    opts0 = petsctools.Options("prefix0_")
    opts1 = petsctools.Options("prefix1_")

    with opts_manager.inserted_options():
        assert opts0.get("param1") is prefix0_param
        assert opts0["param1"] is prefix0_param
        assert opts0.getAll() \
            == {"param1": prefix0_param, "param2": "some_value"}

        assert opts1.get("param1") is prefix1_param
        assert opts1["param1"] is prefix1_param
        # NOTE: ideally we would get the integer back here
        assert opts1.getAll() \
            == {"param1": prefix1_param, "param2": "666"}


class JacobiTestPC:
    prefix = "jacobi_"

    def setFromOptions(self, pc):
        from petsc4py import PETSc
        prefix = (pc.getOptionsPrefix() or "") + self.prefix

        use_prefixed_appctx = PETSc.Options().getBool(
            prefix + "use_prefixed_appctx")

        if use_prefixed_appctx:
            opts = petsctools.Options(prefix)
            self.scale = opts["scale"]
        else:
            opts = petsctools.Options()
            self.scale = opts[prefix + "scale"]

    def apply(self, pc, x, y):
        y.pointwiseMult(x, self.scale)


@pytest.mark.skipnopetsc4py
@pytest.mark.parametrize("use_prefix", ["with_prefix", "without_prefix"])
def test_python_options_ksp(use_prefix):
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
        'jacobi_scale': diag,
    }
    petsctools.set_from_options(
        ksp, parameters=parameters, options_prefix="myksp"
    )

    x, b = mat.createVecs()
    b.setRandom()

    xcheck = x.duplicate()
    xcheck.pointwiseMult(b, diag)

    with petsctools.inserted_options(ksp):
        ksp.solve(b, x)

    assert (x - xcheck).norm() < 1e-14


def test_inserted_options_dict():
    from petsc4py import PETSc
    prefix = "prefix"
    params = {
        "opt_int": 3,
        "opt_flag": None,
    }
    with petsctools.inserted_options(parameters=params, options_prefix=prefix):
        assert PETSc.Options().getInt("prefix_opt_int") == 3
        assert PETSc.Options().getBool("prefix_opt_flag")

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

    # User defined default prefix
    options = petsctools.OptionsManager({}, default_prefix="firedrake")
    assert options.options_prefix.startswith("firedrake_")

    # Explicit prefix overrides default prefix
    options = petsctools.OptionsManager({}, options_prefix="myobj")
    assert options.options_prefix.startswith("myobj_")

    # Explicit prefix overrides default prefix
    options = petsctools.OptionsManager({}, options_prefix="myobj",
                                        default_prefix="firedrake")
    assert options.options_prefix.startswith("myobj_")

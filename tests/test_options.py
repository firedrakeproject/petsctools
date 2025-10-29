import logging
import pytest
import petsctools
PETSc = petsctools.init()


def test_unused_options():
    """Check that unused solver options result in a warning in the log."""

    parameters = {
        "used": 1,
        "not_used": 2,
        "ignored": 3,
    }
    options = petsctools.OptionsManager(parameters, options_prefix="optobj")

    with options.inserted_options():
        _ = PETSc.Options().getInt(options.options_prefix + "used")

    with pytest.warns() as records:
        options.warn_unused_options(options_to_ignore={"ignored"})

    assert len(records) == 1
    message = str(records[0].message)
    # Does the warning include the options prefix?
    assert "optobj" in message
    # Do we only raise a warning for the unused option?
    # Need a space before the option because ("used" in "not_used") == True
    assert " not_used" in message
    assert " used" not in message
    assert " ignored" not in message

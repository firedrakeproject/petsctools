import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "skippetsc4py: mark as skipped unless petsc4py is not installed",
    )
    config.addinivalue_line(
        "markers",
        "skipnopetsc4py: mark as skipped unless petsc4py is installed",
    )


def pytest_collection_modifyitems(session, config, items):
    try:
        import petsc4py  # noqa: F401

        petsc4py_installed = True
    except ImportError:
        petsc4py_installed = False

    for item in items:
        if (
            item.get_closest_marker("skippetsc4py") is not None
            and petsc4py_installed
        ):
            item.add_marker(pytest.mark.skip(reason="Test requires not having petsc4py"))

        if (
            item.get_closest_marker("skipnopetsc4py") is not None
            and not petsc4py_installed
        ):
            item.add_marker(pytest.mark.skip(reason="Test requires petsc4py"))

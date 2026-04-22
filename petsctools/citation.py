"""Module containing functions for registering citations through PETSc.

The functions in this module may be used to record Bibtex citation
information and then register that a particular citation is
relevant for a particular computation.  It hooks up with PETSc's
citation registration mechanism, so that running with
``-citations`` does the right thing.

Example usage::

    petsctools.add_citation("key", "bibtex-entry-for-my-funky-method")

    ...

    if using_funky_method:
        petsctools.cite("key")

"""

_citations_database = {}


def add_citation(cite_key: str, entry: str) -> None:
    """Add a paper to the database of possible citations.

    Parameters
    ----------
    cite_key :
        The key to use.
    entry :
        The bibtex entry.

    """
    _citations_database[cite_key] = entry


def cite(cite_key: str) -> None:
    """Cite a paper.

    The paper should already have been added to the citations database using
    `add_citation`.

    Parameters
    ----------
    cite_key :
        The key of the relevant citation.

    Raises
    ------
    KeyError :
        If no such citation is found in the database.

    """
    from petsc4py import PETSc

    if cite_key in _citations_database:
        citation = _citations_database[cite_key]
        PETSc.Sys.registerCitation(citation)
    else:
        raise KeyError(
            f"Did not find a citation for '{cite_key}', please add it to the "
            "citations database"
        )


def print_citations_at_exit() -> None:
    """Print citations at the end of the program."""
    from petsc4py import PETSc

    # We devolve to PETSc for actually printing citations.
    PETSc.Options()["citations"] = None

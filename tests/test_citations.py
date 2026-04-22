import pytest

import petsctools


@pytest.mark.skipnopetsc4py
def test_cannot_cite_nonexistent_citation():
    with pytest.raises(KeyError):
        petsctools.cite("nonexistent")


@pytest.mark.skipnopetsc4py
def test_cite_citation():
    petsctools.add_citation("mykey", "myentry")
    petsctools.cite("mykey")

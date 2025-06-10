import os

import pytest

import petsctools


def test_get_config():
    config = petsctools.get_config()
    assert config.keys() == {"PETSC_DIR", "PETSC_ARCH"}
    petsc_dir = config["PETSC_DIR"]
    petsc_arch = config["PETSC_ARCH"]

    assert petsc_dir == petsctools.get_petsc_dir()
    assert petsc_arch == petsctools.get_petsc_arch()

    # Make sure that PETSC_DIR and PETSC_ARCH point to a real installation
    assert os.path.exists(
        f"{petsc_dir}/{petsc_arch or ''}/lib/petsc/conf/petscvariables"
    )


def test_get_petscvariables():
    petsctools.get_petscvariables()


def test_get_petscconf_h():
    petsctools.get_petscconf_h()


def test_get_external_packages():
    petsctools.get_external_packages()


@pytest.mark.skipnopetsc4py
def test_get_blas_library():
    petsctools.get_blas_library()

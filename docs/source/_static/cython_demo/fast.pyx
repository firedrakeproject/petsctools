import time

import cython
from petsc4py import PETSc

from petsctools cimport cpetsc


def medium():
    N: cython.int = int(1e8)
    section: PETSc.Section = PETSc.Section().create()
    section.setChart(0, N)

    start = time.time()
    i: cython.int
    for i in range(N):
        if i % 2 == 0:
            section.setDof(i, 1)
    print(f"Time elapsed: {time.time() - start}")


def fast():
    N: cython.int = int(1e8)
    section: cpetsc.PetscSection_py = PETSc.Section().create()
    section.setChart(0, N)

    start = time.time()
    i: cython.int
    for i in range(N):
        if i % 2 == 0:
            cpetsc.CHKERR(cpetsc.PetscSectionSetDof(section.sec, i, 1))
    print(f"Time elapsed: {time.time() - start}")


medium()
fast()

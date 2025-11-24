import time

from petsc4py import PETSc


def slow():
    N = int(1e8)
    section = PETSc.Section().create()
    section.setChart(0, N)

    start = time.time()
    for i in range(N):
        if i % 2 == 0:
            section.setDof(i, 1)
    print(f"Time elapsed: {time.time() - start}")


slow()

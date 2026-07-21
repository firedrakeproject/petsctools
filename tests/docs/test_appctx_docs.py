import pytest
# [appctx_docs create_mat-start]
import numpy as np
import petsctools


def diffusion_mat(sigma):
    """
    AIJ Mat for the diffusion equation with a variable diffusion coefficient.

    (I + D.T@sigma@D)u = b
    """
    from petsc4py import PETSc
    n = sigma.shape[0]
    dtype = sigma.dtype

    # index lists for CSR format
    row_start = [0]
    col_indices = []

    # top row
    idxs = [0, 1]
    col_indices.extend(idxs)
    row_start.append(row_start[-1] + len(idxs))

    # interior rows
    for j in range(1, n-1):
        idxs = [j-1, j, j+1]
        col_indices.extend(idxs)
        row_start.append(row_start[-1] + len(idxs))

    # bottom row
    idxs = [n-2, n-1]
    col_indices.extend(idxs)
    row_start.append(row_start[-1] + len(idxs))

    # values for leading and upper/lower diagonals
    diagonal = 1 + sigma.copy()
    diagonal[:-1] += sigma[1:]
    offdiags = -sigma[1:]

    # interleave diagonal entries
    Avals = np.zeros(3*n-2, dtype=dtype)
    Avals[::3] = diagonal
    Avals[1::3] = offdiags
    Avals[2::3] = offdiags

    A = PETSc.Mat().createAIJWithArrays(
        size=(n, n),
        csr=(row_start, col_indices, Avals)
    )
    return A
# [appctx_docs create_mat-end]


# [appctx_docs pc-start]
class DiffusionJacobiPC:
    prefix = "djacobi_"

    def setFromOptions(self, pc):
        prefix = (pc.getOptionsPrefix() or "") + self.prefix

        options = petsctools.Options()
        scale = options.getReal(prefix + "scale", 1.0)
        sigma = options[prefix + "sigma"]

        Ap = diffusion_mat(sigma)
        P = Ap.getDiagonal()
        P.scale(1/scale)
        self.P = P

    def apply(self, pc, x, y):
        y.pointwiseDivide(x, self.P)
# [appctx_docs pc-end]


@pytest.mark.skipnopetsc4py
def test_appctx_docs():
    # [appctx_docs create_ksp-start]
    PETSc = petsctools.init()
    np.random.seed(13)
    n = 50

    sigma_bar = 2*np.ones(n)
    sigma_prime = -0.2 + 0.4*np.random.random_sample(n)

    sigma = sigma_bar + sigma_prime
    sigma_p = sigma_bar

    A = diffusion_mat(sigma)

    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    # [appctx_docs create_ksp-end]

    # [appctx_docs set_from_options-start]
    petsctools.set_from_options(
        ksp,
        parameters={
            'ksp_converged_reason': None,
            'ksp_type': 'richardson',
            'pc_type': 'python',
            'pc_python_type': DiffusionJacobiPC,
            'djacobi_scale': 0.9,
            'djacobi_sigma': sigma_p,
        },
        options_prefix="",
    )
    # [appctx_docs set_from_options-end]

    # [appctx_docs solve-start]
    u, b = A.createVecs()
    u.zeroEntries()
    b.array[:] = np.random.random_sample(n)

    with petsctools.inserted_options(ksp):
        ksp.solve(b, u)
    # [appctx_docs solve-end]

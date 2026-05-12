import pytest
# [appctx_docs create_mat-start]
import numpy as np
import petsctools


def diffusion_mat(sigma):
    """
    AIJ Mat for the diffusion equation with variable coefficient.
    q - div(sigma*grad(q)) = b
    (I + D.T@sigma@D)q = b
    """
    from petsc4py import PETSc
    n = sigma.shape[0]
    dtype = sigma.dtype

    # top row
    row_start = [0]
    col_indices = [0, 1]
    row_start.append(row_start[-1]+2)

    # interior rows
    for j in range(1, n-1):
        col_indices.extend([j-1, j, j+1])
        row_start.append(row_start[-1]+3)

    # bottom row
    col_indices.extend([n-2, n-1])
    row_start.append(row_start[-1]+2)

    # values for leading and upper/lower diagonals
    diagonal = 1 + sigma.copy()
    diagonal[:-1] += sigma[1:]
    offdiags = -sigma[1:]

    # interleave diagonal entries
    Avals = np.zeros(3*n-2, dtype=dtype)
    Avals[::3] = diagonal
    Avals[1::3] = offdiags
    Avals[2::3] = offdiags

    amat = PETSc.Mat().createAIJWithArrays(
        size=(n, n),
        csr=(row_start, col_indices, Avals)
    )
    return amat
# [appctx_docs create_mat-end]


# [appctx_docs pc-start]
class DiagonalPC:
    prefix = "diagonal_"

    def setFromOptions(self, pc):
        from petsc4py import PETSc
        prefix = (pc.getOptionsPrefix() or "") + self.prefix

        options = PETSc.Options()
        self.scale = options.getReal(prefix + "scale", 1.0)

        appctx = petsctools.AppContext()
        self.vec = appctx[prefix + "vec"]

    def apply(self, pc, x, y):
        y.pointwiseMult(x, self.vec)
        y.scale(self.scale)
# [appctx_docs pc-end]


@pytest.mark.skipnopetsc4py
def test_appctx_docs():
    # [appctx_docs create_ksp-start]
    PETSc = petsctools.init()
    np.random.seed(13)

    n = 50
    sigma_bar = 2
    sigma = sigma_bar*(1 + 0.1*np.random.random_sample(n))

    amat = diffusion_mat(sigma)

    pdiag = PETSc.Vec().createSeq(n)
    pdiag.set(1/(1 + 2*sigma_bar))
    pdiag.setValue(n-1, 1/(1 + sigma_bar))

    ksp = PETSc.KSP().create()
    ksp.setOperators(amat)
    # [appctx_docs create_ksp-end]

    # [appctx_docs set_from_options-start]
    appmngr = petsctools.AppContextManager()

    petsctools.set_from_options(
        ksp, parameters={
            'ksp_converged_reason': None,
            'ksp_type': 'richardson',
            'ksp_richardson_scale': 0.9,
            'pc_type': 'python',
            'pc_python_type': f'{__name__}.DiagonalPC',
            'diagonal_vec': appmngr.add(pdiag),
        },
        appctx=appmngr,
        options_prefix="",
    )
    # [appctx_docs set_from_options-end]

    # [appctx_docs solve-start]
    u, b = amat.createVecs()
    u.zeroEntries()
    b.array[:] = np.random.random_sample(n)

    with petsctools.inserted_options(ksp):
        ksp.solve(b, u)
    # [appctx_docs solve-end]

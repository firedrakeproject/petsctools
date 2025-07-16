from petsc4py import PETSc


def _make_reasons(reasons):
    return {getattr(reasons, r): r
            for r in dir(reasons) if not r.startswith('_')}


PCReasons = _make_reasons(PETSc.PC.FailedReason())
KSPReasons = _make_reasons(PETSc.KSP.ConvergedReason())
SNESReasons = _make_reasons(PETSc.SNES.ConvergedReason())
TAOReasons = _make_reasons(PETSc.TAO.ConvergedReason())
TAOLineSearchReasons = _make_reasons(PETSc.TAOLineSearch.Reason())
TSReasons = _make_reasons(PETSc.TS.ConvergedReason())

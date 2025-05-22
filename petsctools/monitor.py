import abc

from petsc4py import PETSc


class AbstractKSPMonitorFunction(abc.ABC):
    """Abstract base class for a KSP monitor function.

    In order to use it it must be attached to a KSP using the function
    `PETSc.KSP.setMonitor`.

    """

    @abc.abstractmethod
    def __call__(self, ksp: PETSc.KSP, iteration: int, residual_norm: float):
        pass

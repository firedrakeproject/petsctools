import abc


def obj_name(obj):
    return f"{type(obj).__module__}.{type(obj).__name__}"


class PCBase(abc.ABC):
    needs_python_amat = False
    """Set this to True if the A matrix needs to be Python (matfree)."""

    needs_python_pmat = False
    """Set this to False if the P matrix needs to be Python (matfree)."""

    def __init__(self):
        self.initialized = False

    def setUp(self, pc):
        """Called by PETSc to update the PC.

        The first time ``setUp`` is called, the ``initialize`` method will be
        called followed by the ``update`` method. In subsequent calls to
        ``setUp`` only the ``update`` method will be called.
        """
        if not self.initialized:
            if pc.getType() != "python":
                raise ValueError("Expecting PC type python")

            A, P = pc.getOperators()
            pcname = f"{type(self).__module__}.{type(self).__name__}"
            if self.needs_python_amat:
                atype = A.type
                if atype != "python":
                    raise ValueError(
                        f"PC {pcname} needs a python type amat, not {atype}")
                self.amat = A.getPythonContext()
            if self.needs_python_pmat:
                ptype = P.type
                if ptype != "python":
                    raise ValueError(
                        f"PC {pcname} needs a python type pmat, not {ptype}")
                self.pmat = P.getPythonContext()

            self.parent_prefix = pc.getOptionsPrefix() or ""
            self.full_prefix = self.parent_prefix + self.prefix

            self.initialize(pc)
            self.initialized = True

        self.update(pc)

    @abc.abstractmethod
    def initialize(self, pc):
        """Initialize any state in this preconditioner.

        This method is only called on the first time that the ``setUp``
        method is called.
        """
        pass

    @abc.abstractmethod
    def update(self, pc):
        """Update any state in this preconditioner.

        This method is called every time that the ``setUp`` method is called.
        """
        pass

    @abc.abstractmethod
    def apply(self, pc, x, y):
        """Apply the preconditioner to x, putting the result in y.

        Both x and y are PETSc Vecs, y is not guaranteed to be zero on entry.
        """
        pass

    def applyTranspose(self, pc, x, y):
        """Apply the preconditioner transpose to x, putting the result in y.

        Both x and y are PETSc Vecs, y is not guaranteed to be zero on entry.
        """
        raise NotImplementedError(
            "Need to implement the transpose action of this PC")

    def view(self, pc, viewer=None):
        """Write a basic description of this PC.
        """
        from petsc4py import PETSc
        if viewer is None:
            return
        typ = viewer.getType()
        if typ != PETSc.Viewer.Type.ASCII:
            return
        pcname = f"{type(self).__module__}.{type(self).__name__}"
        viewer.printfASCII(
            f"Python type preconditioner {pcname}\n")

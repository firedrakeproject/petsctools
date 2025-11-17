import abc
from .exceptions import PetscToolsException


class PCBase(abc.ABC):
    """Abstract base class for python type PETSc PCs.

    This is a convenience base class that provides two common functionalities
    for python type preconditioners.

    1. Checking whether the PC operators are of python type.

       Often a python type preconditioner will rely on the Mat operators
       also being of python type. If the ``needs_python_amat`` and/or
       ``needs_python_pmat`` attributes are set then the type of the
       ``pc.getOperators()`` will be checked, and an error raised if they
       are not python type. If they are python type then their python contexts
       will be added as the attributes ``amat`` and/or ``pmat`` (e.g.
       ``A.getPythonContext()``).

    2. Separating code to initialize and update the preconditioner.

       Often there are operations to set up the preconditioner which only
       need to be run once, and operations that are needed each time the
       preconditioner is updated. The ``setUp`` method will call the user
       implemented ``initialize`` method only on the first time it is called,
       but will call the ``update`` method on every subsequent call.

    Inheriting classes should also set the ``prefix`` attribute.
    The attributes ``parent_prefix`` and ``full_prefix`` will then be set,
    where ``parent_prefix`` is the unqualified pc prefix and ``full_prefix``
    is the qualified prefix of this context (i.e. ``pc.getOptionsPrefix()``
    and ``parent_prefix+self.prefix``).

    Inheriting classes should implement the following methods:

    * ``initialize``
    * ``update``
    * ``apply``

    They should also set the following class attributes:

    * ``prefix``
    * ``needs_python_amat`` (optional, defaults to False).
    * ``needs_python_pmat`` (optional, defaults to False).

    Notes
    -----
    The ``update`` method is not called on the first call to ``setUp()``, so
    for some preconditioners it may be necessary to call ``update`` at the end
    of the ``initialize`` method.

    If the ``prefix`` attribute does not end in an underscore (``"_"``) then
    one will automatically be appended to the ``full_prefix`` attribute.
    """

    needs_python_amat = False
    """Set this to True if the A matrix needs to be Python (matfree)."""

    needs_python_pmat = False
    """Set this to False if the P matrix needs to be Python (matfree)."""

    prefix = None
    """The options prefix of this PC."""

    def __init__(self):
        self.initialized = False

    def setUp(self, pc):
        """Called by PETSc to update the PC.

        The first time ``setUp`` is called, the ``initialize`` method will be
        called followed by the ``update`` method. In subsequent calls to
        ``setUp`` only the ``update`` method will be called.
        """
        if self.initialized:
            self.update(pc)
        else:
            if pc.getType() != "python":
                raise PetscToolsException("Expecting PC type python")

            A, P = pc.getOperators()
            pcname = f"{type(self).__module__}.{type(self).__name__}"
            if self.needs_python_amat:
                if A.type != "python":
                    raise PetscToolsException(
                        f"PC {pcname} needs a python type amat, not {A.type}")
                self.amat = A.getPythonContext()
            if self.needs_python_pmat:
                if P.type != "python":
                    raise PetscToolsException(
                        f"PC {pcname} needs a python type pmat, not {P.type}")
                self.pmat = P.getPythonContext()

            if not isinstance(self.prefix, str):
                raise PetscToolsException(
                    f"{pcname}.prefix must be a str not {type(self.prefix)}")

            self.parent_prefix = pc.getOptionsPrefix() or ""
            self.full_prefix = self.parent_prefix + self.prefix
            if not self.full_prefix.endswith("_"):
                self.full_prefix += "_"

            self.initialize(pc)
            self.initialized = True

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

        This method is called the on second and later times that the
        ``setUp`` method is called.

        This method is not needed for all preconditioners and can often
        be a no-op.
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

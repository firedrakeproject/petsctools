The OptionsManager and the AppContext
-------------------------------------

The PETSc options provide a simple but powerful DSL for configuring composable solvers.
However, their main limitation is that the values of each option is limited to intrinsic C types, e.g. ``str``, ``float``, ``int``, or ``complex``.
Sometimes more advanced data is useful or essential for building a particular solver.

:class:`petsctools.AppContext <.appctx.AppContext>` fulfils this need by providing a means of passing arbitrary Python types through to Python PETSc types (e.g. Python type PCs).
In this demo we show how to use the :class:`~.appctx.AppContext` to pass data to a custom Python type PC using the variable coefficient diffusion equation as an example.


Diffusion equation with variable coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The diffusion equation with coefficient :math:`\sigma(x)` depending on the spatial coordinate is:

.. math::

    u - \nabla\cdot\left(\sigma(x)\nabla u\right) = b

We will solve this matrix with finite differences with the standard 3 point central stencil.
The particular details of the discretisation are not essential for this demo so we will be brief in the description.

If :math:`D` is the assembled matrix for the finite difference gradient stencil, and :math:`\Sigma` is a diagonal matrix with the value of the diffusion coefficient at each grid point, then the assembled matrix for the diffusion equation is:

.. math::

    Au = \left(I + D^{T}\Sigma D\right)u = b,
    \quad
    \Sigma_{ii} = \sigma(x_{i})

The following Python function takes a numpy array ``sigma`` with the value of :math:`\sigma` at each grid point and assembles a sparse (``aij``) PETSc Mat for the diffusion equation.
We will use it later to build the :class:`~petsc4py.PETSc.Mat` for a :class:`~petsc4py.PETSc.KSP` to solve the diffusion equation.

.. literalinclude:: ../../tests/docs/test_appctx_docs.py
    :language: python3
    :dedent:
    :start-after: [appctx_docs create_mat-start]
    :end-before: [appctx_docs create_mat-end]

A PC needing Python data
~~~~~~~~~~~~~~~~~~~~~~~~

Imagine a scenario where we need to solve :math:`A` multiple times and :math:`\sigma` changes slightly each time, for example if we are solving the unsteady diffusion equation with time-varying coefficients.
Rather than recomputing a preconditioner every time :math:`A` changes, we might instead find a representative :math:`\sigma_{p}` and use that to compute a preconditioner which can be reused for all solves.

For simplicity, in this demo the preconditioner :math:`P` will just be the diagonal of the assembled matrix :math:`A_{p}` for the diffusion equation with :math:`\sigma_{p}` with a simple scaling factor :math:`\omega`:

.. math::

    A_{p} = I + D^{T}\Sigma_{p} D,
    \quad
    P = \omega^{-1}\mathrm{diag}(A_{p}).

The diagonal of a matrix is clearly not expensive to compute, but in practice we would use a factorisation of :math:`A_{p}` which would be more expensive to compute and so would be more worthwhile reusing.

The preconditioner defined above is implemented with a Python type PC called ``DiffusionJacobiPC`` in the code below.
Constructing :math:`P` requires two values, :math:`\sigma_{p}` and :math:`\omega`, which must be provided by the user.

1. The scaling factor :math:`\omega` is just a real number, and can therefore be passed as usual via the :class:`PETSc.Options <petsc4py.PETSc.Options>` using the ``"djacobi_scale"`` option.

2. The diffusion coefficient at each grid point :math:`\sigma_{p}(x_{i})` is defined as a numpy array.
   This is clearly not an intrinsic type and so cannot be passed via the :class:`PETSc.Options <petsc4py.PETSc.Options>` directly.
   Instead, we access it via the :class:`~.appctx.AppContext` using the ``"djacobi_sigma"`` key.

The :class:`~petsctools.appctx.AppContext` mimics the :class:`PETSc.Options <petsc4py.PETSc.Options>` very closely, but can contain arbitrary Python data.
We will see below how to add ``sigma`` into the :class:`~petsctools.appctx.AppContext` so that it is available to the ``DiffusionJacobiPC``.

.. literalinclude:: ../../tests/docs/test_appctx_docs.py
    :language: python3
    :dedent:
    :start-after: [appctx_docs pc-start]
    :end-before: [appctx_docs pc-end]

Assembling the system
~~~~~~~~~~~~~~~~~~~~~

We specify the diffusion coefficient as some random variations :math:`\sigma'` around a constant value :math:`\overline{\sigma}`, i.e. :math:`\sigma(x) = \overline{\sigma} + \sigma'(x)`.
Assuming that :math:`\sigma'` is the component that may vary from solve to solve, we use :math:`\sigma_{p}=\overline{\sigma}`.

.. literalinclude:: ../../tests/docs/test_appctx_docs.py
    :language: python3
    :dedent:
    :start-after: [appctx_docs create_ksp-start]
    :end-before: [appctx_docs create_ksp-end]

The Options and the AppContext
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we configure ``ksp`` by passing PETSc options as key-value pairs in the ``parameters`` dictionary to :func:`petsctools.set_from_options <.options.set_from_options>`.
This function will create a :class:`petsctools.OptionsManager <.options.OptionsManager>` and attach it to ``ksp``.

We can see that common options, e.g. ``"ksp_type"`` are set as usual in the ``parameters`` dictionary.
However, when we come to the ``"djacobi_sigma"`` value we use the :class:`petsctools.AppContextManager <.appctx.AppContextManager>` class.
The :meth:`AppContextManager.add <.appctx.AppContextManager.add>` method returns a unique value which is used to associate a PETSc option to whatever data was passed to ``add``.
For example, adding the key-value pair ``"djacobi_sigma": appmngr.add(sigma_p)`` to the ``parameters`` dictionary means that, during the solve, we will be able to access ``sigma_p`` via ``AppContext()["djacobi_sigma"]``.

We then pass the :class:`~.appctx.AppContextManager` to :func:`~.options.set_from_options` so that it can be attached to ``ksp`` and its data can be made available later on during the solve.

.. literalinclude:: ../../tests/docs/test_appctx_docs.py
    :language: python3
    :dedent:
    :start-after: [appctx_docs set_from_options-start]
    :end-before: [appctx_docs set_from_options-end]

Solving the KSP
~~~~~~~~~~~~~~~

Now we come to actually solving the linear equation :math:`Au=b`.
To avoid memory leaks, :func:`~.options.set_from_options` does not permanently insert the contents of ``parameters`` and the ``appmngr`` into the global :class:`PETSc.Options <petsc4py.PETSc.Options>` and :class:`~.appctx.AppContext` databases respectively.
Instead, we use the :func:`petsctools.inserted_options <.options.inserted_options>` context manager.
On entry, this context manager inserts the contents of ``parameters`` and ``appmngr`` into the global databases, and on exit it removes them again.
This means that we need to use the :func:`~.options.inserted_options` context manager whenever these entries will be needed, for example during the solve when the KSP and PC are being set up.

.. literalinclude:: ../../tests/docs/test_appctx_docs.py
    :language: python3
    :dedent:
    :start-after: [appctx_docs solve-start]
    :end-before: [appctx_docs solve-end]

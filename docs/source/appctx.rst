AppContext demo
---------------

The PETSc options provide a simple but powerful DSL for configuring composable solvers.
However, their main limitation is that the values of each option is limited to intrinsic C types, e.g. ``str``, ``float``, ``int``, or ``complex``.
Sometimes more advanced data is useful or essential for building a particular solver.

The ``AppContext`` fulfils this need by providing a means of passing arbitrary Python types through to Python PETSc types (e.g. Python type PCs).
In this demo we show how to use the ``AppContext`` to pass data to a custom Python type PC using the variable coefficient diffusion equation as an example.


Diffusion equation with variable coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The diffusion equation with coefficient :math:`\sigma(x)` depending on the spatial coordinate is:

.. math::

    u - \nabla\cdot\left(\sigma(x)\nabla u\right) = b

We will solve this matrix with finite differences with the standard 3 point central stencil.
The particular details of the discretisation are not essential for this demo so we will be brief in the description.

If :math:`D` is the assembled matrix for the finite difference gradient stencil, and :math:`\Sigma` is a diagonal matrix with the value of the diffusion coefficient at each grid point, then the assembled matrix for the diffusion equation is:

.. math::

    \left(I + D^{T}\Sigma D\right)u = b

The following Python function takes a numpy array ``sigma`` with the value of :math:`\sigma` at each grid point and assembles a sparse (``aij``) PETSc Mat for the diffusion equation.
We will use it later to build the ``Mat`` for a ``KSP`` to solve the diffusion equation.

.. literalinclude:: ../../tests/docs/test_appctx_docs.py
    :language: python3
    :dedent:
    :start-after: [appctx_docs create_mat-start]
    :end-before: [appctx_docs create_mat-end]

A PC needing Python data
~~~~~~~~~~~~~~~~~~~~~~~~

To precondition this ``Mat`` we will use a diagonal matrix with user specified values on the diagonal.
This might be useful for example if we were to solve multiple diffusion equations with a slightly different diffusion coefficient each time.
We could build the preconditioner from some average diffusion coefficient and reuse the same PC each time.

The following code defines the Python type PC. We need two values, a ``scale`` (``float``), and a ``vec`` (``PETSc.Vec``).

1. The ``scale`` value is a ``float``, and can therefore be passed as standard via the ``PETSc.Options`` using the ``"diagonal_scale"`` option.

2. The ``vec`` value is a ``PETSc.Vec`` which specifies the diagonal matrix to use as the preconditioner.
   This is clearly not an intrinsic type to cannot be passed via the ``PETSc.Options`` directly.
   Instead we access it via the ``petsctools.AppContext`` using the ``"diagonal_vec"`` key. We will see below how to insert this value into the ``AppContext``.

.. literalinclude:: ../../tests/docs/test_appctx_docs.py
    :language: python3
    :dedent:
    :start-after: [appctx_docs pc-start]
    :end-before: [appctx_docs pc-end]

Building the KSP
~~~~~~~~~~~~~~~~

We specify a diffusion coefficient as some random variations :math:`\sigma'` around a mean value :math:`\overline{\sigma}`, i.e. :math:`\sigma(x) = \overline{\sigma} + \sigma'(x)`.
The diagonal for the preconditioner matrix is the diagonal that we would if assembling the matrix with a constant diffusion coefficient.

.. literalinclude:: ../../tests/docs/test_appctx_docs.py
    :language: python3
    :dedent:
    :start-after: [appctx_docs create_ksp-start]
    :end-before: [appctx_docs create_ksp-end]

The Options and the AppContext
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we configure the ``KSP`` using the PETSc options with ``petsctools.set_from_options``.
We can see that common options, e.g. ``ksp_type`` are set as usual in the ``parameters`` dictionary.
However, when we come to passing the ``"diagonal_vec"`` we use the ``AppContextManager`` class.
This class associates entries in the ``Options`` database with arbitrary Python objects, in this instance ``pdiag``.
We then pass the ``appmngr`` to ``set_from_options`` so that it is available later on when solving the ``KSP``.

.. literalinclude:: ../../tests/docs/test_appctx_docs.py
    :language: python3
    :dedent:
    :start-after: [appctx_docs set_from_options-start]
    :end-before: [appctx_docs set_from_options-end]

Solving the KSP
~~~~~~~~~~~~~~~

Now we come to solving the system.
The ``petsctools.inserted_options`` context manager makes sure that any options in the ``parameters`` dictionary are made available in the global ``Options`` database during the solve.
It also makes sure that any objects in the associated ``AppContextManager`` are made available in the global ``AppContext`` database, so that when we access ``"diagonal_vec"`` in the ``DiagonalPC`` we find the ``pdiag`` ``Vec``.

.. literalinclude:: ../../tests/docs/test_appctx_docs.py
    :language: python3
    :dedent:
    :start-after: [appctx_docs solve-start]
    :end-before: [appctx_docs solve-end]

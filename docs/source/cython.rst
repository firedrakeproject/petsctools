PETSc C bindings
----------------

In some circumstances it is desirable to write PETSc C code instead of using
the petsc4py Python bindings. For example:

* The overhead involved in calling Python bindings may be unacceptable. This
  may be the case in very tight loops.
* The desired API functionality is not available in petsc4py.

To support this, petsctools makes some of PETSc's C API available to use
through `Cython <https://cython.org/>`_. It does this in a similar way
to petsc4py but emphasises readability and faithfulness to the API.

Demo
~~~~

To demonstrate this, consider the following simple piece of PETSc code:

.. literalinclude:: _static/cython_demo/slow.py
    :language: python3

This code is written in Python and consists of petsc4py API calls. The
Python overhead is therefore maximised. Run on the author's machine this
code takes 4.0s to run to completion.

Compare this to the examples given in this Cython file:

.. literalinclude:: _static/cython_demo/fast.pyx
    :language: cython

Here we have two cases. The first (``medium``) is very similar to
``slow`` but is able to partially compile itself because it is
written in Cython. The second (``fast``) avoids petsc4py entirely
and instead uses the PETSc C API directly as exposed through
petsctools.

Switching to Cython and using the C API directly both lead to
performance improvements. ``medium`` takes 1.8s and ``fast``
takes 0.78s.

Compiling Cython code
~~~~~~~~~~~~~~~~~~~~~

To compile the Cython code it must be registered as a compiled
extension inside a ``setup.py`` file. A working example for the
extension provided above looks like:

.. literalinclude:: _static/cython_demo/setup.py
    :language: python3

This file should be added to your project along with a suitable
``pyproject.toml`` that indicates ``setuptools`` as the
``build-backend``. ``setuptools``, ``cython``, ``petsc4py`` and
``petsctools`` are all necessary build-time dependencies.

.. note::
   If you encounter errors like::

      /.../petsc/include/petscsys.h:124:12: fatal error: mpi.h: No such file or directory
        124 |   #include <mpi.h>
            |            ^~~~~~~
      compilation terminated.
      error: command '/usr/bin/x86_64-linux-gnu-gcc' failed with exit code 1

   then this usually means that setuptools cannot find your MPI distribution.
   To fix this simply set the environment variable ``CC=mpicc`` and
   try again.

Conventions used
~~~~~~~~~~~~~~~~

* All objects are available inside the ``cpetsc`` namespace after
  ``cimport``-ing it (e.g. ``cpetsc.PetscSectionSetDof``).

* petsc4py PETSc objects are renamed to their C API equivalent with
  a ``_py`` suffix. For example ``cpetsc.PetscSection`` represents the
  C type and ``cpetsc.PetscSection_py`` the Cython type.

* The C handle of the petsc4py objects are available through a specific
  attribute that depends on the type. Examples include:

   * ``cpetsc.Mat_py.mat``  ⟷  ``cpetsc.Mat``
   * ``cpetsc.Vec_py.vec``  ⟷  ``cpetsc.Vec``
   * ``cpetsc.IS_py.iset``  ⟷  ``cpetsc.IS``
   * ``cpetsc.PetscSection_py.sec``  ⟷  ``cpetsc.PetscSection``

  For more information you will have to refer to the `petsc4py
  source code <https://gitlab.com/petsc/petsc/-/blob/main/src/binding/petsc4py/src/petsc4py/PETSc.pxd>`__.

Adding more functions
~~~~~~~~~~~~~~~~~~~~~

Adding additional PETSc functions to petsctools is straightforward. You simply
have to add the bindings to the
`definitions file <https://github.com/firedrakeproject/petsctools/blob/main/petsctools/cpetsc.pxd>`__
in a declarative way, just as you would write any other C file. For more
information please refer to the `Cython documentation
<https://cython.readthedocs.io/en/latest/src/tutorial/pxd_files.html>`__.

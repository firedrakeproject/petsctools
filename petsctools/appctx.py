from typing import Any
import itertools
from functools import cached_property
from contextlib import contextmanager
from petsctools.exceptions import PetscToolsAppctxException

_global_appctx_data = {}
"""The global storage for user data with arbitrary python types."""


class AppContextKey(str):
    """A custom key type for AppContext.

    Warning
    -------
    This type should not be instantiated directly by the user, it
    will be generated as needed by the :class:`.AppContextManager`.

    See Also
    --------
    .AppContext
    .AppContextManager
    """

    _count = itertools.count()

    @classmethod
    def _generate_key(cls):
        return f"petsctools_appctx_key_{next(cls._count)}"


class AppContext:
    """
    A dictionary-like object to pass Python data to solvers analogously to
    passing primitive types with :class:`petsc4py.PETSc.Options`.

    The ``PETSc.Options`` dictionary can only contain primitive types (e.g.
    str, int, float, bool) as values. The ``AppContext`` allows other Python
    types to be passed into PETSc solvers while still making use of the
    namespacing provided by options prefixing.

    This class *must* be used in conjunction with the
    :class:`.AppContextManager`.  The example below shows how to use these
    classes together.

    A typical use case is a Python PC type, here called ``MyCustomPC``, which
    requires some data which is a non-primitive Python type, here an instance
    of ``MyCustomData``. Non-primitive types cannot be passed via the
    ``PETSc.Options``, so instead it is accessed via the
    ``petsctools.AppContext`` using a (fully prefixed) key.

    .. code-block:: python3

        class MyCustomData:
            ...

        class MyCustomPC:
            def setUp(self, pc):
                prefix = pc.getOptionsPrefix()
                option_key = 'custompc_data'
                self.data = petsctools.AppContext()[prefix+option_key]
                # or:
                # self.data = petsctools.AppContext(prefix)[option_key]
            ...

    Data is added to the ``AppContext`` using an :class:`.AppContextManager`,
    which stores Python data associated with a specific PETSc object (e.g. a
    KSP), in conjunction with :func:`.set_from_options` (or with the
    :class:`.OptionsManager` directly).
    Note that data *cannot* be added to the ``AppContext`` directly.

    The ``AppContextManager`` is created before calling ``set_from_options``.
    When specifying the option for the Python data (e.g. the
    ``"custompc_data"`` option key), the value in the ``parameters`` dictionary
    must be ``appmngr.add(data)`` rather than the ``data`` object itself.
    The ``AppContextManager`` is then passed to ``set_from_options``.

    .. code-block:: python3

        data = MyCustomData(...)

        appmngr = petsctools.AppContextManager()

        petsctools.set_from_options(
            ksp,
            parameters={
                'pc_type': 'python',
                'pc_python_type': 'MyCustomPC',
                'custompc_data': appmngr.add(data)},
            options_prefix='solver',
            appmngr=appmngr)

    When the :func:`.inserted_options` context manager is used for the PETSc
    object, all entries added to the ``petsctools.AppContextManager`` will be
    inserted into the global ``petsctools.AppContext`` database (with the
    ``options_prefix`` prepended to the keys), and they can be accessed inside
    the solver as shown above.
    When the context manager exits, the entries are removed from the global
    ``AppContext``.

    .. code-block:: python3

        with petsctools.inserted_options(ksp):
            ksp.solve(b, x)

    Parameters
    ----------
    prefix :
        If provided, all option keys passed to ``__getitem__`` or ``get`` will
        be prepended with this prefix before searching in the global database.
        i.e. ``AppContext("prefix_")["option"]`` is equivalent to
        ``AppContext()["prefix_option"]``.

    See Also
    --------
    .AppContextManager
    .OptionsManager
    petsc4py.PETSc.Options
    """
    def __init__(self, prefix: str | None = None):
        from petsctools.options import _validate_prefix

        # possibly append underscore or cast to str
        self._prefix = _validate_prefix(prefix or "")

    @property
    def prefix(self) -> str:
        """The prefix prepended to all keys before
        before searching the global database.
        """
        return self._prefix

    @cached_property
    def options_object(self):
        """A :class:`PETSc.Options <petsc4py.PETSc.Options>` instance."""
        from petsc4py import PETSc

        return PETSc.Options()

    def _key_from_option(self, option: str) -> AppContextKey:
        """
        Return the internal key for the PETSc option `option`.

        Parameters
        ----------
        option
            The PETSc option.

        Returns
        -------
        key
            An internal key corresponding to ``option``.
        """
        return AppContextKey(
            self.options_object.getString(self.prefix + option)
        )

    def __getitem__(self, option: str | AppContextKey, /) -> Any:
        """
        Return the value corresponding to the key ``option`` in the
        :class:`PETSc.Options <petsc4py.PETSc.Options>` dictionary.

        If this ``AppContext`` instance has a prefix then the value
        corresponding to the key ``self.prefix + option`` will be returned.

        Parameters
        ----------
        option :
            The PETSc option or key.

        Returns
        -------
        Any :
            The value for the key `option`.

        Raises
        ------
        PetscToolsAppctxException
            If the AppContext does contain a value for ``option``.
        """
        try:
            return _global_appctx_data[self._key_from_option(option)]
        except KeyError:
            raise PetscToolsAppctxException(
                f"AppContext does not have an entry for {option}"
            )

    def get(
        self, option: str | AppContextKey, default: Any | None = None
    ) -> Any:
        """
        Return the value corresponding to the key ``option`` in the
        :class:`PETSc.Options <petsc4py.PETSc.Options>` dictionary,
        or the ``default`` value if ``option`` is not found in the
        global options database.

        If this ``AppContext`` instance has a prefix then the value
        corresponding to the key ``self.prefix + option`` will be returned.

        Parameters
        ----------
        option :
            The PETSc option or key.
        default :
            The value to return if ``option`` is not in the ``AppContext``

        Returns
        -------
        Any :
            The value for the key ``option``, or ``default``.
        """
        try:
            return self[option]
        except PetscToolsAppctxException:
            return default


class AppContextManager:
    """
    Class for storing Python data associated with a particular PETSc object.

    This class must be used in conjunction with the :class:`.AppContext`
    class and :func:`.set_from_options` (or :class:`.OptionsManager`).
    See the documentation for the :class:`.AppContext` for a description
    of how these classes are used together.

    See Also
    --------
    .AppContext
    .OptionsManager
    petsc4py.PETSc.Options
    """

    def __init__(self):
        self._data = {}

    def add(self, val: Any) -> AppContextKey:
        """
        Add a value to be inserted into the global :class:`.AppContext`
        database by the :func:`.inserted_options` context manager, or
        by the ``AppContextManager`` directly with
        :meth:`.AppContextManager.inserted_appctx`.

        The autogenerated key returned by this method must only be used
        as the value for the corresponding entry in the parameters dictionary
        passed to :func:`.set_from_options`.

        Parameters
        ----------
        val
            The value to be inserted into the ``AppContext``.

        Returns
        -------
        AppContextKey
            The key to put into the ``PETSc.Options`` dictionary.
        """
        key = AppContextKey._generate_key()
        self._data[key] = val
        return key

    @contextmanager
    def inserted_appctx(self):
        """Context manager inside which the global :class:`.AppContext`
        database contains the entries added to this ``AppContextManager``.
        """
        # We don't overwrite existing entries in the global data,
        # so we need to keep track of what we do actually put in
        # so we don't accidentally remove something we shouldn't.
        to_delete = set()
        try:
            for k, v in self._data.items():
                if k not in _global_appctx_data:
                    _global_appctx_data[k] = v
                    to_delete.add(k)
            yield
        finally:
            for k in self._data:
                if k in to_delete:
                    del _global_appctx_data[k]
                    to_delete.remove(k)
            assert len(to_delete) == 0

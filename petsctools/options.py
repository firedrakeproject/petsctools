import contextlib
import itertools
import warnings

from petsctools.utils import PETSC4PY_INSTALLED
from .exceptions import PetscToolsException


def flatten_parameters(parameters, sep="_"):
    """Flatten a nested parameters dict, joining keys with sep.

    :arg parameters: a dict to flatten.
    :arg sep: separator of keys.

    Used to flatten parameter dictionaries with nested structure to a
    flat dict suitable to pass to PETSc.  For example:

    .. code-block:: python3

       flatten_parameters({"a": {"b": {"c": 4}, "d": 2}, "e": 1}, sep="_")
       => {"a_b_c": 4, "a_d": 2, "e": 1}

    If a "prefix" key already ends with the provided separator, then
    it is not used to concatenate the keys.  Hence:

    .. code-block:: python3

       flatten_parameters({"a_": {"b": {"c": 4}, "d": 2}, "e": 1}, sep="_")
       => {"a_b_c": 4, "a_d": 2, "e": 1}
       # rather than
       => {"a__b_c": 4, "a__d": 2, "e": 1}
    """
    new = type(parameters)()

    if not len(parameters):
        return new

    def flatten(parameters, *prefixes):
        """Iterate over nested dicts, yielding (*keys, value) pairs."""
        sentinel = object()
        try:
            option = sentinel
            for option, value in parameters.items():
                # Recurse into values to flatten any dicts.
                for pair in flatten(value, option, *prefixes):
                    yield pair
            # Make sure zero-length dicts come back.
            if option is sentinel:
                yield (prefixes, parameters)
        except AttributeError:
            # Non dict values are just returned.
            yield (prefixes, parameters)

    def munge(keys):
        """Ensure that each intermediate key in keys ends in sep.

        Also, reverse the list."""
        for key in reversed(keys[1:]):
            if len(key) and not key.endswith(sep):
                yield key + sep
            else:
                yield key
        else:
            yield keys[0]

    for keys, value in flatten(parameters):
        option = "".join(map(str, munge(keys)))
        if option in new:
            warnings.warn(
                f"Ignoring duplicate option: {option} (existing value "
                f"{new[option]}, new value {value})",
            )
        new[option] = value
    return new


if PETSC4PY_INSTALLED:
    from petsc4py import PETSc

    class OptionsManager:
        # What appeared on the commandline, we should never clear these.
        # They will override options passed in as a dict if an
        # options_prefix was supplied.
        commandline_options = frozenset(PETSc.Options().getAll())

        options_object = PETSc.Options()

        count = itertools.count()

        """Class that helps with managing setting petsc options.

        Parameters
        ----------
        parameters: dict
            The dictionary of parameters to use.
        options_prefix: str
            The prefix to look up items in the global options database
            (may be ``None``, in which case only entries from ``parameters``
            will be considered.
            If no trailing underscore is provided, one is appended. Hence
            ``foo_`` and ``foo`` are treated equivalently. As an exception,
            if the prefix is the empty string, no underscore is appended.

        The recommended way to use the ``OptionsManager`` is by using the
        ``attach_options``, ``set_from_options``, and ``inserted_options``
        free functions. These functions ensure that each ``OptionsManager``
        is associated to a single PETSc object.

        For detail on the previous approach of using ``OptionsManager``
        as a mixin class (where the user takes responsibility for ensuring
        a association with a single PETSc object), see below.

        To use the ``OptionsManager``:
        1. Pass a PETSc object, a parameters dictionary, and optionally
           an options prefix to ``attach_options``. This will create an
           ``OptionsManager`` and set the prefix of the PETSc object,
           but will not yet set it up.
        2. Once the object is ready, pass it to ``set_from_options``,
           which will insert the solver options into ``PETSc.Options``
           and call ``obj.setFromOptions``.
        3. The ``inserted_options`` context manager must be used when
           calling methods on the PETSc object within which solver
           options will be read, for example ``solve``.
           This will insert the provided ``parameters`` into PETSc's
           global options dictionary within the context manager, and
           remove them afterwards. This ensures that the global options
           dictionary will not grow uncontrollably if many ``OptionsManager``
           instances are used.

        .. code-block:: python3

            ksp = PETSc.KSP().create(comm=comm)
            ksp.setOperators(mat)

            attach_options(ksp, parameters=parameters,
                           options_prefix=prefix)

            # ...

            set_from_options(ksp)

            # ...

            with inserted_options(ksp):
                ksp.solve(b, x)

        To access the OptionsManager for a PETSc object directly, use
        the ``get_options`` function:

        .. code-block:: python3

            N = get_options(ksp).getInt(prefix+"N")

        Using ``OptionsManager`` as a mixin class:

        To use this, you must call its constructor with the parameters
        you want in the options database, and optionally a prefix to
        extract options from the global database.

        You then call :meth:`set_from_options`, passing the PETSc object
        you'd like to call ``setFromOptions`` on.  Note that this will
        actually only call ``setFromOptions`` the first time (so really
        this parameters object is a once-per-PETSc-object thing).

        So that the runtime monitors which look in the options database
        actually see options, you need to ensure that the options database
        is populated at the time of a ``SNESSolve`` or ``KSPSolve`` call.
        Do that using the :meth:`inserted_options` context manager.

        If using as a mixin class, call the ``OptionsManager`` methods
        directly:

        .. code-block:: python3

           self.set_from_options(self.snes)

           with self.inserted_options():
               self.snes.solve(...)

        This ensures that the options database has the relevant entries
        for the duration of the ``with`` block, before removing them
        afterwards.  This is a much more robust way of dealing with the
        fixed-size options database than trying to clear it out using
        destructors.

        This object can also be used only to manage insertion and deletion
        into the PETSc options database, by using the context manager.

        See Also
        --------
        attach_options
        has_options
        get_options
        set_from_options
        is_set_from_options
        inserted_options
        """

        def __init__(self, parameters, options_prefix):
            super().__init__()
            if parameters is None:
                parameters = {}
            else:
                # Convert nested dicts
                parameters = flatten_parameters(parameters)
            if options_prefix is None:
                self.options_prefix = "firedrake_%d_" % next(self.count)
                self.parameters = parameters
                self.to_delete = set(parameters)
            else:
                if len(options_prefix) and not options_prefix.endswith("_"):
                    options_prefix += "_"
                self.options_prefix = options_prefix
                # Remove those options from the dict that were passed on
                # the commandline.
                self.parameters = {
                    k: v
                    for k, v in parameters.items()
                    if options_prefix + k not in self.commandline_options
                }
                self.to_delete = set(self.parameters)
                # Now update parameters from options, so that they're
                # available to solver setup (for, e.g., matrix-free).
                # Can't ask for the prefixed guy in the options object,
                # since that does not DTRT for flag options.
                for k, v in self.options_object.getAll().items():
                    if k.startswith(self.options_prefix):
                        self.parameters[k[len(self.options_prefix):]] = v
            self._setfromoptions = False

        def set_default_parameter(self, key, val):
            """Set a default parameter value.

            :arg key: The parameter name
            :arg val: The parameter value.

            Ensures that the right thing happens cleaning up the options
            database.
            """
            k = self.options_prefix + key
            if k not in self.options_object and key not in self.parameters:
                self.parameters[key] = val
                self.to_delete.add(key)

        def set_from_options(self, petsc_obj):
            """Set up petsc_obj from the options database.

            :arg petsc_obj: The PETSc object to call setFromOptions on.

            Raises PetscToolsException if this method has already been called.

            Matt says: "Only ever call setFromOptions once".  This
            function ensures we do so.
            """
            if not self._setfromoptions:
                with self.inserted_options():
                    petsc_obj.setOptionsPrefix(self.options_prefix)
                    # Call setfromoptions inserting appropriate options into
                    # the options database.
                    petsc_obj.setFromOptions()
                    self._setfromoptions = True
            else:
                raise PetscToolsException(
                    "setFromOptions has already been called.")

        @contextlib.contextmanager
        def inserted_options(self):
            """Context manager inside which the petsc options database
            contains the parameters from this object."""
            try:
                for k, v in self.parameters.items():
                    self.options_object[self.options_prefix + k] = v
                yield
            finally:
                for k in self.to_delete:
                    del self.options_object[self.options_prefix + k]

    def petscobj2str(obj):
        """Return a string with a PETSc object type and prefix.

        Parameters
        ----------
        obj : petsc4py.PETSc.Object
            The object to stringify.

        Returns
        -------
        name : str
            The stringified name of the object
        """
        return f"{type(obj).__name__} ({obj.getOptionsPrefix()})"

    def attach_options(obj, parameters=None,
                       options_prefix=None):
        """Set up an OptionsManager and attach it to a PETSc Object.

        Parameters
        ----------
        obj : petsc4py.PETSc.Object
            The object to attach an OptionsManager to.
        parameters : Optional[dict]
            The dictionary of parameters to use.
        options_prefix: Optional[str]
            The options prefix to use for this object.
            See the OptionsManager documentation for more detail.

        Returns
        -------
        obj : petsc4py.PETSc.Object
            The original object.

        See Also
        --------
        OptionsManager
        """
        if has_options(obj):
            raise PetscToolsException(
                "An OptionsManager has already been"
                f"  attached to {petscobj2str(obj)}")

        options = OptionsManager(
            parameters=parameters,
            options_prefix=options_prefix)
        obj.setAttr("options", options)
        return obj

    def has_options(obj):
        """Return whether this PETSc object has an OptionsManager attached.

        Parameters
        ----------
        obj : petsc4py.PETSc.Object
            The object which may have an OptionsManager.

        Returns
        -------
        object_has_options : bool
            Whether the object has an OptionsManager.

        See Also
        --------
        OptionsManager
        """
        return (
            "options" in obj.getDict()
            and isinstance(obj.getAttr("options"), OptionsManager)
        )

    def get_options(obj):
        """Return the OptionsManager attached to this PETSc object.

        Parameters
        ----------
        obj : petsc4py.PETSc.Object
            The object to get the OptionsManager from.

        Returns
        -------
        options : OptionsManager
            The OptionsManager attached to the object.

        Raises
        ------
        PetscToolsException
            If the object does not have an OptionsManager.

        See Also
        --------
        OptionsManager
        """
        if not has_options(obj):
            raise PetscToolsException(
                "No OptionsManager attached to {petscobj2str(obj)}")
        return obj.getAttr("options")

    def set_from_options(obj):
        """Set up a PETSc object from the options in its OptionsManager.

        Parameters
        ----------
        obj : petsc4py.PETSc.Object
            The PETSc object to call setFromOptions on.

        Returns
        -------
        obj : petsc4py.PETSc.Object
            The original object.

        Raises
        ------
        PetscToolsException
            If the object does not have an OptionsManager.
        PetscToolsException
            If set_from_options has already been called for this object.

        See Also
        --------
        OptionsManager
        OptionsManager.set_from_options
        """
        if is_set_from_options(obj):
            raise PetscToolsException(
                "setFromOptions has already been"
                f" called for {petscobj2str(obj)}")
        get_options(obj).set_from_options(obj)
        return obj

    def is_set_from_options(obj):
        """
        Return whether this PETSc object has been set by the OptionsManager.

        Parameters
        ----------
        obj : petsc4py.PETSc.Object
            The object which may have been set from options.

        Returns
        -------
        object_is_set_from_options : bool
            Whether the object has previously been set from options.

        Raises
        ------
        PetscToolsException
            If the object does not have an OptionsManager.

        See Also
        --------
        OptionsManager
        """
        return get_options(obj)._setfromoptions

    @contextlib.contextmanager
    def inserted_options(obj):
        """Context manager inside which the PETSc options database
        contains the parameters from this object's OptionsManager.

        See Also
        --------
        OptionsManager
        OptionsManager.inserted_options
        """
        with get_options(obj).inserted_options():
            yield

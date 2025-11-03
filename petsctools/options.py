from __future__ import annotations

import weakref
import contextlib
import functools
import itertools
import warnings
from typing import Any, Iterable

import petsc4py

from petsctools.exceptions import (
    PetscToolsException,
    PetscToolsWarning,
    PetscToolsNotInitialisedException,
)


_commandline_options = None


def get_commandline_options() -> frozenset:
    """Return the PETSc options passed on the command line."""
    if _commandline_options is None:
        raise PetscToolsNotInitialisedException(
            "'petsctools.init' has not been called so the command line "
            "options have not been set"
        )
    return _commandline_options


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
                f"{new[option]}, new value {value})", PetscToolsWarning
            )
        new[option] = value
    return new


def _warn_unused_options(all_options: Iterable, used_options: Iterable,
                         options_prefix: str = ""):
    """
    Raise warnings for PETSc options which were not used.

    This is meant only as a weakref.finalize callback for the OptionsManager.

    Parameters
    ----------
    all_options :
        The full set of options passed to the OptionsManager.
    used_options :
        The options which were used during the OptionsManager's lifetime.
    options_prefix :
        The options_prefix of the OptionsManager.

    Raises
    ------
        PetscToolsWarning :
            For every entry in all_options which is not in used_options.
    """
    unused_options = set(all_options) - set(used_options)

    for option in sorted(unused_options):
        warnings.warn(
            f"Unused PETSc option: {options_prefix+option}",
            PetscToolsWarning
        )


class OptionsManager:
    """Class that helps with managing setting PETSc options.

    The recommended way to use the ``OptionsManager`` is by using the
    ``attach_options``, ``set_from_options``, and ``inserted_options``
    free functions. These functions ensure that each ``OptionsManager``
    is associated to a single PETSc object.

    For detail on the previous approach of using ``OptionsManager``
    as a mixin class (where the user takes responsibility for ensuring
    an association with a single PETSc object), see below.

    To use the ``OptionsManager``:

    1. Pass a PETSc object a parameters dictionary, and optionally
       an options prefix, to ``attach_options``. This will create an
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
       dictionary will not grow indefinitely if many ``OptionsManager``
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
    Do that using the `OptionsManager.inserted_options` context manager.

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

    Parameters
    ----------
    parameters
        The dictionary of parameters to use.
    options_prefix
        The prefix to look up items in the global options database
        (may be ``None``, in which case only entries from ``parameters``
        will be considered.
        If no trailing underscore is provided, one is appended. Hence
        ``foo_`` and ``foo`` are treated equivalently. As an exception,
        if the prefix is the empty string, no underscore is appended.
    default_prefix
        The base string to generate default prefixes. If options_prefix
        is not provided then a prefix is automatically generated with the
        form "{default_prefix}_{n}", where n is a unique integer. Note that
        because the unique integer is not stable any options passed via the
        command line with a matching prefix will be ignored.

    See Also
    --------
    attach_options
    has_options
    get_options
    set_from_options
    is_set_from_options
    inserted_options
    """

    count = itertools.count()

    def __init__(self, parameters: dict,
                 options_prefix: str | None = None,
                 default_prefix: str | None = None):
        super().__init__()
        if parameters is None:
            parameters = {}
        else:
            # Convert nested dicts
            parameters = flatten_parameters(parameters)
        if options_prefix is None:
            default_prefix = default_prefix or "petsctools_"
            if not default_prefix.endswith("_"):
                default_prefix += "_"
            self.options_prefix = f"{default_prefix}{next(self.count)}_"
            self.parameters = parameters
            self.to_delete = set(parameters)
        else:
            if options_prefix and not options_prefix.endswith("_"):
                options_prefix += "_"
            self.options_prefix = options_prefix
            # Remove those options from the dict that were passed on
            # the commandline.
            self.parameters = {
                k: v
                for k, v in parameters.items()
                if options_prefix + k not in get_commandline_options()
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
        # Keep track of options used between invocations of inserted_options().
        self._used_options = set()

        # Decide whether to warn for unused options
        with self.inserted_options():
            if self.options_object.getBool("options_left", False):
                weakref.finalize(self, _warn_unused_options,
                                 self.to_delete, self._used_options,
                                 options_prefix=self.options_prefix)

    def set_default_parameter(self, key: str, val: Any) -> None:
        """Set a default parameter value.

        Parameters
        ----------
        key :
            The parameter name.
        val :
            The parameter value.

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

        Raises PetscToolsWarning if this method has already been called.

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
            warnings.warn(
                "setFromOptions has already been called", PetscToolsWarning
            )

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
                if self.options_object.used(self.options_prefix + k):
                    self._used_options.add(k)
                del self.options_object[self.options_prefix + k]

    @functools.cached_property
    def options_object(self):
        from petsc4py import PETSc

        return PETSc.Options()


def petscobj2str(obj: petsc4py.PETSc.Object) -> str:
    """Return a string with a PETSc object type and prefix.

    Parameters
    ----------
    obj
        The object to stringify.

    Returns
    -------
        The stringified name of the object
    """
    return f"{type(obj).__name__} ({obj.getOptionsPrefix()})"


def attach_options(
    obj: petsc4py.PETSc.Object,
    parameters: dict | None = None,
    options_prefix: str | None = None,
    default_prefix: str | None = None
) -> None:
    """Set up an OptionsManager and attach it to a PETSc Object.

    Parameters
    ----------
    obj
        The object to attach an OptionsManager to.
    parameters
        The dictionary of parameters to use.
    options_prefix
        The options prefix to use for this object.
    default_prefix
        Base string for autogenerated default prefixes.

    See Also
    --------
    OptionsManager
    """
    if has_options(obj):
        raise PetscToolsException(
            "An OptionsManager has already been"
            f"  attached to {petscobj2str(obj)}"
        )

    options = OptionsManager(
        parameters=parameters,
        options_prefix=options_prefix,
        default_prefix=default_prefix,
    )
    obj.setAttr("options", options)


def has_options(obj: petsc4py.PETSc.Object) -> bool:
    """Return whether this PETSc object has an OptionsManager attached.

    Parameters
    ----------
    obj
        The object which may have an OptionsManager.

    Returns
    -------
        Whether the object has an OptionsManager.

    See Also
    --------
    OptionsManager
    """
    return "options" in obj.getDict() and isinstance(
        obj.getAttr("options"), OptionsManager
    )


def get_options(obj: petsc4py.PETSc.Object) -> OptionsManager:
    """Return the OptionsManager attached to this PETSc object.

    Parameters
    ----------
    obj
        The object to get the OptionsManager from.

    Returns
    -------
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
            "No OptionsManager attached to {petscobj2str(obj)}"
        )
    return obj.getAttr("options")


def set_default_parameter(
    obj: petsc4py.PETSc.Object, key: str, val: Any
) -> None:
    """Set a default parameter value in the OptionsManager of a PETSc object.

    Parameters
    ----------
    obj
        The object to get the OptionsManager from.
    key
        The options parameter name
    val
        The options parameter value

    Raises
    ------
    PetscToolsException
        If the object does not have an OptionsManager.

    See Also
    --------
    OptionsManager
    OptionsManager.set_default_parameter
    """
    get_options(obj).set_default_parameter(key, val)


def set_from_options(
    obj: petsc4py.PETSc.Object,
    parameters: dict | None = None,
    options_prefix: str | None = None,
    default_prefix: str | None = None,
) -> None:
    """Set up a PETSc object from the options in its OptionsManager.

    Calls ``obj.setOptionsPrefix`` and ``obj.setFromOptions`` whilst
    inside the ``inserted_options`` context manager, which ensures
    that all options from ``parameters`` are in the global
    ``PETSc.Options`` dictionary.

    If neither ``parameters`` nor ``options_prefix`` are provided,
    assumes that ``attach_options`` has been called with ``obj``.
    If either ``parameters`` and/or ``options_prefix`` are provided,
    then ``attach_options`` is called before setting up the ``obj``.

    Parameters
    ----------
    obj
        The PETSc object to call setFromOptions on.
    parameters
        The dictionary of parameters to use.
    options_prefix
        The options prefix to use for this object.
    default_prefix
        Base string for autogenerated default prefixes.

    Raises
    ------
    PetscToolsException
        If the neither ``parameters`` nor ``options_prefix`` are
        provided but ``obj`` does not have an OptionsManager attached.
    PetscToolsException
        If the either ``parameters`` or ``options_prefix`` are provided
        but ``obj`` already has an OptionsManager attached.
    PetscToolsWarning
        If set_from_options has already been called for this object.

    See Also
    --------
    OptionsManager
    OptionsManager.set_from_options
    """
    if has_options(obj):
        if parameters is not None or options_prefix is not None:
            raise PetscToolsException(
                f"{petscobj2str(obj)} already has an OptionsManager"
                " but parameters and/or options_prefix were provided"
                " to set_from_options"
            )
    else:
        if parameters is None and options_prefix is None:
            raise PetscToolsException(
                f"{petscobj2str(obj)} does not have an OptionsManager"
                " but neither parameters nor options_prefix were"
                " provided to set_from_options"
            )
        attach_options(
            obj, parameters=parameters,
            options_prefix=options_prefix,
            default_prefix=default_prefix,
        )

    if is_set_from_options(obj):
        warnings.warn(
            f"setFromOptions has already been called for {petscobj2str(obj)}",
            PetscToolsWarning,
        )

    get_options(obj).set_from_options(obj)


def is_set_from_options(obj: petsc4py.PETSc.Object) -> bool:
    """
    Return whether this PETSc object has been set by the OptionsManager.

    Parameters
    ----------
    obj :
        The object which may have been set from options.

    Returns
    -------
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

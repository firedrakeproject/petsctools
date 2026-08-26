from __future__ import annotations

import contextlib
import functools
import warnings
import weakref
from collections.abc import Iterable
from functools import cached_property
from typing import Any

import petsc4py

import petsctools.log
from petsctools.appctx import AppContextManager
from petsctools.exceptions import (
    PetscToolsException,
    PetscToolsNotInitialisedException,
    PetscToolsWarning,
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
                yield from flatten(value, option, *prefixes)
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

    This is meant only as a weakref.finalize callback for the
    :class:`OptionsManager`.

    Parameters
    ----------
    all_options :
        The full set of options passed to the :class:`OptionsManager`.
    used_options :
        The options which were used during the :class:`OptionsManager`'s
        lifetime.
    options_prefix :
        The options_prefix of the :class:`OptionsManager`.

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


def _validate_prefix(prefix):
    """Valid prefixes are strings ending with an underscore.
    """
    if prefix is None:
        raise ValueError("Cannot validate None as a prefix")
    prefix = str(prefix)
    if prefix and not prefix.endswith("_"):
        prefix += "_"
    return prefix


class DefaultOptionSet:
    """
    Defines a set of common default options shared by multiple PETSc objects.

    Some solvers, e.g. PCFieldsplit, create multiple subsolvers whose prefixes
    differ only by the final characters, e.g. ``'fieldsplit_0'``,
    ``'fieldsplit_1'``.  It is often useful to be able to set default options
    for these subsolvers using the un-specialised prefix e.g.
    ``'fieldsplit_ksp_type'``. However, just grabbing all options with the
    ``'fieldsplit'`` prefix will erroneously find options like ``'0_ksp_type'``
    and ``'1_ksp_type'`` that were meant for a specific subsolver.

    ``DefaultOptionSet`` defines a base prefix (e.g. ``'fieldsplit'``) and a
    set of custom prefix endings (e.g.  ``[0, 1]``). If passed to an
    :class:`OptionsManager` then any default options present in the global
    options database will be used if those options are not present either:
    in the ``parameters`` passed to the :class:`OptionsManager`; or in the
    global ``PETSc.Options`` database with the ``options_prefix`` passed to
    the :class:`OptionsManager`.

    For example, to set up a fieldsplit solver you might have the following
    options, where both fields are to use ILU as the preconditioner but each
    field uses a different KSP type.

    .. code-block:: python3

       -fieldsplit_pc_type ilu
       -fieldsplit_0_ksp_type preonly
       -fieldsplit_1_ksp_type richardson

    To create an :class:`OptionsManager` for each field you would call:

    .. code-block:: python3

       default_options_set = DefaultOptionSet(
            base_prefix='fieldsplit',
            custom_prefix_endings=(0, 1))

       fieldsplit_0_options = OptionsManager(
           parameters={},
            options_prefix="fieldsplit_0",
           default_options_set=default_options_set)

       fieldsplit_1_options = OptionsManager(
           parameters={},
            options_prefix="fieldsplit_1",
           default_options_set=default_options_set)

    Parameters
    ----------
    base_prefix :
        The prefix for the default options, which is the beginning of
        each full custom prefix.
    custom_prefix_endings :
        The ends of each individual custom prefix. Often a range of integers.

    Notes
    -----
    The base prefix and each custom prefix ending will be converted to a
    string and have an underscore appended if they do not already have one.

    See Also
    --------
    OptionsManager
    get_default_options
    attach_options
    set_from_options
    """

    def __init__(self, base_prefix: str, custom_prefix_endings: Iterable):
        if not custom_prefix_endings:
            raise ValueError("custom_prefix_endings cannot be empty")

        base_prefix = _validate_prefix(base_prefix)

        self._base_prefix = base_prefix
        self._custom_prefix_endings = tuple(
            _validate_prefix(end) for end in custom_prefix_endings)

    @property
    def base_prefix(self):
        """The prefix for the default options."""
        return self._base_prefix

    @property
    def custom_prefix_endings(self):
        """The ends of each individual custom prefix."""
        return self._custom_prefix_endings

    @cached_property
    def custom_prefixes(self):
        """The full custom prefixes."""
        return tuple(self.base_prefix + ending
                     for ending in self.custom_prefix_endings)


def get_default_options(default_options_set: DefaultOptionSet,
                        options: petsc4py.PETSc.Options | None = None) -> dict:
    """
    Extract default options for subsolvers with similar prefixes.

    Parameters
    ----------
    default_options_set
        The :class:`DefaultOptionSet` which defines the shared options.
    options
        The ``PETSc.Options`` database to use. If not provided then the global
        database will be used.

    Returns
    -------
        The dictionary of default options with the base prefix stripped.

    See Also
    --------
    DefaultOptionSet
    """
    if options is None:
        from petsc4py import PETSc
        options = PETSc.Options()

    base_prefix = default_options_set.base_prefix
    custom_prefixes = default_options_set.custom_prefixes
    custom_prefix_endings = default_options_set.custom_prefix_endings

    default_options = {
        k.removeprefix(base_prefix): v
        for k, v in options.getAll().items()
        if (k.startswith(base_prefix)
            and not any(k.startswith(prefix) for prefix in custom_prefixes))
    }
    # Sanity check, this should never happen.
    assert not any(k.startswith(str(end))
                   for k in default_options
                   for end in custom_prefix_endings)
    return default_options


# TODO: Add note about how we 'freeze' options at instantiation
class OptionsManager:
    """Class that helps with managing setting PETSc options.

    The recommended way to use the ``OptionsManager`` is by using the
    :func:`attach_options`, :func:`set_from_options`, and
    :func:`inserted_options` free functions. These functions ensure that
    each ``OptionsManager`` is associated to a single PETSc object.

    For detail on the previous approach of using ``OptionsManager``
    as a mixin class (where the user takes responsibility for ensuring
    an association with a single PETSc object), see below.

    To use the ``OptionsManager``:

    1. Pass a PETSc object a parameters dictionary, and optionally
       an options prefix, to :func:`attach_options`. This will create
       an ``OptionsManager`` and set the prefix of the PETSc object,
       but will not yet set it up.
    2. Once the object is ready, pass it to :func:`set_from_options`,
       which will insert the solver options into ``PETSc.Options``
       and call ``obj.setFromOptions``.
    3. The :func:`inserted_options` context manager must be used when
       calling methods on the PETSc object within which solver options
       will be read, for example ``solve``.
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

    To access the ``OptionsManager`` for a PETSc object directly
    use the :func:`get_options` function:

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
    Do that using the :meth:`OptionsManager.inserted_options` context manager.

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
        The prefix to look up items in the global options database. If `None`
        then the prefix is generated from ``default_prefix``. If no trailing
        underscore is provided, one is appended. Hence ``foo_`` and ``foo``
        are treated equivalently. As an exception, if the prefix is the empty
        string, no underscore is appended.
    default_prefix
        The base string to generate default prefixes. If options_prefix
        is not provided then a prefix is automatically generated with the
        form "{default_prefix}_{n}", where n is a unique integer. Note that
        because the unique integer is not stable the auto-generated prefix for
        a particular solver may change between Python invocations due to
        changes elsewhere in the code.
    default_options_set
        The prefix set for any default shared with other solvers.
        See :class:`DefaultOptionSet` for more information.
    appmngr
        The :class:`AppContextManager` containing user python data.

    See Also
    --------
    attach_options
    has_options
    get_options
    set_from_options
    is_set_from_options
    inserted_options
    DefaultOptionSet
    AppContext
    AppContextManager
    """

    count = 0

    def __init__(self, parameters: dict,
                 options_prefix: str | None = None,
                 default_prefix: str | None = None,
                 default_options_set: DefaultOptionSet | None = None,
                 appmngr: AppContextManager | None = None):
        if parameters is None:
            parameters = {}
        else:
            # Convert nested dicts
            parameters = flatten_parameters(parameters)

        # If no prefix is provided generate a default prefix
        if options_prefix is None:
            default_prefix = default_prefix or "petsctools_"
            default_prefix = _validate_prefix(default_prefix)
            options_prefix = f"{default_prefix}{self.count}_"
            self.count += 1
            unsafe_prefix = True
        else:
            options_prefix = _validate_prefix(options_prefix)
            unsafe_prefix = False

        # Are we part of a solver set sharing defaults?
        if default_options_set:
            if options_prefix not in default_options_set.custom_prefixes:
                raise ValueError(
                    f"The options_prefix {options_prefix} must be one"
                    f" of the custom_prefixes of the DefaultOptionSet"
                    f" {default_options_set.custom_prefixes}")
            default_options = get_default_options(
                default_options_set, self.options_object)
        else:
            default_options = {}

        # The parameters to drop from the global options when we leave the
        # inserted_options context. This is everything except for options
        # passed on the command line.
        to_delete = set(parameters.keys())

        # Start building parameters from the defaults so
        # that they will overwritten by any other source.
        parameters = default_options | parameters
        unsafe_options = []
        for full_key, v in self.options_object.getAll().items():
            if full_key.startswith(options_prefix):
                if unsafe_prefix:
                    unsafe_options.append(full_key)

                key = full_key[len(options_prefix):]
                parameters[key] = v

                if key in to_delete:
                    # option is set globally, don't drop when we exit the
                    # context manager
                    to_delete.remove(key)
        if unsafe_options:
            unsafe_options_str = "\n".join((
                f"  {opt}" for opt in unsafe_options
            ))
            petsctools.log.warning(f"""\
Setting options using an autogenerated prefix '{options_prefix}' is unsafe:
{unsafe_options_str}"""
            )

        self.parameters = parameters
        self.to_delete = to_delete
        self.options_prefix = options_prefix

        self._setfromoptions = False

        # user data
        self.appmngr = appmngr

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

        Before calling ``petsc_obj.setFromOptions``, the options from
        this OptionsManager's ``parameters`` are inserted into the global
        :class:`PETSc.Options`, and if this OptionsManager has an ``appmngr``
        then all entries are inserted into the :class:`AppContext`.

        Parameters
        ----------
        petsc_obj
            The PETSc object to call setFromOptions on.

        Raises
        ------
        PetscToolsWarning
            If this method has already been called.

        """
        # Matt says: "Only ever call setFromOptions once".  This
        # function ensures we do so.
        if not self._setfromoptions:
            # Call setfromoptions inserting appropriate options
            # and user data into the global databases.
            with self.inserted_options():
                petsc_obj.setOptionsPrefix(self.options_prefix)
                petsc_obj.setFromOptions()
                self._setfromoptions = True
        else:
            warnings.warn(
                "setFromOptions has already been called", PetscToolsWarning
            )

    @contextlib.contextmanager
    def inserted_options(self):
        """Context manager inside which the petsc options database
        contains the parameters from this object.
        If this OptionsManager has an ``appmngr`` then all entries
        are inserted into the :class:`AppContext`.
        """
        try:
            for k, v in self.parameters.items():
                self.options_object[self.options_prefix + k] = v
            if self.appmngr:
                with self.appmngr.inserted_appctx():
                    yield
            else:
                yield
        finally:
            for k in self.parameters:
                if self.options_object.used(self.options_prefix + k):
                    self._used_options.add(k)
            for k in self.to_delete:
                del self.options_object[self.options_prefix + k]

    @functools.cached_property
    def options_object(self):
        from petsc4py import PETSc

        # We can't pass the prefix here because that doesn't DTRT
        # for flag options
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
    default_prefix: str | None = None,
    default_options_set: DefaultOptionSet | None = None,
    appmngr: AppContextManager | None = None,
) -> None:
    """Set up an :class:`OptionsManager` and attach it to a PETSc Object.

    Parameters
    ----------
    obj
        The object to attach an :class:`OptionsManager` to.
    parameters
        The dictionary of parameters to use.
    options_prefix
        The options prefix to use for this object.
    default_prefix
        Base string for autogenerated default prefixes.
    default_options_set
        The prefix set for any default shared with other solvers.
    appmngr
        The :class:`AppContextManager` containing user python data.

    See Also
    --------
    OptionsManager
    set_from_options
    DefaultOptionSet
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
        default_options_set=default_options_set,
        appmngr=appmngr,
    )
    obj.setAttr("options", options)


def has_options(obj: petsc4py.PETSc.Object) -> bool:
    """Return whether this PETSc object has an :class:`OptionsManager`
    attached.

    Parameters
    ----------
    obj
        The object which may have an :class:`OptionsManager`.

    Returns
    -------
        Whether the object has an :class:`OptionsManager`.

    See Also
    --------
    OptionsManager
    attach_options
    set_from_options
    """
    return "options" in obj.getDict() and isinstance(
        obj.getAttr("options"), OptionsManager
    )


def get_options(obj: petsc4py.PETSc.Object) -> OptionsManager:
    """Return the :class:`OptionsManager` attached to this PETSc object.

    Parameters
    ----------
    obj
        The object to get the :class:`OptionsManager` from.

    Returns
    -------
        The :class:`OptionsManager` attached to the object.

    Raises
    ------
    PetscToolsException
        If the object does not have an :class:`OptionsManager`.

    See Also
    --------
    OptionsManager
    attach_options
    set_from_options
    """
    if not has_options(obj):
        raise PetscToolsException(
            "No OptionsManager attached to {petscobj2str(obj)}"
        )
    return obj.getAttr("options")


def set_default_parameter(
    obj: petsc4py.PETSc.Object, key: str, val: Any
) -> None:
    """Set a default parameter value in the :class:`OptionsManager` of a
    PETSc object.

    Parameters
    ----------
    obj
        The object to get the :class:`OptionsManager` from.
    key
        The options parameter name
    val
        The options parameter value

    Raises
    ------
    PetscToolsException
        If the object does not have an :class:`OptionsManager`.

    See Also
    --------
    OptionsManager
    OptionsManager.set_default_parameter
    attach_options
    set_from_options
    """
    get_options(obj).set_default_parameter(key, val)


def set_from_options(
    obj: petsc4py.PETSc.Object,
    parameters: dict | None = None,
    options_prefix: str | None = None,
    default_prefix: str | None = None,
    default_options_set: DefaultOptionSet | None = None,
    appmngr: AppContextManager | None = None,
) -> None:
    """Set up a PETSc object from the options in its :class:`OptionsManager`.

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
    default_options_set
        The prefix set for any default shared with other solvers.
    appmngr
        An application context for passing non-native python types.

    Raises
    ------
    PetscToolsException
        If the neither ``parameters`` nor ``options_prefix`` are
        provided but ``obj`` does not have an :class:`OptionsManager` attached.
    PetscToolsException
        If the either ``parameters`` or ``options_prefix`` are provided
        but ``obj`` already has an :class:`OptionsManager` attached.
    PetscToolsWarning
        If set_from_options has already been called for this object.

    See Also
    --------
    OptionsManager
    OptionsManager.set_from_options
    attach_options
    DefaultOptionSet
    AppContextManager
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
            default_options_set=default_options_set,
            appmngr=appmngr,
        )

    if is_set_from_options(obj):
        warnings.warn(
            f"setFromOptions has already been called for {petscobj2str(obj)}",
            PetscToolsWarning,
        )

    get_options(obj).set_from_options(obj)


def is_set_from_options(obj: petsc4py.PETSc.Object) -> bool:
    """
    Return whether this PETSc object has been set by the
    :class:`OptionsManager`.

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
        If the object does not have an :class:`OptionsManager`.

    See Also
    --------
    OptionsManager
    attach_options
    set_from_options
    """
    return get_options(obj)._setfromoptions


@contextlib.contextmanager
def inserted_options(
    obj: petsc4py.PETSc.Object | None = None,
    *,
    parameters: dict | None = None,
    options_prefix: str | None = None,
    default_prefix: str | None = None,
    default_options_set: DefaultOptionSet | None = None,
):
    """Context manager inside which the PETSc options database
    contains the parameters either from obj's :class:`OptionsManager`,
    or from the parameters dictionary with the given prefix.
    If the OptionsManager has an ``appmngr`` then all entries are
    inserted into the :class:`AppContext`.

    Parameters
    ----------
    obj :
        The object which may have been set from options.
    parameters
        The dictionary of parameters to use.
        Ignored if obj is passed.
    options_prefix
        The options prefix to use for the parameters.
        Ignored if obj is passed.
    default_prefix
        Base string for autogenerated default prefixes.
        Ignored if obj is passed.
    default_options_set
        The prefix set for any default shared with other solvers.
        Ignored if obj is passed.

    Raises
    ------
    PetscToolsException
        If the object does not have an :class:`OptionsManager`.

    See Also
    --------
    OptionsManager
    OptionsManager.inserted_options
    attach_options
    set_from_options
    """
    if obj:
        opts = get_options(obj)
    else:
        opts = OptionsManager(
            parameters=parameters,
            options_prefix=options_prefix,
            default_prefix=default_prefix,
            default_options_set=default_options_set,
        )

    with opts.inserted_options():
        yield

from __future__ import annotations

import collections
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

import petsctools.exceptions

if TYPE_CHECKING:
    from petsc4py import PETSc


_CONTEXT_KEY = "_petsctools_dmctx"
"""Key used to store the context on the DM."""


class DMContext:
    """TODO"""

    def __init__(self):
        self._attr_stacks = collections.defaultdict(list)

    def __getitem__(self, key: Hashable, /) -> Any:
        stack = self._attr_stacks[key]
        try:
            return self._attr_stack[key][-1]
        except IndexError:
            raise KeyError(f"Attribute '{key}' not found")

    def get(self, key: Hashable, /, default: Any) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def push(self, key: Hashable, value: Any, /) -> None:
        self._attr_stacks[key].append(value)

    def pop(self, key: Hashable, /) -> Any:
        try:
            return self._attr_stacks[key].pop(-1)
        except IndexError:
            raise KeyError(f"No value for '{key}' available to pop")

    def add_hook(
        self,
        key: Hashable,
        /,
        setup: Callable[[], None],
        teardown: Callable[[], None],
    ) -> Any:
        ...


def has_dmctx(dm) -> bool:
    return _CONTEXT_KEY in dm.getDict()


def attach_dmctx(dm: PETSc.DM, ctx: DMContext) -> None:
    if has_dmctx(dm):
        raise petsctools.exceptions.PetscToolsException(
            "DM already has a DMContext attached"
        )
    dm.setAttr(_CONTEXT_KEY, ctx)


def get_dmctx(dm: PETSc.DM) -> DMContext:
    if not has_dmctx(dm):
        raise petsctools.exceptions.PetscToolsException(
            "DM does not have a DMContext attached to it"
        )
    return dm.getAttr(_CONTEXT_KEY)


def 


# in setup
dmctx = petsctools.get_dmctx(pc.dm)
dmctx.push("function_space", V, setup, teardown, other hooks?)
dmctx.add_hook(setup, teardown)

dmctx.push_setup_hook()
dmctx.push_teardown_hook()

dmctx.push()
dmctx.add_teardown_hook()

# but the hooks mostly look like pushing and popping attrs... is the setup more involved? yep somethings I think, look at pmg.py

with petsctools.dm(dm):
    # call setup and teardown hooks
    ...


# in the pc
dmctx = petsctools.get_dmctx(pc.dm)

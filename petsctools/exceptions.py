class PetscToolsException(Exception):
    """Generic base class for petsctools exceptions."""


class PetscToolsNotInitialisedException(PetscToolsException):
    """Exception raised when petsctools should have been initialised."""


class PetscToolsAppctxException(PetscToolsException):
    """Exception raised when the Appctx is missing an entry."""


class PetscToolsWarning(UserWarning):
    """Generic base class for petsctools warnings."""

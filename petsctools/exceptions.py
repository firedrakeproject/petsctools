class PetscToolsException(Exception):
    """Generic base class for petsctools exceptions."""


class PetscToolsNotInitialisedException(PetscToolsException):
    """Exception raised when petsctools should have been initialised."""


class PetscToolsWarning(UserWarning):
    """Generic base class for petsctools warnings."""

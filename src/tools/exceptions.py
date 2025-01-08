# src/tools/exceptions.py
class ModificationError(Exception):
    """Raised when code modification fails."""
    pass

class ValidationError(Exception):
    """Raised when code validation fails."""
    pass

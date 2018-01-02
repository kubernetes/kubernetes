class Error(Exception):
    pass

class KeyError(Error):
    """Raised when key constraints are violated."""
    pass

class OperationalError(Error):
    """Raised when a database operation fails, e.g., because of a timeout.
    May be raised by all Database operations including __init__"""
    pass

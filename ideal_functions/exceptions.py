"""Application-specific exceptions used throughout the project."""


class IdealFunctionError(Exception):
    """Base exception for errors raised by the application."""


class DataSchemaError(IdealFunctionError):
    """Raised when input data does not match the expected structure."""


class DataConsistencyError(IdealFunctionError):
    """Raised when data is valid but cannot be used for the requested operation."""


class DatabaseOperationError(IdealFunctionError):
    """Raised when a database operation cannot be completed."""
class DatasetValidationError(ValueError):
    """Base class for errors that make a dataset unsafe to process."""


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the configured source file does not exist."""


class DatasetLoadError(DatasetValidationError):
    """Raised when a CSV cannot be parsed with the requested options."""


class EmptyDatasetError(DatasetValidationError):
    """Raised when a source CSV has no data records."""


class MissingColumnsError(DatasetValidationError):
    """Raised when one or more required columns are absent."""


class InvalidDataTypeError(DatasetValidationError):
    """Raised when a column does not have the type required by the contract."""


class MissingValuesError(DatasetValidationError):
    """Raised when mandatory values are missing."""


class DuplicateIdentifierError(DatasetValidationError):
    """Raised when more than one record has the same company identifier."""


class InvalidTargetError(DatasetValidationError):
    """Raised when a target value is outside its declared domain."""


class FinancialConsistencyError(DatasetValidationError):
    """Raised when basic accounting relationships do not hold."""
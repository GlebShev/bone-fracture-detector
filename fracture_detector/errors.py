class DetectorError(Exception):
    """Base domain error."""


class InvalidImageError(DetectorError):
    """Uploaded payload is not a supported image."""


class ModelNotFoundError(DetectorError):
    """Requested model identifier is unknown."""


class ModelUnavailableError(DetectorError):
    """Requested model exists in configuration but its weights cannot be loaded."""

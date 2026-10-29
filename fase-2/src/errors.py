"""Excepciones propias para comunicar errores previsibles al usuario."""


class AppError(Exception):
    """Error base de la aplicación."""


class DataValidationError(AppError):
    """Los datos de entrada no cumplen el esquema o el contenido esperado."""


class ModelNotFoundError(AppError):
    """El artefacto del modelo no existe o no corresponde a una versión válida."""


class ModelLoadError(AppError):
    """El artefacto existe pero no pudo cargarse."""


class VersionAlreadyExistsError(AppError):
    """Se intentó crear una versión con un identificador que ya existe."""

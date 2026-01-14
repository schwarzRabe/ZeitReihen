"""
Custom exceptions for the forecast bot.
"""


class ForecastError(Exception):
    """Base exception for all forecasting errors."""
    pass


class DataLoadError(ForecastError):
    """Error loading data from external source."""
    pass


class InsufficientDataError(ForecastError):
    """Not enough historical data for forecasting."""
    pass


class ModelTrainingError(ForecastError):
    """Error during model training."""
    pass


class InvalidTickerError(ForecastError):
    """Invalid ticker symbol."""
    pass

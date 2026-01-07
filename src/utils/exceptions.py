"""
Custom exceptions for the Workforce Planning application.
Provides structured error handling with user-friendly messages.
"""
from typing import Optional, Dict, Any


class WorkforcePlanningError(Exception):
    """Base exception for all application errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = None,
        details: Dict[str, Any] = None,
        user_message: str = None
    ):
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        self.user_message = user_message or message
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "user_message": self.user_message,
            "details": self.details
        }


# ===========================================
# Authentication Errors
# ===========================================

class AuthenticationError(WorkforcePlanningError):
    """Authentication related errors."""
    pass


class InvalidCredentialsError(AuthenticationError):
    """Invalid username or password."""
    
    def __init__(self, username: str = None):
        super().__init__(
            message=f"Invalid credentials for user: {username}" if username else "Invalid credentials",
            error_code="AUTH_INVALID_CREDENTIALS",
            user_message="Invalid username or password. Please try again.",
            details={"username": username} if username else {}
        )


class AccountLockedError(AuthenticationError):
    """Account is locked due to too many failed attempts."""
    
    def __init__(self, username: str, unlock_time: str = None):
        super().__init__(
            message=f"Account locked for user: {username}",
            error_code="AUTH_ACCOUNT_LOCKED",
            user_message=f"Account is temporarily locked. Please try again {unlock_time or 'later'}.",
            details={"username": username, "unlock_time": unlock_time}
        )


class SessionExpiredError(AuthenticationError):
    """Session has expired."""
    
    def __init__(self):
        super().__init__(
            message="Session expired",
            error_code="AUTH_SESSION_EXPIRED",
            user_message="Your session has expired. Please log in again."
        )


class UnauthorizedError(AuthenticationError):
    """User is not authorized for this action."""
    
    def __init__(self, action: str = None, required_role: str = None):
        super().__init__(
            message=f"Unauthorized access to: {action}",
            error_code="AUTH_UNAUTHORIZED",
            user_message="You don't have permission to perform this action.",
            details={"action": action, "required_role": required_role}
        )


# ===========================================
# Data Errors
# ===========================================

class DataError(WorkforcePlanningError):
    """Data related errors."""
    pass


class DataLoadError(DataError):
    """Error loading data from file."""
    
    def __init__(self, filepath: str, reason: str = None):
        super().__init__(
            message=f"Failed to load data from: {filepath}",
            error_code="DATA_LOAD_ERROR",
            user_message=f"Could not load the data file. {reason or 'Please check the file format.'}",
            details={"filepath": filepath, "reason": reason}
        )


class DataValidationError(DataError):
    """Data validation failed."""
    
    def __init__(self, field: str = None, reason: str = None, value: Any = None):
        super().__init__(
            message=f"Data validation failed for field: {field}",
            error_code="DATA_VALIDATION_ERROR",
            user_message=f"Invalid data: {reason or 'Please check your input.'}",
            details={"field": field, "reason": reason, "value": str(value) if value else None}
        )


class MissingDataError(DataError):
    """Required data is missing."""
    
    def __init__(self, data_type: str, required_columns: list = None):
        super().__init__(
            message=f"Missing required data: {data_type}",
            error_code="DATA_MISSING",
            user_message=f"Required data is missing: {data_type}. Please upload the necessary files.",
            details={"data_type": data_type, "required_columns": required_columns}
        )


class DataFormatError(DataError):
    """Data format is incorrect."""
    
    def __init__(self, expected_format: str, actual_format: str = None):
        super().__init__(
            message=f"Invalid data format. Expected: {expected_format}",
            error_code="DATA_FORMAT_ERROR",
            user_message=f"The data format is incorrect. Expected {expected_format} format.",
            details={"expected_format": expected_format, "actual_format": actual_format}
        )


# ===========================================
# Model Errors
# ===========================================

class ModelError(WorkforcePlanningError):
    """Model related errors."""
    pass


class ModelNotTrainedError(ModelError):
    """Model has not been trained yet."""
    
    def __init__(self):
        super().__init__(
            message="Model not trained",
            error_code="MODEL_NOT_TRAINED",
            user_message="The model has not been trained yet. Please train the model first."
        )


class ModelTrainingError(ModelError):
    """Error during model training."""
    
    def __init__(self, reason: str = None):
        super().__init__(
            message=f"Model training failed: {reason}",
            error_code="MODEL_TRAINING_ERROR",
            user_message=f"Model training failed. {reason or 'Please check the data and try again.'}",
            details={"reason": reason}
        )


class ModelLoadError(ModelError):
    """Error loading saved model."""
    
    def __init__(self, model_path: str = None, reason: str = None):
        super().__init__(
            message=f"Failed to load model from: {model_path}",
            error_code="MODEL_LOAD_ERROR",
            user_message="Could not load the saved model. Please retrain the model.",
            details={"model_path": model_path, "reason": reason}
        )


class ForecastError(ModelError):
    """Error generating forecast."""
    
    def __init__(self, reason: str = None):
        super().__init__(
            message=f"Forecast generation failed: {reason}",
            error_code="FORECAST_ERROR",
            user_message=f"Could not generate forecast. {reason or 'Please try again.'}",
            details={"reason": reason}
        )


# ===========================================
# Configuration Errors
# ===========================================

class ConfigurationError(WorkforcePlanningError):
    """Configuration related errors."""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Invalid configuration value."""
    
    def __init__(self, config_key: str, value: Any = None, reason: str = None):
        super().__init__(
            message=f"Invalid configuration: {config_key}",
            error_code="CONFIG_INVALID",
            user_message=f"Invalid configuration value for {config_key}. {reason or ''}",
            details={"config_key": config_key, "value": str(value), "reason": reason}
        )


class MissingConfigurationError(ConfigurationError):
    """Required configuration is missing."""
    
    def __init__(self, config_key: str):
        super().__init__(
            message=f"Missing required configuration: {config_key}",
            error_code="CONFIG_MISSING",
            user_message=f"Required configuration '{config_key}' is not set.",
            details={"config_key": config_key}
        )


# ===========================================
# Export Errors
# ===========================================

class ExportError(WorkforcePlanningError):
    """Export related errors."""
    pass


class ExportFormatError(ExportError):
    """Unsupported export format."""
    
    def __init__(self, format: str, supported_formats: list = None):
        super().__init__(
            message=f"Unsupported export format: {format}",
            error_code="EXPORT_FORMAT_ERROR",
            user_message=f"Export format '{format}' is not supported.",
            details={"format": format, "supported_formats": supported_formats or ["xlsx", "csv"]}
        )


"""
Environment-based settings management.
Loads configuration from environment variables and .env file.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
from functools import lru_cache
import logging

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, use system env vars only


def get_env(key: str, default: str = None, cast: type = str) -> any:
    """Get environment variable with type casting."""
    value = os.getenv(key, default)
    if value is None:
        return None
    
    if cast == bool:
        return value.lower() in ("true", "1", "yes", "on")
    elif cast == int:
        return int(value)
    elif cast == float:
        return float(value)
    elif cast == list:
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


@dataclass
class AppSettings:
    """Application settings loaded from environment."""
    
    # Application
    app_name: str = "Workforce Planning ML"
    app_env: str = "development"
    debug: bool = True
    secret_key: str = "dev-secret-key-change-in-production"
    
    # Authentication
    session_timeout_minutes: int = 60
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_raw_path: Path = None
    data_processed_path: Path = None
    models_path: Path = None
    outputs_path: Path = None
    logs_path: Path = None
    
    # Model settings
    default_forecast_horizon_days: int = 30
    default_test_size_days: int = 14
    model_auto_save: bool = True
    
    # Capacity planning
    default_service_level: float = 0.80
    default_service_time_seconds: int = 20
    default_shrinkage_factor: float = 0.30
    working_hours_start: int = 8
    working_hours_end: int = 20
    
    # Holidays
    holiday_country: str = "DE"
    holiday_state: str = "BY"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: str = "logs/app.log"
    
    def __post_init__(self):
        """Initialize paths after dataclass creation."""
        if self.data_raw_path is None:
            self.data_raw_path = self.project_root / "data" / "raw"
        if self.data_processed_path is None:
            self.data_processed_path = self.project_root / "data" / "processed"
        if self.models_path is None:
            self.models_path = self.project_root / "models"
        if self.outputs_path is None:
            self.outputs_path = self.project_root / "outputs"
        if self.logs_path is None:
            self.logs_path = self.project_root / "logs"
        
        # Ensure directories exist
        for path in [self.data_raw_path, self.data_processed_path, 
                     self.models_path, self.outputs_path, self.logs_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "AppSettings":
        """Create settings from environment variables."""
        project_root = Path(__file__).parent.parent.parent
        
        return cls(
            app_name=get_env("APP_NAME", "Workforce Planning ML"),
            app_env=get_env("APP_ENV", "development"),
            debug=get_env("DEBUG", "true", bool),
            secret_key=get_env("SECRET_KEY", "dev-secret-key-change-in-production"),
            
            session_timeout_minutes=get_env("SESSION_TIMEOUT_MINUTES", "60", int),
            max_login_attempts=get_env("MAX_LOGIN_ATTEMPTS", "5", int),
            lockout_duration_minutes=get_env("LOCKOUT_DURATION_MINUTES", "15", int),
            
            project_root=project_root,
            data_raw_path=project_root / get_env("DATA_RAW_PATH", "data/raw"),
            data_processed_path=project_root / get_env("DATA_PROCESSED_PATH", "data/processed"),
            models_path=project_root / get_env("MODELS_PATH", "models"),
            outputs_path=project_root / get_env("OUTPUTS_PATH", "outputs"),
            logs_path=project_root / get_env("LOGS_PATH", "logs"),
            
            default_forecast_horizon_days=get_env("DEFAULT_FORECAST_HORIZON_DAYS", "30", int),
            default_test_size_days=get_env("DEFAULT_TEST_SIZE_DAYS", "14", int),
            model_auto_save=get_env("MODEL_AUTO_SAVE", "true", bool),
            
            default_service_level=get_env("DEFAULT_SERVICE_LEVEL", "0.80", float),
            default_service_time_seconds=get_env("DEFAULT_SERVICE_TIME_SECONDS", "20", int),
            default_shrinkage_factor=get_env("DEFAULT_SHRINKAGE_FACTOR", "0.30", float),
            working_hours_start=get_env("WORKING_HOURS_START", "8", int),
            working_hours_end=get_env("WORKING_HOURS_END", "20", int),
            
            holiday_country=get_env("HOLIDAY_COUNTRY", "DE"),
            holiday_state=get_env("HOLIDAY_STATE", "BY"),
            
            log_level=get_env("LOG_LEVEL", "INFO"),
            log_format=get_env("LOG_FORMAT", "json"),
            log_file=get_env("LOG_FILE", "logs/app.log"),
        )
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.app_env.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.app_env.lower() == "development"


@lru_cache()
def get_settings() -> AppSettings:
    """Get cached application settings."""
    return AppSettings.from_env()


# Convenience instance
settings = get_settings()


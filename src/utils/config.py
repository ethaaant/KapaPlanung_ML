"""
Configuration and constants for the Workforce Planning ML system.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
for directory in [DATA_RAW, DATA_PROCESSED, OUTPUTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class TaskConfig:
    """Configuration for a specific task type."""
    name: str
    avg_handling_time_minutes: float  # Average handling time in minutes
    concurrency: float = 1.0  # How many tasks an agent can handle simultaneously


@dataclass
class CapacityConfig:
    """Configuration for capacity planning."""
    # Service level targets
    service_level_target: float = 0.80  # 80% of calls answered
    service_level_time_seconds: int = 20  # within 20 seconds
    
    # Shrinkage factor (breaks, meetings, absenteeism, etc.)
    shrinkage_factor: float = 0.30  # 30% shrinkage
    
    # Working hours
    working_hours_start: int = 8  # 8 AM
    working_hours_end: int = 20  # 8 PM
    
    # Task configurations
    tasks: Dict[str, TaskConfig] = field(default_factory=lambda: {
        "calls": TaskConfig(name="Inbound Calls", avg_handling_time_minutes=5.0),
        "emails": TaskConfig(name="E-Mails", avg_handling_time_minutes=8.0, concurrency=2.0),
        "outbound_ook": TaskConfig(name="Outbound OOK", avg_handling_time_minutes=6.0),
        "outbound_omk": TaskConfig(name="Outbound OMK", avg_handling_time_minutes=6.0),
        "outbound_nb": TaskConfig(name="Outbound NB", avg_handling_time_minutes=4.0),
    })


@dataclass
class ForecastConfig:
    """Configuration for forecasting models."""
    # Forecast horizon
    forecast_horizon_days: int = 30  # 1 month ahead
    forecast_granularity: str = "hourly"
    
    # Training configuration
    test_size_days: int = 14  # Use last 2 weeks for validation
    
    # Feature engineering
    lag_hours: List[int] = field(default_factory=lambda: [1, 2, 3, 24, 48, 168])  # 168 = 1 week
    rolling_windows: List[int] = field(default_factory=lambda: [24, 48, 168])  # hours
    
    # Model parameters (LightGBM)
    lgbm_params: Dict = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
    })


# Default configurations
DEFAULT_CAPACITY_CONFIG = CapacityConfig()
DEFAULT_FORECAST_CONFIG = ForecastConfig()

# German holidays country code
HOLIDAY_COUNTRY = "DE"
HOLIDAY_STATE = "BY"  # Bavaria - adjust as needed


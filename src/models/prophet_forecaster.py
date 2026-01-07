"""
Prophet-based forecasting for workforce planning.
Handles complex seasonality (daily, weekly, yearly) with built-in holiday support.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import joblib
from pathlib import Path
from datetime import datetime

# Try to import Prophet
try:
    from prophet import Prophet
    from prophet.serialize import model_to_json, model_from_json
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None

import holidays

from src.utils.config import MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProphetForecastResult:
    """Container for Prophet forecast results."""
    predictions: pd.DataFrame
    timestamps: pd.Series
    metrics: Dict[str, float]
    components: Optional[pd.DataFrame] = None  # Trend, seasonality breakdown
    confidence_intervals: Optional[Dict[str, pd.DataFrame]] = None


@dataclass
class ProphetConfig:
    """Configuration for Prophet model."""
    # Seasonality
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = True
    
    # Seasonality mode
    seasonality_mode: str = "multiplicative"  # or "additive"
    
    # Trend
    growth: str = "linear"  # or "logistic" for saturation
    changepoint_prior_scale: float = 0.05  # Flexibility of trend
    
    # Holidays
    country_holidays: str = "DE"  # German holidays
    
    # Custom seasonalities
    add_monthly_seasonality: bool = True
    
    # Uncertainty
    uncertainty_samples: int = 1000
    interval_width: float = 0.95  # 95% confidence interval
    
    # Training
    mcmc_samples: int = 0  # 0 = MAP estimation (faster), >0 = full Bayesian


class ProphetForecaster:
    """
    Prophet-based multi-target forecaster for customer service workload.
    
    Advantages over gradient boosting:
    - Native handling of multiple seasonalities
    - Built-in holiday effects
    - Automatic trend detection
    - Robust confidence intervals
    - Interpretable components
    """
    
    def __init__(self, config: ProphetConfig = None):
        """
        Initialize Prophet forecaster.
        
        Args:
            config: Prophet configuration
        """
        if not PROPHET_AVAILABLE:
            raise ImportError(
                "Prophet is required. Install with: pip install prophet"
            )
        
        self.config = config or ProphetConfig()
        self.models: Dict[str, Prophet] = {}
        self.target_columns: List[str] = []
        self.is_fitted = False
        self.training_metrics: Dict[str, Dict[str, float]] = {}
        
        # German holidays
        self.german_holidays = self._create_german_holidays()
        
        # Custom events (can be extended)
        self.custom_events: Optional[pd.DataFrame] = None
    
    def _create_german_holidays(self, years: List[int] = None) -> pd.DataFrame:
        """Create German holiday dataframe for Prophet."""
        if years is None:
            years = list(range(2020, 2030))
        
        # Get German holidays
        de_holidays = holidays.Germany(years=years, prov='BY')  # Bavaria
        
        # Convert to Prophet format
        holiday_df = pd.DataFrame([
            {"ds": date, "holiday": name}
            for date, name in de_holidays.items()
        ])
        
        # Add special retail events
        special_events = []
        for year in years:
            # Black Friday (4th Friday of November)
            nov_1 = datetime(year, 11, 1)
            days_until_friday = (4 - nov_1.weekday()) % 7
            fourth_friday = nov_1 + pd.Timedelta(days=days_until_friday + 21)
            
            special_events.extend([
                {"ds": fourth_friday, "holiday": "Black Friday"},
                {"ds": fourth_friday + pd.Timedelta(days=1), "holiday": "Black Saturday"},
                {"ds": fourth_friday + pd.Timedelta(days=2), "holiday": "Cyber Monday Weekend"},
                {"ds": fourth_friday + pd.Timedelta(days=3), "holiday": "Cyber Monday"},
            ])
            
            # Pre-Christmas peak (Dec 15-23)
            for day in range(15, 24):
                special_events.append({
                    "ds": datetime(year, 12, day),
                    "holiday": "Pre-Christmas Peak"
                })
        
        special_df = pd.DataFrame(special_events)
        holiday_df = pd.concat([holiday_df, special_df], ignore_index=True)
        
        return holiday_df
    
    def add_custom_events(self, events: pd.DataFrame):
        """
        Add custom events (e.g., newsletters, campaigns).
        
        Args:
            events: DataFrame with columns ['ds', 'event_name'] or ['date', 'event_name']
        """
        if 'date' in events.columns:
            events = events.rename(columns={'date': 'ds'})
        
        events = events.rename(columns={'event_name': 'holiday'})
        events['ds'] = pd.to_datetime(events['ds'])
        
        self.custom_events = events
    
    def _create_model(self, target: str) -> Prophet:
        """Create a new Prophet model with configured settings."""
        model = Prophet(
            growth=self.config.growth,
            yearly_seasonality=self.config.yearly_seasonality,
            weekly_seasonality=self.config.weekly_seasonality,
            daily_seasonality=self.config.daily_seasonality,
            seasonality_mode=self.config.seasonality_mode,
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            uncertainty_samples=self.config.uncertainty_samples,
            interval_width=self.config.interval_width,
            mcmc_samples=self.config.mcmc_samples
        )
        
        # Add holidays
        if self.german_holidays is not None:
            model.add_country_holidays(country_name='DE')
        
        # Add custom events as holidays
        if self.custom_events is not None:
            for event_name in self.custom_events['holiday'].unique():
                model.add_regressor(f'event_{event_name}', mode='additive')
        
        # Add monthly seasonality if configured
        if self.config.add_monthly_seasonality:
            model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5
            )
        
        # Add hourly patterns (for sub-daily data)
        model.add_seasonality(
            name='hourly_weekday',
            period=1,
            fourier_order=12,
            condition_name='is_weekday'
        )
        
        model.add_seasonality(
            name='hourly_weekend',
            period=1,
            fourier_order=8,
            condition_name='is_weekend'
        )
        
        return model
    
    def _prepare_data(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
        """Prepare data for Prophet (requires 'ds' and 'y' columns)."""
        df = pd.DataFrame()
        df['ds'] = pd.to_datetime(data['timestamp'])
        df['y'] = data[target].values
        
        # Add weekday/weekend indicator for conditional seasonality
        df['is_weekday'] = df['ds'].dt.dayofweek < 5
        df['is_weekend'] = ~df['is_weekday']
        
        return df
    
    def fit(
        self,
        data: pd.DataFrame,
        target_columns: List[str] = None,
        verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Fit Prophet models for each target column.
        
        Args:
            data: DataFrame with 'timestamp' and target columns
            target_columns: List of columns to forecast
            verbose: Print progress
            
        Returns:
            Dictionary of metrics for each target
        """
        if target_columns is None:
            # Auto-detect numeric columns
            target_columns = [c for c in data.columns 
                            if c not in ['timestamp', 'date', 'hour']
                            and pd.api.types.is_numeric_dtype(data[c])]
        
        self.target_columns = target_columns
        all_metrics = {}
        
        for target in target_columns:
            if verbose:
                logger.info(f"\n🔮 Training Prophet model for: {target}")
            
            # Prepare data
            train_df = self._prepare_data(data, target)
            
            # Remove any rows with NaN
            train_df = train_df.dropna()
            
            # Create and fit model
            model = self._create_model(target)
            
            # Suppress Prophet's verbose output
            import logging as log
            log.getLogger('prophet').setLevel(log.WARNING)
            log.getLogger('cmdstanpy').setLevel(log.WARNING)
            
            model.fit(train_df)
            
            self.models[target] = model
            
            # Cross-validation for metrics (on last 30 days)
            metrics = self._calculate_cv_metrics(model, train_df, target)
            all_metrics[target] = metrics
            self.training_metrics[target] = metrics
            
            if verbose:
                logger.info(f"  RMSE: {metrics['rmse']:.2f}")
                logger.info(f"  MAE: {metrics['mae']:.2f}")
                logger.info(f"  MAPE: {metrics['mape']:.1f}%")
        
        self.is_fitted = True
        return all_metrics
    
    def _calculate_cv_metrics(
        self,
        model: Prophet,
        train_df: pd.DataFrame,
        target: str,
        horizon_days: int = 14
    ) -> Dict[str, float]:
        """Calculate metrics using holdout validation."""
        # Use last N days as validation
        cutoff = train_df['ds'].max() - pd.Timedelta(days=horizon_days)
        train = train_df[train_df['ds'] <= cutoff].copy()
        valid = train_df[train_df['ds'] > cutoff].copy()
        
        if len(valid) < 24:  # Need at least 1 day
            # Not enough validation data, use in-sample metrics
            forecast = model.predict(train_df)
            y_true = train_df['y'].values
            y_pred = forecast['yhat'].values
        else:
            # Fit on training, predict on validation
            temp_model = self._create_model(target)
            
            import logging as log
            log.getLogger('prophet').setLevel(log.WARNING)
            
            temp_model.fit(train)
            
            future = valid[['ds', 'is_weekday', 'is_weekend']].copy()
            forecast = temp_model.predict(future)
            
            y_true = valid['y'].values
            y_pred = forecast['yhat'].values
        
        # Calculate metrics
        y_pred = np.maximum(0, y_pred)  # Ensure non-negative
        
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        
        # MAPE (avoiding division by zero)
        mask = y_true > 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0.0
        
        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
    
    def forecast(
        self,
        horizon_hours: int,
        start_date: datetime = None,
        include_history: bool = False
    ) -> ProphetForecastResult:
        """
        Generate forecast for future periods.
        
        Args:
            horizon_hours: Number of hours to forecast
            start_date: Start date for forecast (default: end of training data)
            include_history: Include historical fitted values
            
        Returns:
            ProphetForecastResult with predictions and components
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = {}
        confidence_intervals = {}
        components_list = []
        
        for target in self.target_columns:
            model = self.models[target]
            
            # Create future dataframe
            future = model.make_future_dataframe(
                periods=horizon_hours,
                freq='H',
                include_history=include_history
            )
            
            # Filter to start from start_date if provided
            if start_date is not None:
                future = future[future['ds'] >= pd.Timestamp(start_date)]
            
            # Add conditional seasonality columns
            future['is_weekday'] = future['ds'].dt.dayofweek < 5
            future['is_weekend'] = ~future['is_weekday']
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Ensure non-negative predictions
            forecast['yhat'] = np.maximum(0, forecast['yhat'])
            forecast['yhat_lower'] = np.maximum(0, forecast['yhat_lower'])
            forecast['yhat_upper'] = np.maximum(0, forecast['yhat_upper'])
            
            predictions[target] = forecast['yhat'].values
            
            # Store confidence intervals
            confidence_intervals[target] = pd.DataFrame({
                'lower': forecast['yhat_lower'].values,
                'upper': forecast['yhat_upper'].values,
                'prediction': forecast['yhat'].values
            })
            
            # Store components for first target
            if len(components_list) == 0:
                components_list.append(forecast[['ds', 'trend', 'weekly', 'yearly']])
        
        # Create predictions DataFrame
        pred_df = pd.DataFrame(predictions)
        timestamps = pd.Series(future['ds'].values)
        
        # Components
        components = components_list[0] if components_list else None
        
        return ProphetForecastResult(
            predictions=pred_df,
            timestamps=timestamps,
            metrics=self.training_metrics,
            components=components,
            confidence_intervals=confidence_intervals
        )
    
    def get_seasonality_components(self, target: str = None) -> pd.DataFrame:
        """
        Get seasonality components for visualization.
        
        Returns:
            DataFrame with trend, weekly, yearly, daily patterns
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        target = target or self.target_columns[0]
        model = self.models[target]
        
        # Create a typical year of hourly data for component visualization
        future = model.make_future_dataframe(periods=24*365, freq='H')
        future['is_weekday'] = future['ds'].dt.dayofweek < 5
        future['is_weekend'] = ~future['is_weekday']
        
        forecast = model.predict(future)
        
        return forecast[['ds', 'trend', 'weekly', 'yearly', 'yhat']]
    
    def plot_components(self, target: str = None):
        """Plot Prophet components (trend, seasonality)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        target = target or self.target_columns[0]
        model = self.models[target]
        
        # This returns a matplotlib figure
        future = model.make_future_dataframe(periods=24*30, freq='H')
        future['is_weekday'] = future['ds'].dt.dayofweek < 5
        future['is_weekend'] = ~future['is_weekday']
        forecast = model.predict(future)
        
        return model.plot_components(forecast)
    
    def save(self, filepath: Path = None):
        """Save trained models to disk."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Nothing to save.")
        
        filepath = filepath or MODELS_DIR / "prophet_forecaster.joblib"
        
        # Serialize Prophet models
        serialized_models = {}
        for target, model in self.models.items():
            serialized_models[target] = model_to_json(model)
        
        save_data = {
            "models_json": serialized_models,
            "target_columns": self.target_columns,
            "config": self.config,
            "training_metrics": self.training_metrics,
            "german_holidays": self.german_holidays
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Saved Prophet models to: {filepath}")
    
    @classmethod
    def load(cls, filepath: Path = None) -> "ProphetForecaster":
        """Load trained models from disk."""
        filepath = filepath or MODELS_DIR / "prophet_forecaster.joblib"
        
        save_data = joblib.load(filepath)
        
        forecaster = cls(config=save_data["config"])
        forecaster.target_columns = save_data["target_columns"]
        forecaster.training_metrics = save_data["training_metrics"]
        forecaster.german_holidays = save_data["german_holidays"]
        
        # Deserialize Prophet models
        for target, model_json in save_data["models_json"].items():
            forecaster.models[target] = model_from_json(model_json)
        
        forecaster.is_fitted = True
        
        logger.info(f"Loaded Prophet models from: {filepath}")
        return forecaster


def create_prophet_forecaster(config: ProphetConfig = None) -> ProphetForecaster:
    """Factory function to create a Prophet forecaster."""
    return ProphetForecaster(config)


if __name__ == "__main__":
    # Test with sample data
    from src.data.loader import DataLoader, create_sample_data
    
    # Create sample data
    create_sample_data()
    
    # Load data
    loader = DataLoader()
    combined = loader.combine_data()
    
    print(f"Data shape: {combined.shape}")
    print(f"Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
    
    # Train Prophet
    forecaster = ProphetForecaster()
    metrics = forecaster.fit(
        combined,
        target_columns=['calls', 'emails', 'outbound_total']
    )
    
    print("\nTraining Metrics:")
    for target, m in metrics.items():
        print(f"  {target}:")
        for name, value in m.items():
            print(f"    {name}: {value:.3f}")
    
    # Generate forecast
    result = forecaster.forecast(horizon_hours=24*7)
    print(f"\nForecast shape: {result.predictions.shape}")
    print(f"Forecast dates: {result.timestamps.min()} to {result.timestamps.max()}")
    
    # Save model
    forecaster.save()
    
    # Test loading
    loaded = ProphetForecaster.load()
    print(f"\nLoaded model with {len(loaded.target_columns)} targets")


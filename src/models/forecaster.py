"""
Time series forecasting models for workload prediction.
Uses sklearn's HistGradientBoostingRegressor as the primary model.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import joblib
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.config import DEFAULT_FORECAST_CONFIG, ForecastConfig, MODELS_DIR
from src.data.preprocessor import Preprocessor, FeatureSet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Container for forecast results."""
    predictions: pd.DataFrame
    timestamps: pd.Series
    metrics: Dict[str, float]
    feature_importance: Optional[pd.DataFrame] = None
    confidence_intervals: Optional[Dict[str, pd.DataFrame]] = None  # {target: DataFrame with lower, upper}


class WorkloadForecaster:
    """
    Multi-target forecasting model for customer service workload.
    
    Trains separate HistGradientBoosting models for each target (calls, emails, outbound).
    Supports multi-step forecasting using recursive prediction.
    Includes confidence interval estimation.
    """
    
    def __init__(self, config: ForecastConfig = None):
        """
        Initialize forecaster.
        
        Args:
            config: Forecast configuration with model parameters.
        """
        self.config = config or DEFAULT_FORECAST_CONFIG
        self.models: Dict[str, HistGradientBoostingRegressor] = {}
        self.preprocessor: Optional[Preprocessor] = None
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        self.is_fitted = False
        
        # Store training residuals for confidence intervals
        self._training_residuals: Dict[str, np.ndarray] = {}
        self._training_metrics: Dict[str, Dict[str, float]] = {}
        
    def _create_model(self) -> HistGradientBoostingRegressor:
        """Create a new HistGradientBoostingRegressor model."""
        model = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.05,
            max_depth=8,
            max_leaf_nodes=31,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            random_state=42
        )
        return model
    
    def fit(
        self,
        feature_set: FeatureSet,
        test_size_days: int = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Fit models for all target columns.
        
        Args:
            feature_set: FeatureSet from preprocessor.
            test_size_days: Days to use for validation.
            verbose: Print training progress.
            
        Returns:
            Dictionary of metrics for each target.
        """
        test_size_days = test_size_days or self.config.test_size_days
        
        self.feature_columns = feature_set.feature_columns
        self.target_columns = feature_set.target_columns
        
        df = feature_set.features.copy()
        
        # Remove rows with NaN in features
        df = self._handle_missing(df)
        
        # Train/test split
        test_samples = test_size_days * 24
        split_idx = len(df) - test_samples
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train = train_df[self.feature_columns]
        X_test = test_df[self.feature_columns]
        
        all_metrics = {}
        
        for target in self.target_columns:
            if verbose:
                logger.info(f"\nTraining model for: {target}")
            
            y_train = train_df[target]
            y_test = test_df[target]
            
            # Create and train model
            model = self._create_model()
            
            # Fit model (early stopping is built-in via validation_fraction)
            model.fit(X_train, y_train)
            
            self.models[target] = model
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_pred = np.maximum(0, y_pred)  # Ensure non-negative
            
            metrics = self._calculate_metrics(y_test, y_pred)
            all_metrics[target] = metrics
            
            # Store residuals for confidence intervals
            self._training_residuals[target] = (np.array(y_test) - y_pred)
            self._training_metrics[target] = metrics
            
            if verbose:
                logger.info(f"  RMSE: {metrics['rmse']:.2f}")
                logger.info(f"  MAE: {metrics['mae']:.2f}")
                logger.info(f"  R²: {metrics['r2']:.3f}")
                logger.info(f"  MAPE: {metrics['mape']:.1f}%")
        
        self.is_fitted = True
        return all_metrics
    
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        df = df.copy()
        
        # Forward fill then backward fill for time series continuity
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method="ffill").fillna(method="bfill")
        
        # Fill remaining NaN with 0
        df = df.fillna(0)
        
        return df
    
    def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate forecast metrics."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Avoid division by zero in MAPE
        mask = y_true > 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0.0
        
        return {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "mape": mape
        }
    
    def predict(
        self,
        data: pd.DataFrame,
        preprocessor: Preprocessor = None
    ) -> ForecastResult:
        """
        Make predictions on prepared data.
        
        Args:
            data: DataFrame with features already prepared.
            preprocessor: Optional preprocessor to transform data.
            
        Returns:
            ForecastResult with predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        df = data.copy()
        
        if preprocessor:
            df = preprocessor.transform(df)
        
        df = self._handle_missing(df)
        X = df[self.feature_columns]
        
        predictions = {}
        for target in self.target_columns:
            pred = self.models[target].predict(X)
            predictions[target] = np.maximum(0, pred)  # Ensure non-negative
        
        pred_df = pd.DataFrame(predictions)
        
        return ForecastResult(
            predictions=pred_df,
            timestamps=df["timestamp"],
            metrics={},
            feature_importance=self.get_feature_importance()
        )
    
    def forecast_horizon(
        self,
        last_known_data: pd.DataFrame,
        horizon_hours: int = None,
        preprocessor: Preprocessor = None,
        start_date: pd.Timestamp = None,
        confidence_level: float = 0.95
    ) -> ForecastResult:
        """
        Generate forecasts for future time periods.
        
        Uses recursive forecasting: predicts one step ahead, 
        then uses that prediction as input for the next step.
        
        Args:
            last_known_data: Historical data including recent observations.
            horizon_hours: Number of hours to forecast ahead.
            preprocessor: Preprocessor for feature engineering.
            start_date: Optional start date for forecast (defaults to after last data).
            confidence_level: Confidence level for intervals (default 0.95 = 95%).
            
        Returns:
            ForecastResult with future predictions and confidence intervals.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        horizon_hours = horizon_hours or self.config.forecast_horizon_days * 24
        
        df = last_known_data.copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Determine start timestamp
        if start_date is not None:
            # Ensure it's a proper timestamp
            start_ts = pd.Timestamp(start_date)
            # Start from the beginning of that day
            start_ts = start_ts.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            # Default: start after last known data
            start_ts = df["timestamp"].max() + pd.Timedelta(hours=1)
        
        # Generate future timestamps
        future_timestamps = pd.date_range(
            start=start_ts,
            periods=horizon_hours,
            freq="H"
        )
        
        # Create future dataframe with NaN targets
        future_df = pd.DataFrame({"timestamp": future_timestamps})
        for target in self.target_columns:
            future_df[target] = np.nan
        
        # Combine with historical data for feature engineering
        combined = pd.concat([df, future_df], ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        
        # Recursive forecasting
        predictions = {target: [] for target in self.target_columns}
        
        for i in range(horizon_hours):
            # Get current position in combined dataframe
            current_idx = len(df) + i
            
            # Create features using all available data up to current point
            if preprocessor is None:
                preprocessor = Preprocessor(self.config)
            
            # Process data up to current point
            temp_df = combined.iloc[:current_idx + 1].copy()
            feature_set = preprocessor.fit_transform(
                temp_df, 
                target_columns=self.target_columns
            )
            
            # Get features for current timestamp
            processed = self._handle_missing(feature_set.features)
            X_current = processed[self.feature_columns].iloc[-1:].copy()
            
            # Predict each target
            for target in self.target_columns:
                pred = self.models[target].predict(X_current)[0]
                pred = max(0, pred)  # Ensure non-negative
                predictions[target].append(pred)
                
                # Update combined dataframe with prediction for next iteration
                combined.loc[current_idx, target] = pred
            
            if (i + 1) % 24 == 0:
                logger.info(f"Forecast progress: {i + 1}/{horizon_hours} hours")
        
        # Create result DataFrame
        pred_df = pd.DataFrame(predictions)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            predictions=predictions,
            horizon_hours=horizon_hours,
            confidence_level=confidence_level
        )
        
        return ForecastResult(
            predictions=pred_df,
            timestamps=pd.Series(future_timestamps),
            metrics={},
            feature_importance=self.get_feature_importance(),
            confidence_intervals=confidence_intervals
        )
    
    def _calculate_confidence_intervals(
        self,
        predictions: Dict[str, List[float]],
        horizon_hours: int,
        confidence_level: float = 0.95
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate confidence intervals for predictions.
        
        Uses residual-based approach: estimates prediction uncertainty
        based on training residuals, with uncertainty growing over time.
        
        Args:
            predictions: Dictionary of predictions per target
            horizon_hours: Forecast horizon
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary mapping target to DataFrame with 'lower' and 'upper' columns
        """
        from scipy import stats
        
        confidence_intervals = {}
        
        # Z-score for confidence level
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        for target in self.target_columns:
            pred_values = np.array(predictions[target])
            
            # Get residual standard deviation from training
            if target in self._training_residuals and len(self._training_residuals[target]) > 0:
                residual_std = np.std(self._training_residuals[target])
            else:
                # Fallback: estimate from training metrics
                if target in self._training_metrics:
                    residual_std = self._training_metrics[target].get('rmse', 0)
                else:
                    # Last resort: use 10% of mean prediction
                    residual_std = np.mean(pred_values) * 0.1
            
            # Uncertainty grows with forecast horizon (using sqrt scaling)
            # This reflects the increasing uncertainty in recursive forecasting
            hours = np.arange(1, horizon_hours + 1)
            uncertainty_factor = np.sqrt(hours / 24)  # Normalize by day
            
            # Calculate interval width
            interval_width = z_score * residual_std * (1 + uncertainty_factor * 0.5)
            
            # Calculate bounds
            lower = np.maximum(0, pred_values - interval_width)  # Floor at 0
            upper = pred_values + interval_width
            
            confidence_intervals[target] = pd.DataFrame({
                'lower': lower,
                'upper': upper,
                'prediction': pred_values
            })
        
        return confidence_intervals
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance across all models.
        
        Returns:
            DataFrame with feature importance for each target.
        """
        if not self.is_fitted:
            return pd.DataFrame()
        
        importance_data = {"feature": self.feature_columns}
        
        for target, model in self.models.items():
            # HistGradientBoostingRegressor may not have feature_importances_ in all versions
            if hasattr(model, 'feature_importances_'):
                importance_data[f"{target}_importance"] = model.feature_importances_
            else:
                # Use zeros as placeholder if not available
                importance_data[f"{target}_importance"] = np.zeros(len(self.feature_columns))
        
        importance_df = pd.DataFrame(importance_data)
        
        # Add average importance
        importance_cols = [c for c in importance_df.columns if c.endswith("_importance")]
        importance_df["avg_importance"] = importance_df[importance_cols].mean(axis=1)
        
        return importance_df.sort_values("avg_importance", ascending=False)
    
    def save(self, filepath: Path = None):
        """
        Save trained models to disk.
        
        Args:
            filepath: Path to save models. Defaults to models directory.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Nothing to save.")
        
        filepath = filepath or MODELS_DIR / "workload_forecaster.joblib"
        
        save_data = {
            "models": self.models,
            "feature_columns": self.feature_columns,
            "target_columns": self.target_columns,
            "config": self.config
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Saved models to: {filepath}")
    
    @classmethod
    def load(cls, filepath: Path = None) -> "WorkloadForecaster":
        """
        Load trained models from disk.
        
        Args:
            filepath: Path to load models from.
            
        Returns:
            Loaded WorkloadForecaster instance.
        """
        filepath = filepath or MODELS_DIR / "workload_forecaster.joblib"
        
        save_data = joblib.load(filepath)
        
        forecaster = cls(config=save_data["config"])
        forecaster.models = save_data["models"]
        forecaster.feature_columns = save_data["feature_columns"]
        forecaster.target_columns = save_data["target_columns"]
        forecaster.is_fitted = True
        
        logger.info(f"Loaded models from: {filepath}")
        return forecaster


class QuickForecaster:
    """
    Simplified forecaster for quick predictions without recursive steps.
    Faster but less accurate for long horizons.
    """
    
    def __init__(self, config: ForecastConfig = None):
        self.config = config or DEFAULT_FORECAST_CONFIG
        self.forecaster = WorkloadForecaster(config)
        self.preprocessor = Preprocessor(config)
        
    def fit(self, data: pd.DataFrame, target_columns: List[str] = None) -> Dict:
        """
        Fit the model on historical data.
        
        Args:
            data: Historical data with timestamp and value columns.
            target_columns: Columns to forecast.
            
        Returns:
            Training metrics.
        """
        feature_set = self.preprocessor.fit_transform(data, target_columns)
        return self.forecaster.fit(feature_set)
    
    def predict_simple(
        self, 
        horizon_days: int = 30
    ) -> pd.DataFrame:
        """
        Generate forecasts using a simpler approach.
        
        Instead of recursive prediction, uses average patterns
        from historical data.
        
        Args:
            horizon_days: Days to forecast ahead.
            
        Returns:
            DataFrame with predictions.
        """
        # This is a placeholder for a simpler forecasting method
        # Can be enhanced with pattern-based prediction
        raise NotImplementedError("Use forecast_horizon for accurate predictions")


def evaluate_model(
    forecaster: WorkloadForecaster,
    test_data: pd.DataFrame,
    preprocessor: Preprocessor
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on test data.
    
    Args:
        forecaster: Trained forecaster.
        test_data: Test data with actual values.
        preprocessor: Fitted preprocessor.
        
    Returns:
        Metrics for each target.
    """
    feature_set = preprocessor.fit_transform(
        test_data, 
        target_columns=forecaster.target_columns
    )
    
    df = feature_set.features.copy()
    df = forecaster._handle_missing(df)
    
    X = df[forecaster.feature_columns]
    
    all_metrics = {}
    for target in forecaster.target_columns:
        y_true = df[target]
        y_pred = forecaster.models[target].predict(X)
        y_pred = np.maximum(0, y_pred)
        
        all_metrics[target] = forecaster._calculate_metrics(y_true, y_pred)
    
    return all_metrics


if __name__ == "__main__":
    from src.data.loader import DataLoader, create_sample_data
    from src.data.preprocessor import Preprocessor
    
    # Create sample data
    create_sample_data()
    
    # Load and preprocess
    loader = DataLoader()
    data = loader.load_all()
    combined = loader.combine_data()
    
    print(f"Data shape: {combined.shape}")
    
    # Preprocess
    preprocessor = Preprocessor()
    feature_set = preprocessor.fit_transform(combined)
    
    print(f"Features: {len(feature_set.feature_columns)}")
    print(f"Targets: {feature_set.target_columns}")
    
    # Train model
    forecaster = WorkloadForecaster()
    metrics = forecaster.fit(feature_set)
    
    print("\nTraining Metrics:")
    for target, m in metrics.items():
        print(f"  {target}:")
        for name, value in m.items():
            print(f"    {name}: {value:.3f}")
    
    # Feature importance
    importance = forecaster.get_feature_importance()
    print("\nTop 10 Important Features:")
    print(importance.head(10))
    
    # Save model
    forecaster.save()
    
    # Test loading
    loaded = WorkloadForecaster.load()
    print(f"\nLoaded model with {len(loaded.target_columns)} targets")


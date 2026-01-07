"""
Forecast Tracking Module.
Tracks actual vs forecast values and calculates accuracy metrics.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json


@dataclass
class ForecastRecord:
    """A single forecast record."""
    forecast_id: str
    created_at: datetime
    model_version: str
    forecast_start: datetime
    forecast_end: datetime
    predictions: Dict[str, List[float]]  # {column: [values]}
    timestamps: List[datetime]
    created_by: str = ""
    notes: str = ""


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for forecast evaluation."""
    mae: float  # Mean Absolute Error
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Square Error
    mse: float   # Mean Square Error
    r2: float    # R-squared
    
    # Additional metrics
    bias: float  # Average (predicted - actual)
    accuracy_pct: float  # 100 - MAPE
    
    # Breakdowns
    hourly_mape: Dict[int, float] = field(default_factory=dict)
    daily_mape: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "mae": self.mae,
            "mape": self.mape,
            "rmse": self.rmse,
            "mse": self.mse,
            "r2": self.r2,
            "bias": self.bias,
            "accuracy_pct": self.accuracy_pct,
            "hourly_mape": self.hourly_mape,
            "daily_mape": self.daily_mape
        }


class ForecastTracker:
    """
    Tracks and compares forecasts with actual values.
    """
    
    def __init__(self, storage_dir: str = "outputs/forecast_tracking"):
        """
        Initialize the forecast tracker.
        
        Args:
            storage_dir: Directory to store forecast records
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.forecasts_file = self.storage_dir / "forecasts.json"
        self.actuals_file = self.storage_dir / "actuals.parquet"
        
        self.forecasts: Dict[str, ForecastRecord] = self._load_forecasts()
    
    def _load_forecasts(self) -> Dict[str, ForecastRecord]:
        """Load stored forecasts."""
        if not self.forecasts_file.exists():
            return {}
        
        try:
            with open(self.forecasts_file, 'r') as f:
                data = json.load(f)
            
            forecasts = {}
            for fid, record in data.items():
                forecasts[fid] = ForecastRecord(
                    forecast_id=record["forecast_id"],
                    created_at=datetime.fromisoformat(record["created_at"]),
                    model_version=record["model_version"],
                    forecast_start=datetime.fromisoformat(record["forecast_start"]),
                    forecast_end=datetime.fromisoformat(record["forecast_end"]),
                    predictions=record["predictions"],
                    timestamps=[datetime.fromisoformat(t) for t in record["timestamps"]],
                    created_by=record.get("created_by", ""),
                    notes=record.get("notes", "")
                )
            return forecasts
        except (json.JSONDecodeError, KeyError):
            return {}
    
    def _save_forecasts(self):
        """Save forecasts to disk."""
        data = {}
        for fid, record in self.forecasts.items():
            data[fid] = {
                "forecast_id": record.forecast_id,
                "created_at": record.created_at.isoformat(),
                "model_version": record.model_version,
                "forecast_start": record.forecast_start.isoformat(),
                "forecast_end": record.forecast_end.isoformat(),
                "predictions": record.predictions,
                "timestamps": [t.isoformat() for t in record.timestamps],
                "created_by": record.created_by,
                "notes": record.notes
            }
        
        with open(self.forecasts_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_forecast(
        self,
        forecast_df: pd.DataFrame,
        model_version: str,
        created_by: str = "",
        notes: str = ""
    ) -> str:
        """
        Save a forecast for later comparison.
        
        Args:
            forecast_df: DataFrame with 'timestamp' and prediction columns
            model_version: Version of the model used
            created_by: User who created the forecast
            notes: Optional notes
            
        Returns:
            Forecast ID
        """
        if "timestamp" not in forecast_df.columns:
            raise ValueError("Forecast must have 'timestamp' column")
        
        timestamps = pd.to_datetime(forecast_df["timestamp"]).tolist()
        
        # Get prediction columns
        pred_columns = [col for col in forecast_df.columns 
                       if col not in ["timestamp", "date"]]
        
        predictions = {}
        for col in pred_columns:
            predictions[col] = forecast_df[col].tolist()
        
        # Generate forecast ID
        forecast_id = f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        record = ForecastRecord(
            forecast_id=forecast_id,
            created_at=datetime.now(),
            model_version=model_version,
            forecast_start=min(timestamps),
            forecast_end=max(timestamps),
            predictions=predictions,
            timestamps=timestamps,
            created_by=created_by,
            notes=notes
        )
        
        self.forecasts[forecast_id] = record
        self._save_forecasts()
        
        return forecast_id
    
    def save_actuals(self, actuals_df: pd.DataFrame):
        """
        Save or update actual values.
        
        Args:
            actuals_df: DataFrame with 'timestamp' and actual value columns
        """
        if "timestamp" not in actuals_df.columns:
            raise ValueError("Actuals must have 'timestamp' column")
        
        # Load existing actuals
        existing = self.load_actuals()
        
        if existing is not None and not existing.empty:
            # Merge, keeping new values for duplicates
            actuals_df = actuals_df.set_index("timestamp")
            existing = existing.set_index("timestamp")
            
            combined = existing.combine_first(actuals_df)
            combined = combined.reset_index()
        else:
            combined = actuals_df
        
        # Save
        combined.to_parquet(self.actuals_file, index=False)
    
    def load_actuals(
        self,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Optional[pd.DataFrame]:
        """
        Load actual values.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with actuals or None if not found
        """
        if not self.actuals_file.exists():
            return None
        
        try:
            df = pd.read_parquet(self.actuals_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            if start_date:
                df = df[df["timestamp"] >= start_date]
            if end_date:
                df = df[df["timestamp"] <= end_date]
            
            return df
        except Exception:
            return None
    
    def get_forecast(self, forecast_id: str) -> Optional[ForecastRecord]:
        """Get a specific forecast."""
        return self.forecasts.get(forecast_id)
    
    def list_forecasts(
        self,
        limit: int = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[ForecastRecord]:
        """
        List all forecasts.
        
        Args:
            limit: Optional maximum number to return
            start_date: Filter by forecast start date
            end_date: Filter by forecast end date
            
        Returns:
            List of forecast records
        """
        forecasts = list(self.forecasts.values())
        
        if start_date:
            forecasts = [f for f in forecasts if f.forecast_start >= start_date]
        if end_date:
            forecasts = [f for f in forecasts if f.forecast_end <= end_date]
        
        # Sort by creation date, newest first
        forecasts.sort(key=lambda x: x.created_at, reverse=True)
        
        if limit:
            forecasts = forecasts[:limit]
        
        return forecasts
    
    def compare_forecast_to_actuals(
        self,
        forecast_id: str,
        column: str = None
    ) -> Tuple[pd.DataFrame, Dict[str, AccuracyMetrics]]:
        """
        Compare a forecast to actual values.
        
        Args:
            forecast_id: The forecast to compare
            column: Optional specific column to compare
            
        Returns:
            Tuple of (comparison DataFrame, metrics by column)
        """
        forecast = self.get_forecast(forecast_id)
        if not forecast:
            raise ValueError(f"Forecast '{forecast_id}' not found")
        
        # Load actuals for the forecast period
        actuals = self.load_actuals(
            start_date=forecast.forecast_start,
            end_date=forecast.forecast_end
        )
        
        if actuals is None or actuals.empty:
            raise ValueError("No actual data available for the forecast period")
        
        # Build comparison DataFrame
        forecast_df = pd.DataFrame({
            "timestamp": forecast.timestamps,
            **{f"pred_{k}": v for k, v in forecast.predictions.items()}
        })
        
        # Merge with actuals
        comparison = pd.merge(
            forecast_df,
            actuals,
            on="timestamp",
            how="inner",
            suffixes=("", "_actual")
        )
        
        if comparison.empty:
            raise ValueError("No matching timestamps between forecast and actuals")
        
        # Calculate metrics for each column
        metrics = {}
        columns_to_compare = [column] if column else list(forecast.predictions.keys())
        
        for col in columns_to_compare:
            pred_col = f"pred_{col}"
            actual_col = col
            
            if pred_col not in comparison.columns or actual_col not in comparison.columns:
                continue
            
            predicted = comparison[pred_col].values
            actual = comparison[actual_col].values
            
            # Filter out NaN values
            mask = ~(np.isnan(predicted) | np.isnan(actual))
            predicted = predicted[mask]
            actual = actual[mask]
            
            if len(predicted) == 0:
                continue
            
            metrics[col] = self._calculate_metrics(
                predicted, actual, comparison[mask]["timestamp"]
            )
        
        return comparison, metrics
    
    def _calculate_metrics(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        timestamps: pd.Series = None
    ) -> AccuracyMetrics:
        """Calculate accuracy metrics."""
        
        # Basic metrics
        errors = predicted - actual
        abs_errors = np.abs(errors)
        sq_errors = errors ** 2
        
        mae = np.mean(abs_errors)
        mse = np.mean(sq_errors)
        rmse = np.sqrt(mse)
        bias = np.mean(errors)
        
        # MAPE (handle zeros)
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_errors = np.abs(errors / actual) * 100
            pct_errors = pct_errors[np.isfinite(pct_errors)]
            mape = np.mean(pct_errors) if len(pct_errors) > 0 else 0
        
        # R-squared
        ss_res = np.sum(sq_errors)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Accuracy percentage
        accuracy_pct = max(0, 100 - mape)
        
        # Hourly breakdown
        hourly_mape = {}
        daily_mape = {}
        
        if timestamps is not None:
            ts = pd.to_datetime(timestamps)
            
            # Hourly
            for hour in range(24):
                hour_mask = ts.dt.hour == hour
                if hour_mask.any():
                    hour_actual = actual[hour_mask]
                    hour_pred = predicted[hour_mask]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        hour_pct = np.abs((hour_pred - hour_actual) / hour_actual) * 100
                        hour_pct = hour_pct[np.isfinite(hour_pct)]
                        hourly_mape[hour] = float(np.mean(hour_pct)) if len(hour_pct) > 0 else 0
            
            # Daily
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            for dow, day_name in enumerate(days):
                day_mask = ts.dt.dayofweek == dow
                if day_mask.any():
                    day_actual = actual[day_mask]
                    day_pred = predicted[day_mask]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        day_pct = np.abs((day_pred - day_actual) / day_actual) * 100
                        day_pct = day_pct[np.isfinite(day_pct)]
                        daily_mape[day_name] = float(np.mean(day_pct)) if len(day_pct) > 0 else 0
        
        return AccuracyMetrics(
            mae=float(mae),
            mape=float(mape),
            rmse=float(rmse),
            mse=float(mse),
            r2=float(r2),
            bias=float(bias),
            accuracy_pct=float(accuracy_pct),
            hourly_mape=hourly_mape,
            daily_mape=daily_mape
        )
    
    def get_forecast_accuracy_summary(self, limit: int = 10) -> pd.DataFrame:
        """
        Get accuracy summary for recent forecasts.
        
        Returns:
            DataFrame with forecast accuracy summary
        """
        forecasts = self.list_forecasts(limit=limit)
        
        rows = []
        for forecast in forecasts:
            try:
                _, metrics = self.compare_forecast_to_actuals(forecast.forecast_id)
                
                for col, m in metrics.items():
                    rows.append({
                        "forecast_id": forecast.forecast_id,
                        "created_at": forecast.created_at,
                        "model_version": forecast.model_version,
                        "column": col,
                        "mae": m.mae,
                        "mape": m.mape,
                        "rmse": m.rmse,
                        "accuracy_pct": m.accuracy_pct,
                        "forecast_start": forecast.forecast_start,
                        "forecast_end": forecast.forecast_end
                    })
            except ValueError:
                # No actuals available for this forecast
                continue
        
        if not rows:
            return pd.DataFrame()
        
        return pd.DataFrame(rows)
    
    def delete_forecast(self, forecast_id: str):
        """Delete a forecast."""
        if forecast_id in self.forecasts:
            del self.forecasts[forecast_id]
            self._save_forecasts()


# Global tracker instance
_forecast_tracker = None


def get_forecast_tracker(storage_dir: str = "outputs/forecast_tracking") -> ForecastTracker:
    """Get or create the global forecast tracker."""
    global _forecast_tracker
    if _forecast_tracker is None:
        _forecast_tracker = ForecastTracker(storage_dir)
    return _forecast_tracker


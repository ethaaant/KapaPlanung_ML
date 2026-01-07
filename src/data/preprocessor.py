"""
Data preprocessing and feature engineering for time series forecasting.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import holidays
import logging

from src.utils.config import (
    HOLIDAY_COUNTRY, 
    HOLIDAY_STATE, 
    DEFAULT_FORECAST_CONFIG,
    ForecastConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Container for feature engineering results."""
    features: pd.DataFrame
    target_columns: List[str]
    feature_columns: List[str]
    categorical_columns: List[str]
    timestamp_column: str = "timestamp"


class Preprocessor:
    """
    Preprocess data and engineer features for time series forecasting.
    
    Features include:
    - Time-based features (hour, day, week, month)
    - Lag features (same hour previous days/weeks)
    - Rolling statistics
    - Holiday indicators
    - Cyclical encodings
    """
    
    def __init__(self, config: ForecastConfig = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Forecast configuration with feature parameters.
        """
        self.config = config or DEFAULT_FORECAST_CONFIG
        self.holiday_calendar = holidays.country_holidays(
            HOLIDAY_COUNTRY, 
            subdiv=HOLIDAY_STATE
        )
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        self.categorical_columns: List[str] = []
        
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.
        
        Args:
            df: DataFrame with timestamp column.
            
        Returns:
            DataFrame with added time features.
        """
        df = df.copy()
        ts = df["timestamp"]
        
        # Basic time features
        df["hour"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["day_of_month"] = ts.dt.day
        df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
        df["month"] = ts.dt.month
        df["quarter"] = ts.dt.quarter
        df["year"] = ts.dt.year
        
        # Boolean features
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_month_start"] = ts.dt.is_month_start.astype(int)
        df["is_month_end"] = ts.dt.is_month_end.astype(int)
        
        # Work hours indicator (8am - 8pm)
        df["is_work_hours"] = ((df["hour"] >= 8) & (df["hour"] < 20)).astype(int)
        
        # Peak hours (10am-12pm, 2pm-4pm)
        df["is_peak_hours"] = (
            ((df["hour"] >= 10) & (df["hour"] < 12)) | 
            ((df["hour"] >= 14) & (df["hour"] < 16))
        ).astype(int)
        
        return df
    
    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cyclical encoding for periodic features.
        
        Uses sin/cos encoding to capture cyclical nature of time.
        
        Args:
            df: DataFrame with basic time features.
            
        Returns:
            DataFrame with cyclical features.
        """
        df = df.copy()
        
        # Hour of day (0-23)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        
        # Day of week (0-6)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        
        # Month (1-12)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        
        # Day of month (1-31)
        df["dom_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
        df["dom_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)
        
        return df
    
    def add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add holiday indicators.
        
        Args:
            df: DataFrame with timestamp column.
            
        Returns:
            DataFrame with holiday features.
        """
        df = df.copy()
        
        # Is holiday
        df["is_holiday"] = df["timestamp"].dt.date.apply(
            lambda x: x in self.holiday_calendar
        ).astype(int)
        
        # Days to/from nearest holiday
        dates = df["timestamp"].dt.date.unique()
        min_date = min(dates)
        max_date = max(dates)
        
        # Get all holidays in the range
        holiday_dates = [
            d for d in self.holiday_calendar.keys()
            if min_date <= d <= max_date
        ]
        
        if holiday_dates:
            def days_to_holiday(date):
                if date in self.holiday_calendar:
                    return 0
                future = [h for h in holiday_dates if h > date]
                if future:
                    return (min(future) - date).days
                return 30  # Default if no upcoming holiday
            
            def days_from_holiday(date):
                if date in self.holiday_calendar:
                    return 0
                past = [h for h in holiday_dates if h < date]
                if past:
                    return (date - max(past)).days
                return 30  # Default if no past holiday
            
            df["days_to_holiday"] = df["timestamp"].dt.date.apply(days_to_holiday)
            df["days_from_holiday"] = df["timestamp"].dt.date.apply(days_from_holiday)
        else:
            df["days_to_holiday"] = 30
            df["days_from_holiday"] = 30
        
        # Day before/after holiday
        df["is_day_before_holiday"] = (df["days_to_holiday"] == 1).astype(int)
        df["is_day_after_holiday"] = (df["days_from_holiday"] == 1).astype(int)
        
        return df
    
    def add_lag_features(
        self, 
        df: pd.DataFrame, 
        target_cols: List[str],
        lag_hours: List[int] = None
    ) -> pd.DataFrame:
        """
        Add lag features for target columns.
        
        Args:
            df: DataFrame with target columns.
            target_cols: Columns to create lags for.
            lag_hours: List of lag periods in hours.
            
        Returns:
            DataFrame with lag features.
        """
        df = df.copy()
        lag_hours = lag_hours or self.config.lag_hours
        
        for col in target_cols:
            if col not in df.columns:
                continue
                
            for lag in lag_hours:
                lag_name = f"{col}_lag_{lag}h"
                df[lag_name] = df[col].shift(lag)
        
        return df
    
    def add_rolling_features(
        self, 
        df: pd.DataFrame, 
        target_cols: List[str],
        windows: List[int] = None
    ) -> pd.DataFrame:
        """
        Add rolling statistics for target columns.
        
        Args:
            df: DataFrame with target columns.
            target_cols: Columns to create rolling stats for.
            windows: List of window sizes in hours.
            
        Returns:
            DataFrame with rolling features.
        """
        df = df.copy()
        windows = windows or self.config.rolling_windows
        
        for col in target_cols:
            if col not in df.columns:
                continue
                
            for window in windows:
                # Rolling mean
                df[f"{col}_roll_mean_{window}h"] = df[col].shift(1).rolling(
                    window=window, min_periods=1
                ).mean()
                
                # Rolling std
                df[f"{col}_roll_std_{window}h"] = df[col].shift(1).rolling(
                    window=window, min_periods=1
                ).std()
                
                # Rolling min/max
                df[f"{col}_roll_min_{window}h"] = df[col].shift(1).rolling(
                    window=window, min_periods=1
                ).min()
                
                df[f"{col}_roll_max_{window}h"] = df[col].shift(1).rolling(
                    window=window, min_periods=1
                ).max()
        
        return df
    
    def add_same_time_features(
        self, 
        df: pd.DataFrame, 
        target_cols: List[str]
    ) -> pd.DataFrame:
        """
        Add features for same time in previous periods.
        
        Args:
            df: DataFrame with target columns.
            target_cols: Columns to create features for.
            
        Returns:
            DataFrame with same-time features.
        """
        df = df.copy()
        
        for col in target_cols:
            if col not in df.columns:
                continue
            
            # Same hour yesterday
            df[f"{col}_same_hour_yesterday"] = df[col].shift(24)
            
            # Same hour last week
            df[f"{col}_same_hour_last_week"] = df[col].shift(168)
            
            # Average of same hour over past 4 weeks
            same_hour_cols = [df[col].shift(168 * i) for i in range(1, 5)]
            df[f"{col}_same_hour_4week_avg"] = pd.concat(
                same_hour_cols, axis=1
            ).mean(axis=1)
        
        return df
    
    def add_trend_features(
        self, 
        df: pd.DataFrame, 
        target_cols: List[str]
    ) -> pd.DataFrame:
        """
        Add trend-related features.
        
        Args:
            df: DataFrame with target columns.
            target_cols: Columns to create trend features for.
            
        Returns:
            DataFrame with trend features.
        """
        df = df.copy()
        
        for col in target_cols:
            if col not in df.columns:
                continue
            
            # Hour-over-hour change
            df[f"{col}_hourly_change"] = df[col].diff(1)
            
            # Day-over-day change (same hour)
            df[f"{col}_daily_change"] = df[col].diff(24)
            
            # Week-over-week change (same hour)
            df[f"{col}_weekly_change"] = df[col].diff(168)
            
            # Percentage changes
            df[f"{col}_hourly_pct_change"] = df[col].pct_change(1)
            df[f"{col}_daily_pct_change"] = df[col].pct_change(24)
        
        return df
    
    def fit_transform(
        self, 
        data: pd.DataFrame,
        target_columns: List[str] = None
    ) -> FeatureSet:
        """
        Fit preprocessor and transform data.
        
        Args:
            data: Combined data with timestamp and target columns.
            target_columns: List of target columns to forecast.
            
        Returns:
            FeatureSet with all engineered features.
        """
        df = data.copy()
        
        # Ensure sorted by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Identify target columns if not provided
        if target_columns is None:
            target_columns = [
                col for col in df.columns 
                if col not in ["timestamp"] and df[col].dtype in [np.int64, np.float64]
            ]
        
        self.target_columns = target_columns
        logger.info(f"Target columns: {target_columns}")
        
        # Add all feature types
        logger.info("Adding time features...")
        df = self.add_time_features(df)
        
        logger.info("Adding cyclical features...")
        df = self.add_cyclical_features(df)
        
        logger.info("Adding holiday features...")
        df = self.add_holiday_features(df)
        
        logger.info("Adding lag features...")
        df = self.add_lag_features(df, target_columns)
        
        logger.info("Adding rolling features...")
        df = self.add_rolling_features(df, target_columns)
        
        logger.info("Adding same-time features...")
        df = self.add_same_time_features(df, target_columns)
        
        logger.info("Adding trend features...")
        df = self.add_trend_features(df, target_columns)
        
        # Identify feature columns (exclude timestamp and targets)
        exclude_cols = ["timestamp"] + target_columns
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # Identify categorical columns
        self.categorical_columns = [
            "hour", "day_of_week", "day_of_month", "week_of_year", 
            "month", "quarter", "year"
        ]
        
        # Fill infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"Created {len(self.feature_columns)} features")
        
        return FeatureSet(
            features=df,
            target_columns=self.target_columns,
            feature_columns=self.feature_columns,
            categorical_columns=self.categorical_columns
        )
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            data: New data with same structure as training data.
            
        Returns:
            Transformed DataFrame.
        """
        df = data.copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        df = self.add_time_features(df)
        df = self.add_cyclical_features(df)
        df = self.add_holiday_features(df)
        df = self.add_lag_features(df, self.target_columns)
        df = self.add_rolling_features(df, self.target_columns)
        df = self.add_same_time_features(df, self.target_columns)
        df = self.add_trend_features(df, self.target_columns)
        
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df


def prepare_train_test_split(
    feature_set: FeatureSet,
    test_days: int = 14,
    min_train_samples: int = 168  # At least 1 week
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets chronologically.
    
    Args:
        feature_set: FeatureSet from preprocessor.
        test_days: Number of days to use for testing.
        min_train_samples: Minimum training samples required.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    df = feature_set.features.copy()
    
    # Remove rows with NaN in lag features (beginning of series)
    # Find the minimum number of valid rows needed
    lag_cols = [c for c in df.columns if "_lag_" in c or "_roll_" in c or "_same_hour_" in c]
    df_valid = df.dropna(subset=lag_cols)
    
    if len(df_valid) < min_train_samples + test_days * 24:
        logger.warning(
            f"Not enough data after removing NaN. "
            f"Available: {len(df_valid)}, Required: {min_train_samples + test_days * 24}"
        )
        # Use less strict NaN handling
        df_valid = df.copy()
        for col in lag_cols:
            df_valid[col] = df_valid[col].fillna(df_valid[col].median())
    
    # Split chronologically
    test_samples = test_days * 24
    split_idx = len(df_valid) - test_samples
    
    train_df = df_valid.iloc[:split_idx]
    test_df = df_valid.iloc[split_idx:]
    
    X_train = train_df[feature_set.feature_columns]
    X_test = test_df[feature_set.feature_columns]
    
    y_train = train_df[feature_set.target_columns]
    y_test = test_df[feature_set.target_columns]
    
    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from src.data.loader import DataLoader, create_sample_data
    
    # Create sample data if needed
    create_sample_data()
    
    # Load data
    loader = DataLoader()
    data = loader.load_all()
    combined = loader.combine_data()
    
    print(f"Combined data shape: {combined.shape}")
    print(f"Columns: {combined.columns.tolist()}")
    
    # Preprocess
    preprocessor = Preprocessor()
    feature_set = preprocessor.fit_transform(combined)
    
    print(f"\nFeature set shape: {feature_set.features.shape}")
    print(f"Target columns: {feature_set.target_columns}")
    print(f"Number of features: {len(feature_set.feature_columns)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = prepare_train_test_split(feature_set)
    
    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")


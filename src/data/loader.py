"""
Data loader for CSV and Excel files.
Handles loading and validating historical data for calls, emails, and outbound tasks.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Union
from dataclasses import dataclass
import logging

from src.utils.config import DATA_RAW, DATA_PROCESSED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataSchema:
    """Schema definition for data validation."""
    required_columns: List[str]
    timestamp_column: str
    value_columns: List[str]
    optional_columns: List[str] = None
    
    def __post_init__(self):
        if self.optional_columns is None:
            self.optional_columns = []


# Define schemas for different data types
SCHEMAS = {
    "calls": DataSchema(
        required_columns=["timestamp"],
        timestamp_column="timestamp",
        value_columns=["call_volume"],
        optional_columns=["call_volume", "duration", "queue"]
    ),
    "emails": DataSchema(
        required_columns=["timestamp"],
        timestamp_column="timestamp",
        value_columns=["email_count"],
        optional_columns=["email_count", "type"]
    ),
    "outbound": DataSchema(
        required_columns=["timestamp"],
        timestamp_column="timestamp",
        value_columns=["count"],
        optional_columns=["count", "type"]
    ),
}


class DataLoader:
    """
    Load and validate data from CSV and Excel files.
    
    Supports flexible data formats:
    - Aggregated data (hourly volumes)
    - Individual records (will be aggregated automatically)
    """
    
    def __init__(self, data_dir: Path = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing raw data files. Defaults to DATA_RAW.
        """
        self.data_dir = data_dir or DATA_RAW
        self.loaded_data: Dict[str, pd.DataFrame] = {}
        
    def _read_file(self, filepath: Path) -> pd.DataFrame:
        """
        Read a CSV or Excel file.
        
        Args:
            filepath: Path to the file.
            
        Returns:
            DataFrame with file contents.
        """
        suffix = filepath.suffix.lower()
        
        if suffix == ".csv":
            df = pd.read_csv(filepath)
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        # Normalize column names (lowercase, strip whitespace)
        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
        
        return df
    
    def _parse_timestamp(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """
        Parse timestamp column to datetime.
        
        Args:
            df: DataFrame with timestamp column.
            timestamp_col: Name of timestamp column.
            
        Returns:
            DataFrame with parsed timestamp.
        """
        df = df.copy()
        
        # Try different date formats
        date_formats = [
            None,  # Let pandas infer
            "%Y-%m-%d %H:%M:%S",
            "%d.%m.%Y %H:%M:%S",
            "%d.%m.%Y %H:%M",
            "%Y-%m-%d %H:%M",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
        ]
        
        for fmt in date_formats:
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=fmt)
                logger.info(f"Parsed timestamp with format: {fmt or 'inferred'}")
                break
            except (ValueError, TypeError):
                continue
        else:
            # Last resort - let pandas try its best
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], infer_datetime_format=True)
        
        return df
    
    def _aggregate_to_hourly(
        self, 
        df: pd.DataFrame, 
        timestamp_col: str,
        value_col: str = None,
        agg_func: str = "sum"
    ) -> pd.DataFrame:
        """
        Aggregate data to hourly granularity.
        
        Args:
            df: DataFrame with timestamp.
            timestamp_col: Name of timestamp column.
            value_col: Column to aggregate (if None, counts records).
            agg_func: Aggregation function ('sum', 'mean', 'count').
            
        Returns:
            Hourly aggregated DataFrame.
        """
        df = df.copy()
        df["hour"] = df[timestamp_col].dt.floor("H")
        
        if value_col and value_col in df.columns:
            hourly = df.groupby("hour").agg({value_col: agg_func}).reset_index()
        else:
            # Count records per hour
            hourly = df.groupby("hour").size().reset_index(name="count")
        
        hourly = hourly.rename(columns={"hour": "timestamp"})
        return hourly
    
    def _validate_data(self, df: pd.DataFrame, schema: DataSchema) -> bool:
        """
        Validate data against schema.
        
        Args:
            df: DataFrame to validate.
            schema: Expected schema.
            
        Returns:
            True if valid, raises ValueError otherwise.
        """
        # Check for required columns
        missing = [col for col in schema.required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Check timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[schema.timestamp_column]):
            raise ValueError(f"Column '{schema.timestamp_column}' must be datetime")
        
        # Check for null timestamps
        null_count = df[schema.timestamp_column].isnull().sum()
        if null_count > 0:
            logger.warning(f"Found {null_count} null timestamps, will be dropped")
        
        return True
    
    def load_calls(
        self, 
        filepath: Union[str, Path] = None,
        filename_pattern: str = "*call*"
    ) -> pd.DataFrame:
        """
        Load call data.
        
        Args:
            filepath: Specific file path, or None to search in data_dir.
            filename_pattern: Glob pattern to find call files.
            
        Returns:
            Hourly aggregated call volume DataFrame.
        """
        if filepath:
            files = [Path(filepath)]
        else:
            files = list(self.data_dir.glob(f"{filename_pattern}.csv")) + \
                    list(self.data_dir.glob(f"{filename_pattern}.xlsx"))
        
        if not files:
            logger.warning("No call data files found")
            return pd.DataFrame()
        
        dfs = []
        for f in files:
            logger.info(f"Loading calls from: {f}")
            df = self._read_file(f)
            
            # Find timestamp column
            ts_col = self._find_column(df, ["timestamp", "date", "datetime", "time", "datum"])
            if ts_col:
                df = df.rename(columns={ts_col: "timestamp"})
            
            df = self._parse_timestamp(df, "timestamp")
            
            # Find value column or aggregate records
            value_col = self._find_column(df, ["call_volume", "calls", "volume", "count", "anzahl"])
            if value_col:
                df = df.rename(columns={value_col: "call_volume"})
                df = self._aggregate_to_hourly(df, "timestamp", "call_volume")
            else:
                df = self._aggregate_to_hourly(df, "timestamp")
                df = df.rename(columns={"count": "call_volume"})
            
            dfs.append(df)
        
        result = pd.concat(dfs, ignore_index=True)
        result = result.groupby("timestamp").agg({"call_volume": "sum"}).reset_index()
        result = result.sort_values("timestamp").reset_index(drop=True)
        
        self.loaded_data["calls"] = result
        logger.info(f"Loaded {len(result)} hourly call records")
        
        return result
    
    def load_emails(
        self, 
        filepath: Union[str, Path] = None,
        filename_pattern: str = "*email*"
    ) -> pd.DataFrame:
        """
        Load email data.
        
        Args:
            filepath: Specific file path, or None to search in data_dir.
            filename_pattern: Glob pattern to find email files.
            
        Returns:
            Hourly aggregated email volume DataFrame.
        """
        if filepath:
            files = [Path(filepath)]
        else:
            files = list(self.data_dir.glob(f"{filename_pattern}.csv")) + \
                    list(self.data_dir.glob(f"{filename_pattern}.xlsx"))
        
        if not files:
            logger.warning("No email data files found")
            return pd.DataFrame()
        
        dfs = []
        for f in files:
            logger.info(f"Loading emails from: {f}")
            df = self._read_file(f)
            
            # Find timestamp column
            ts_col = self._find_column(df, ["timestamp", "date", "datetime", "time", "datum"])
            if ts_col:
                df = df.rename(columns={ts_col: "timestamp"})
            
            df = self._parse_timestamp(df, "timestamp")
            
            # Find value column or aggregate records
            value_col = self._find_column(df, ["email_count", "emails", "count", "anzahl", "volume"])
            if value_col:
                df = df.rename(columns={value_col: "email_count"})
                df = self._aggregate_to_hourly(df, "timestamp", "email_count")
            else:
                df = self._aggregate_to_hourly(df, "timestamp")
                df = df.rename(columns={"count": "email_count"})
            
            dfs.append(df)
        
        result = pd.concat(dfs, ignore_index=True)
        result = result.groupby("timestamp").agg({"email_count": "sum"}).reset_index()
        result = result.sort_values("timestamp").reset_index(drop=True)
        
        self.loaded_data["emails"] = result
        logger.info(f"Loaded {len(result)} hourly email records")
        
        return result
    
    def load_outbound(
        self, 
        filepath: Union[str, Path] = None,
        filename_pattern: str = "*outbound*"
    ) -> pd.DataFrame:
        """
        Load outbound task data (OOK, OMK, NB).
        
        Args:
            filepath: Specific file path, or None to search in data_dir.
            filename_pattern: Glob pattern to find outbound files.
            
        Returns:
            Hourly aggregated outbound volume DataFrame with type breakdown.
        """
        if filepath:
            files = [Path(filepath)]
        else:
            files = list(self.data_dir.glob(f"{filename_pattern}.csv")) + \
                    list(self.data_dir.glob(f"{filename_pattern}.xlsx"))
        
        if not files:
            logger.warning("No outbound data files found")
            return pd.DataFrame()
        
        dfs = []
        for f in files:
            logger.info(f"Loading outbound from: {f}")
            df = self._read_file(f)
            
            # Find timestamp column
            ts_col = self._find_column(df, ["timestamp", "date", "datetime", "time", "datum"])
            if ts_col:
                df = df.rename(columns={ts_col: "timestamp"})
            
            df = self._parse_timestamp(df, "timestamp")
            
            # Find type column
            type_col = self._find_column(df, ["type", "typ", "category", "kategorie", "art"])
            if type_col:
                df = df.rename(columns={type_col: "outbound_type"})
            else:
                df["outbound_type"] = "unknown"
            
            # Normalize type values
            df["outbound_type"] = df["outbound_type"].str.upper().str.strip()
            
            # Find value column or aggregate records
            value_col = self._find_column(df, ["count", "volume", "anzahl", "amount"])
            if value_col:
                df = df.rename(columns={value_col: "outbound_count"})
            else:
                df["outbound_count"] = 1
            
            df["hour"] = df["timestamp"].dt.floor("H")
            
            # Aggregate by hour and type
            hourly = df.groupby(["hour", "outbound_type"]).agg({
                "outbound_count": "sum"
            }).reset_index()
            hourly = hourly.rename(columns={"hour": "timestamp"})
            
            dfs.append(hourly)
        
        if not dfs:
            return pd.DataFrame()
        
        result = pd.concat(dfs, ignore_index=True)
        result = result.groupby(["timestamp", "outbound_type"]).agg({
            "outbound_count": "sum"
        }).reset_index()
        
        # Pivot to get columns for each type
        pivot = result.pivot_table(
            index="timestamp",
            columns="outbound_type",
            values="outbound_count",
            fill_value=0
        ).reset_index()
        
        # Flatten column names
        pivot.columns = ["timestamp"] + [f"outbound_{col.lower()}" for col in pivot.columns[1:]]
        
        # Add total
        outbound_cols = [c for c in pivot.columns if c.startswith("outbound_")]
        pivot["outbound_total"] = pivot[outbound_cols].sum(axis=1)
        
        pivot = pivot.sort_values("timestamp").reset_index(drop=True)
        
        self.loaded_data["outbound"] = pivot
        logger.info(f"Loaded {len(pivot)} hourly outbound records")
        
        return pivot
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available data types.
        
        Returns:
            Dictionary with DataFrames for each data type found.
        """
        result = {}
        
        calls = self.load_calls()
        if not calls.empty:
            result["calls"] = calls
        
        emails = self.load_emails()
        if not emails.empty:
            result["emails"] = emails
        
        outbound = self.load_outbound()
        if not outbound.empty:
            result["outbound"] = outbound
        
        return result
    
    def combine_data(self, data: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """
        Combine all data types into a single DataFrame.
        
        Args:
            data: Dictionary of DataFrames, or None to use loaded_data.
            
        Returns:
            Combined DataFrame with all task volumes.
        """
        data = data or self.loaded_data
        
        if not data:
            raise ValueError("No data loaded. Call load_all() first.")
        
        # Start with calls or first available
        base = None
        for name, df in data.items():
            if df.empty:
                continue
            if base is None:
                base = df[["timestamp"]].copy()
                for col in df.columns:
                    if col != "timestamp":
                        base[col] = df[col]
            else:
                base = base.merge(df, on="timestamp", how="outer")
        
        if base is None:
            raise ValueError("No valid data to combine")
        
        # Sort and fill missing values
        base = base.sort_values("timestamp").reset_index(drop=True)
        
        # Fill missing hours (create complete hourly range)
        min_ts = base["timestamp"].min()
        max_ts = base["timestamp"].max()
        full_range = pd.date_range(start=min_ts, end=max_ts, freq="H")
        
        full_df = pd.DataFrame({"timestamp": full_range})
        combined = full_df.merge(base, on="timestamp", how="left")
        
        # Fill NaN with 0 for count columns
        numeric_cols = combined.select_dtypes(include=[np.number]).columns
        combined[numeric_cols] = combined[numeric_cols].fillna(0)
        
        return combined
    
    def save_processed(self, df: pd.DataFrame, filename: str = "combined_data.csv"):
        """
        Save processed data to the processed folder.
        
        Args:
            df: DataFrame to save.
            filename: Output filename.
        """
        output_path = DATA_PROCESSED / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to: {output_path}")


def create_sample_data(output_dir: Path = DATA_RAW):
    """
    Create sample data files for testing.
    
    Args:
        output_dir: Directory to save sample files.
    """
    import numpy as np
    from datetime import datetime, timedelta
    
    np.random.seed(42)
    
    # Generate 6 months of hourly data ending yesterday
    end_date = datetime.now().replace(hour=23, minute=0, second=0, microsecond=0) - timedelta(days=1)
    hours = 24 * 180  # 6 months
    start_date = end_date - timedelta(hours=hours-1)
    timestamps = [start_date + timedelta(hours=i) for i in range(hours)]
    
    # Create realistic patterns
    def generate_volume(timestamps, base, daily_pattern, weekly_pattern, noise_level=0.2):
        volumes = []
        for ts in timestamps:
            hour = ts.hour
            weekday = ts.weekday()
            
            # Base volume with daily pattern (peak during work hours)
            if 8 <= hour <= 18:
                daily_factor = daily_pattern[hour - 8]
            else:
                daily_factor = 0.1
            
            # Weekly pattern (lower on weekends)
            weekly_factor = weekly_pattern[weekday]
            
            # Calculate volume
            vol = base * daily_factor * weekly_factor
            vol *= (1 + np.random.normal(0, noise_level))
            volumes.append(max(0, int(vol)))
        
        return volumes
    
    # Daily patterns (index 0 = 8am, index 10 = 6pm)
    call_daily = [0.6, 0.8, 1.0, 1.2, 1.1, 0.9, 1.0, 1.1, 1.0, 0.8, 0.5]
    email_daily = [0.8, 1.0, 1.1, 1.0, 0.9, 0.8, 0.9, 1.0, 0.9, 0.7, 0.4]
    
    # Weekly patterns (Mon=0 to Sun=6)
    weekly = [1.0, 1.1, 1.0, 0.95, 0.9, 0.3, 0.2]
    
    # Generate call data
    calls_df = pd.DataFrame({
        "timestamp": timestamps,
        "call_volume": generate_volume(timestamps, 50, call_daily, weekly)
    })
    calls_df.to_csv(output_dir / "calls_history.csv", index=False)
    
    # Generate email data
    emails_df = pd.DataFrame({
        "timestamp": timestamps,
        "email_count": generate_volume(timestamps, 30, email_daily, weekly, 0.3)
    })
    emails_df.to_csv(output_dir / "emails_history.csv", index=False)
    
    # Generate outbound data
    outbound_types = ["OOK", "OMK", "NB"]
    outbound_records = []
    
    for ts in timestamps:
        for otype in outbound_types:
            base_vol = {"OOK": 15, "OMK": 10, "NB": 20}[otype]
            vol = generate_volume([ts], base_vol, call_daily, weekly, 0.25)[0]
            if vol > 0:
                outbound_records.append({
                    "timestamp": ts,
                    "type": otype,
                    "count": vol
                })
    
    outbound_df = pd.DataFrame(outbound_records)
    outbound_df.to_csv(output_dir / "outbound_history.csv", index=False)
    
    logger.info(f"Created sample data files in: {output_dir}")


if __name__ == "__main__":
    # Create sample data for testing
    create_sample_data()
    
    # Test loading
    loader = DataLoader()
    data = loader.load_all()
    
    print("\nLoaded data summary:")
    for name, df in data.items():
        print(f"  {name}: {len(df)} records")
    
    combined = loader.combine_data()
    print(f"\nCombined data: {len(combined)} records")
    print(combined.head())


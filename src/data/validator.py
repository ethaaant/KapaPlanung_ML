"""
Data validation module.
Validates uploaded data and provides warnings for potential issues.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"      # Data cannot be used
    WARNING = "warning"  # Data can be used but may have issues
    INFO = "info"        # Informational message


@dataclass
class ValidationIssue:
    """A single validation issue."""
    severity: ValidationSeverity
    code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    affected_rows: List[int] = field(default_factory=list)
    
    def __str__(self):
        return f"[{self.severity.value.upper()}] {self.message}"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]
    
    @property
    def infos(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.INFO]
    
    def add_issue(
        self,
        severity: ValidationSeverity,
        code: str,
        message: str,
        details: Dict = None,
        affected_rows: List[int] = None
    ):
        """Add a validation issue."""
        self.issues.append(ValidationIssue(
            severity=severity,
            code=code,
            message=message,
            details=details or {},
            affected_rows=affected_rows or []
        ))
        
        # Update is_valid if error
        if severity == ValidationSeverity.ERROR:
            self.is_valid = False


class DataValidator:
    """
    Validates workload data for the forecasting system.
    """
    
    REQUIRED_COLUMNS = {
        "calls": ["timestamp", "calls"],
        "emails": ["timestamp", "emails"],
        "outbound": ["timestamp", "outbound_ook", "outbound_omk", "outbound_nb"]
    }
    
    EXPECTED_VALUE_RANGES = {
        "calls": (0, 10000),         # Per hour
        "emails": (0, 5000),          # Per hour
        "outbound_ook": (0, 2000),    # Per hour
        "outbound_omk": (0, 2000),    # Per hour
        "outbound_nb": (0, 2000),     # Per hour
        "total_workload": (0, 50000)  # Per hour
    }
    
    def __init__(self):
        self.result = ValidationResult(is_valid=True)
    
    def validate(self, data: pd.DataFrame, data_type: str = None) -> ValidationResult:
        """
        Validate a dataframe.
        
        Args:
            data: The data to validate
            data_type: Type of data ('calls', 'emails', 'outbound', 'combined')
            
        Returns:
            ValidationResult with issues and summary
        """
        self.result = ValidationResult(is_valid=True)
        
        if data is None or data.empty:
            self.result.add_issue(
                ValidationSeverity.ERROR,
                "DATA_EMPTY",
                "The provided data is empty"
            )
            return self.result
        
        # Basic structure validation
        self._validate_structure(data, data_type)
        
        # Timestamp validation
        if "timestamp" in data.columns:
            self._validate_timestamps(data)
        
        # Value validation
        self._validate_values(data)
        
        # Pattern validation
        self._validate_patterns(data)
        
        # Add summary
        self.result.summary = self._create_summary(data)
        
        return self.result
    
    def _validate_structure(self, data: pd.DataFrame, data_type: str):
        """Validate data structure and columns."""
        
        # Check for timestamp column
        if "timestamp" not in data.columns:
            self.result.add_issue(
                ValidationSeverity.ERROR,
                "MISSING_TIMESTAMP",
                "Data must contain a 'timestamp' column"
            )
        
        # Check for value columns
        value_columns = [
            "calls", "emails", "outbound_ook", "outbound_omk", 
            "outbound_nb", "total_workload"
        ]
        has_values = any(col in data.columns for col in value_columns)
        
        if not has_values:
            self.result.add_issue(
                ValidationSeverity.ERROR,
                "MISSING_VALUE_COLUMNS",
                "Data must contain at least one workload value column",
                details={"expected": value_columns}
            )
        
        # Check for specific data type requirements
        if data_type and data_type in self.REQUIRED_COLUMNS:
            required = self.REQUIRED_COLUMNS[data_type]
            missing = [col for col in required if col not in data.columns]
            if missing:
                self.result.add_issue(
                    ValidationSeverity.WARNING,
                    "MISSING_REQUIRED_COLUMNS",
                    f"Missing expected columns for {data_type} data: {missing}",
                    details={"missing": missing, "data_type": data_type}
                )
    
    def _validate_timestamps(self, data: pd.DataFrame):
        """Validate timestamp column."""
        
        # Check if timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            try:
                pd.to_datetime(data["timestamp"])
            except Exception:
                self.result.add_issue(
                    ValidationSeverity.ERROR,
                    "INVALID_TIMESTAMP_FORMAT",
                    "Cannot parse timestamp column as datetime"
                )
                return
        
        ts = pd.to_datetime(data["timestamp"])
        
        # Check for future timestamps
        future_mask = ts > datetime.now() + timedelta(days=1)
        if future_mask.any():
            future_count = future_mask.sum()
            self.result.add_issue(
                ValidationSeverity.WARNING,
                "FUTURE_TIMESTAMPS",
                f"Found {future_count} rows with timestamps in the future",
                affected_rows=data[future_mask].index.tolist()[:10]
            )
        
        # Check for very old timestamps
        old_cutoff = datetime.now() - timedelta(days=365 * 5)
        old_mask = ts < old_cutoff
        if old_mask.any():
            old_count = old_mask.sum()
            self.result.add_issue(
                ValidationSeverity.INFO,
                "OLD_TIMESTAMPS",
                f"Found {old_count} rows with timestamps older than 5 years",
                affected_rows=data[old_mask].index.tolist()[:10]
            )
        
        # Check for gaps in data
        if len(ts) > 1:
            ts_sorted = ts.sort_values()
            diffs = ts_sorted.diff().dropna()
            median_diff = diffs.median()
            
            # Find gaps larger than 3x median
            gap_mask = diffs > median_diff * 3
            if gap_mask.any():
                gap_count = gap_mask.sum()
                self.result.add_issue(
                    ValidationSeverity.WARNING,
                    "DATA_GAPS",
                    f"Found {gap_count} gaps in the data that are larger than expected",
                    details={
                        "median_interval": str(median_diff),
                        "largest_gap": str(diffs.max())
                    }
                )
        
        # Check for duplicate timestamps
        duplicates = ts.duplicated()
        if duplicates.any():
            dup_count = duplicates.sum()
            self.result.add_issue(
                ValidationSeverity.WARNING,
                "DUPLICATE_TIMESTAMPS",
                f"Found {dup_count} duplicate timestamps",
                affected_rows=data[duplicates].index.tolist()[:10]
            )
        
        # Check timestamp granularity
        if len(ts) > 1:
            ts_sorted = ts.sort_values()
            diffs = ts_sorted.diff().dropna()
            mode_diff = diffs.mode()
            
            if len(mode_diff) > 0:
                granularity = mode_diff.iloc[0]
                self.result.add_issue(
                    ValidationSeverity.INFO,
                    "TIMESTAMP_GRANULARITY",
                    f"Data appears to be at {self._humanize_timedelta(granularity)} granularity",
                    details={"granularity_seconds": granularity.total_seconds()}
                )
    
    def _validate_values(self, data: pd.DataFrame):
        """Validate numeric values."""
        
        value_columns = [col for col in data.columns 
                        if col in self.EXPECTED_VALUE_RANGES]
        
        for col in value_columns:
            if col not in data.columns:
                continue
            
            values = data[col]
            
            # Check for null values
            null_mask = values.isna()
            if null_mask.any():
                null_count = null_mask.sum()
                null_pct = (null_count / len(values)) * 100
                
                severity = (ValidationSeverity.ERROR if null_pct > 50 
                           else ValidationSeverity.WARNING if null_pct > 10
                           else ValidationSeverity.INFO)
                
                self.result.add_issue(
                    severity,
                    f"NULL_VALUES_{col.upper()}",
                    f"Found {null_count} ({null_pct:.1f}%) null values in '{col}'",
                    affected_rows=data[null_mask].index.tolist()[:10]
                )
            
            # Check for negative values
            neg_mask = values < 0
            if neg_mask.any():
                neg_count = neg_mask.sum()
                self.result.add_issue(
                    ValidationSeverity.ERROR,
                    f"NEGATIVE_VALUES_{col.upper()}",
                    f"Found {neg_count} negative values in '{col}' (workload cannot be negative)",
                    affected_rows=data[neg_mask].index.tolist()[:10]
                )
            
            # Check for out-of-range values
            min_val, max_val = self.EXPECTED_VALUE_RANGES.get(col, (0, float('inf')))
            high_mask = values > max_val
            if high_mask.any():
                high_count = high_mask.sum()
                max_found = values.max()
                self.result.add_issue(
                    ValidationSeverity.WARNING,
                    f"HIGH_VALUES_{col.upper()}",
                    f"Found {high_count} values in '{col}' above expected maximum ({max_val}). Max found: {max_found:.0f}",
                    details={"max_expected": max_val, "max_found": float(max_found)},
                    affected_rows=data[high_mask].index.tolist()[:10]
                )
            
            # Check for unusual zeros
            zero_mask = values == 0
            zero_pct = (zero_mask.sum() / len(values)) * 100
            if zero_pct > 30:
                self.result.add_issue(
                    ValidationSeverity.INFO,
                    f"HIGH_ZERO_RATIO_{col.upper()}",
                    f"{zero_pct:.1f}% of values in '{col}' are zero",
                    details={"zero_percentage": zero_pct}
                )
    
    def _validate_patterns(self, data: pd.DataFrame):
        """Validate expected patterns in data."""
        
        if "timestamp" not in data.columns:
            return
        
        ts = pd.to_datetime(data["timestamp"])
        
        # Check data range
        date_range = (ts.max() - ts.min()).days
        if date_range < 30:
            self.result.add_issue(
                ValidationSeverity.WARNING,
                "INSUFFICIENT_HISTORY",
                f"Data spans only {date_range} days. At least 30 days recommended for reliable forecasting",
                details={"days": date_range}
            )
        elif date_range < 90:
            self.result.add_issue(
                ValidationSeverity.INFO,
                "LIMITED_HISTORY",
                f"Data spans {date_range} days. 90+ days recommended for seasonal pattern detection",
                details={"days": date_range}
            )
        
        # Check for consistent hour coverage
        if len(data) > 24:
            hours = ts.dt.hour
            hour_counts = hours.value_counts()
            
            if len(hour_counts) < 24:
                missing_hours = set(range(24)) - set(hour_counts.index)
                if len(missing_hours) <= 8:  # Normal if night hours missing
                    self.result.add_issue(
                        ValidationSeverity.INFO,
                        "MISSING_HOURS",
                        f"Data is missing for hours: {sorted(missing_hours)}",
                        details={"missing_hours": sorted(missing_hours)}
                    )
    
    def _create_summary(self, data: pd.DataFrame) -> dict:
        """Create a summary of the validated data."""
        summary = {
            "total_rows": len(data),
            "columns": list(data.columns),
            "error_count": len(self.result.errors),
            "warning_count": len(self.result.warnings),
            "info_count": len(self.result.infos)
        }
        
        if "timestamp" in data.columns:
            ts = pd.to_datetime(data["timestamp"])
            summary["date_range"] = {
                "start": ts.min().isoformat() if not ts.empty else None,
                "end": ts.max().isoformat() if not ts.empty else None,
                "days": (ts.max() - ts.min()).days if not ts.empty else 0
            }
        
        # Value statistics
        value_columns = ["calls", "emails", "outbound_ook", "outbound_omk", 
                        "outbound_nb", "total_workload"]
        summary["statistics"] = {}
        
        for col in value_columns:
            if col in data.columns:
                summary["statistics"][col] = {
                    "mean": float(data[col].mean()) if not data[col].isna().all() else None,
                    "median": float(data[col].median()) if not data[col].isna().all() else None,
                    "std": float(data[col].std()) if not data[col].isna().all() else None,
                    "min": float(data[col].min()) if not data[col].isna().all() else None,
                    "max": float(data[col].max()) if not data[col].isna().all() else None
                }
        
        return summary
    
    def _humanize_timedelta(self, td: timedelta) -> str:
        """Convert timedelta to human-readable string."""
        seconds = td.total_seconds()
        
        if seconds < 60:
            return f"{int(seconds)} second(s)"
        elif seconds < 3600:
            return f"{int(seconds / 60)} minute(s)"
        elif seconds < 86400:
            return f"{int(seconds / 3600)} hour(s)"
        else:
            return f"{int(seconds / 86400)} day(s)"


def validate_data(data: pd.DataFrame, data_type: str = None) -> ValidationResult:
    """
    Convenience function to validate data.
    
    Args:
        data: The DataFrame to validate
        data_type: Optional type ('calls', 'emails', 'outbound', 'combined')
        
    Returns:
        ValidationResult
    """
    validator = DataValidator()
    return validator.validate(data, data_type)


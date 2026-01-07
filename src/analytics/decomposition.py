"""
Time Series Decomposition Module.
Provides trend, seasonality, and residual analysis with ACF/PACF plots.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Try to import statsmodels
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import acf, pacf, adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


@dataclass
class DecompositionResult:
    """Container for decomposition results."""
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    observed: pd.Series
    timestamps: pd.Series
    period: int
    model: str


@dataclass 
class StationarityResult:
    """Container for stationarity test results."""
    is_stationary: bool
    adf_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    n_lags: int


class TimeSeriesDecomposer:
    """
    Time series decomposition and analysis.
    Extracts trend, seasonality, and residual components.
    """
    
    def __init__(self):
        if not STATSMODELS_AVAILABLE:
            raise ImportError(
                "statsmodels is required for time series decomposition. "
                "Install with: pip install statsmodels"
            )
        self.decomposition: Optional[DecompositionResult] = None
        self.stationarity: Optional[StationarityResult] = None
    
    def decompose(
        self,
        data: pd.DataFrame,
        column: str,
        period: int = 24,  # Default: daily seasonality (hourly data)
        model: str = "additive"
    ) -> DecompositionResult:
        """
        Decompose time series into trend, seasonality, and residual.
        
        Args:
            data: DataFrame with timestamp and value columns
            column: Column name to decompose
            period: Seasonality period (24 for daily with hourly data)
            model: 'additive' or 'multiplicative'
            
        Returns:
            DecompositionResult with components
        """
        # Prepare data
        df = data.copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Handle missing values
        series = df[column].interpolate(method="linear")
        
        # Perform decomposition
        result = seasonal_decompose(
            series,
            period=period,
            model=model,
            extrapolate_trend="freq"
        )
        
        self.decomposition = DecompositionResult(
            trend=pd.Series(result.trend),
            seasonal=pd.Series(result.seasonal),
            residual=pd.Series(result.resid),
            observed=pd.Series(result.observed),
            timestamps=df["timestamp"],
            period=period,
            model=model
        )
        
        return self.decomposition
    
    def test_stationarity(
        self,
        data: pd.DataFrame,
        column: str,
        max_lags: int = None
    ) -> StationarityResult:
        """
        Test for stationarity using Augmented Dickey-Fuller test.
        
        Args:
            data: DataFrame with the time series
            column: Column to test
            max_lags: Maximum lags to include in test
            
        Returns:
            StationarityResult with test statistics
        """
        series = data[column].dropna()
        
        result = adfuller(series, maxlag=max_lags, autolag="AIC")
        
        self.stationarity = StationarityResult(
            is_stationary=result[1] < 0.05,
            adf_statistic=result[0],
            p_value=result[1],
            critical_values={
                "1%": result[4]["1%"],
                "5%": result[4]["5%"],
                "10%": result[4]["10%"]
            },
            n_lags=result[2]
        )
        
        return self.stationarity
    
    def calculate_acf_pacf(
        self,
        data: pd.DataFrame,
        column: str,
        n_lags: int = 48
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ACF and PACF values.
        
        Args:
            data: DataFrame with time series
            column: Column to analyze
            n_lags: Number of lags to compute
            
        Returns:
            Tuple of (acf_values, acf_conf, pacf_values, pacf_conf)
        """
        series = data[column].dropna()
        
        acf_values, acf_conf = acf(series, nlags=n_lags, alpha=0.05)
        pacf_values, pacf_conf = pacf(series, nlags=n_lags, alpha=0.05)
        
        return acf_values, acf_conf, pacf_values, pacf_conf
    
    def plot_decomposition(self) -> go.Figure:
        """Plot the decomposition components."""
        if self.decomposition is None:
            raise ValueError("Run decompose() first")
        
        d = self.decomposition
        
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
            vertical_spacing=0.05
        )
        
        # Observed
        fig.add_trace(
            go.Scatter(
                x=d.timestamps, y=d.observed,
                name="Observed",
                line=dict(color="#667eea", width=1)
            ),
            row=1, col=1
        )
        
        # Trend
        fig.add_trace(
            go.Scatter(
                x=d.timestamps, y=d.trend,
                name="Trend",
                line=dict(color="#10b981", width=2)
            ),
            row=2, col=1
        )
        
        # Seasonal
        fig.add_trace(
            go.Scatter(
                x=d.timestamps, y=d.seasonal,
                name="Seasonal",
                line=dict(color="#f59e0b", width=1)
            ),
            row=3, col=1
        )
        
        # Residual
        fig.add_trace(
            go.Scatter(
                x=d.timestamps, y=d.residual,
                name="Residual",
                mode="markers",
                marker=dict(color="#ef4444", size=2, opacity=0.5)
            ),
            row=4, col=1
        )
        
        fig.update_layout(
            title=f"Time Series Decomposition ({d.model.title()}, Period={d.period})",
            height=700,
            showlegend=False
        )
        
        return fig
    
    def plot_seasonal_pattern(self, aggregate_by: str = "hour") -> go.Figure:
        """
        Plot aggregated seasonal pattern.
        
        Args:
            aggregate_by: 'hour' for daily pattern, 'dayofweek' for weekly pattern
        """
        if self.decomposition is None:
            raise ValueError("Run decompose() first")
        
        d = self.decomposition
        df = pd.DataFrame({
            "timestamp": d.timestamps,
            "seasonal": d.seasonal,
            "observed": d.observed
        })
        
        if aggregate_by == "hour":
            df["group"] = pd.to_datetime(df["timestamp"]).dt.hour
            x_label = "Hour of Day"
            x_vals = list(range(24))
        else:
            df["group"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
            x_label = "Day of Week"
            x_vals = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        
        # Calculate mean and std for each group
        grouped = df.groupby("group").agg({
            "seasonal": ["mean", "std"],
            "observed": ["mean", "std"]
        }).reset_index()
        
        grouped.columns = ["group", "seasonal_mean", "seasonal_std", 
                          "observed_mean", "observed_std"]
        
        fig = go.Figure()
        
        # Observed pattern
        fig.add_trace(go.Scatter(
            x=grouped["group"] if aggregate_by == "hour" else x_vals,
            y=grouped["observed_mean"],
            name="Average Observed",
            line=dict(color="#667eea", width=2),
            mode="lines+markers"
        ))
        
        # Add confidence band
        fig.add_trace(go.Scatter(
            x=list(grouped["group"]) + list(grouped["group"])[::-1] if aggregate_by == "hour" 
              else x_vals + x_vals[::-1],
            y=list(grouped["observed_mean"] + grouped["observed_std"]) + 
              list(grouped["observed_mean"] - grouped["observed_std"])[::-1],
            fill="toself",
            fillcolor="rgba(102, 126, 234, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="±1 Std Dev",
            showlegend=True
        ))
        
        fig.update_layout(
            title=f"Seasonal Pattern by {x_label}",
            xaxis_title=x_label,
            yaxis_title="Value",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def plot_acf_pacf(
        self,
        data: pd.DataFrame,
        column: str,
        n_lags: int = 48
    ) -> go.Figure:
        """Plot ACF and PACF."""
        acf_vals, acf_conf, pacf_vals, pacf_conf = self.calculate_acf_pacf(
            data, column, n_lags
        )
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                "Autocorrelation Function (ACF)",
                "Partial Autocorrelation Function (PACF)"
            ),
            vertical_spacing=0.15
        )
        
        lags = list(range(len(acf_vals)))
        
        # ACF
        fig.add_trace(
            go.Bar(
                x=lags, y=acf_vals,
                name="ACF",
                marker_color="#667eea"
            ),
            row=1, col=1
        )
        
        # ACF confidence interval
        conf_upper = 1.96 / np.sqrt(len(data))
        fig.add_hline(y=conf_upper, line_dash="dash", line_color="#ef4444", row=1, col=1)
        fig.add_hline(y=-conf_upper, line_dash="dash", line_color="#ef4444", row=1, col=1)
        fig.add_hline(y=0, line_color="black", row=1, col=1)
        
        # PACF
        fig.add_trace(
            go.Bar(
                x=lags[:len(pacf_vals)], y=pacf_vals,
                name="PACF",
                marker_color="#10b981"
            ),
            row=2, col=1
        )
        
        # PACF confidence interval
        fig.add_hline(y=conf_upper, line_dash="dash", line_color="#ef4444", row=2, col=1)
        fig.add_hline(y=-conf_upper, line_dash="dash", line_color="#ef4444", row=2, col=1)
        fig.add_hline(y=0, line_color="black", row=2, col=1)
        
        fig.update_layout(
            title=f"ACF and PACF Analysis: {column.replace('_', ' ').title()}",
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Lag", row=1, col=1)
        fig.update_xaxes(title_text="Lag", row=2, col=1)
        fig.update_yaxes(title_text="Correlation", row=1, col=1)
        fig.update_yaxes(title_text="Correlation", row=2, col=1)
        
        return fig
    
    def plot_trend_analysis(self) -> go.Figure:
        """Plot trend with moving averages."""
        if self.decomposition is None:
            raise ValueError("Run decompose() first")
        
        d = self.decomposition
        df = pd.DataFrame({
            "timestamp": d.timestamps,
            "observed": d.observed,
            "trend": d.trend
        })
        
        fig = go.Figure()
        
        # Observed data
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["observed"],
            name="Observed",
            line=dict(color="#e5e7eb", width=1),
            opacity=0.5
        ))
        
        # Trend
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["trend"],
            name="Trend",
            line=dict(color="#667eea", width=2)
        ))
        
        # 7-day moving average
        ma_7d = df["observed"].rolling(window=24*7).mean()
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=ma_7d,
            name="7-Day MA",
            line=dict(color="#10b981", width=2)
        ))
        
        # 30-day moving average
        ma_30d = df["observed"].rolling(window=24*30).mean()
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=ma_30d,
            name="30-Day MA",
            line=dict(color="#f59e0b", width=2)
        ))
        
        fig.update_layout(
            title="Trend Analysis with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Value",
            height=450,
            showlegend=True
        )
        
        return fig
    
    def get_decomposition_summary(self) -> Dict:
        """Get summary statistics from decomposition."""
        if self.decomposition is None:
            raise ValueError("Run decompose() first")
        
        d = self.decomposition
        
        # Trend statistics
        trend_clean = d.trend.dropna()
        trend_change = (trend_clean.iloc[-1] - trend_clean.iloc[0]) / trend_clean.iloc[0] * 100
        
        # Seasonal strength
        var_residual = d.residual.var()
        var_seasonal = d.seasonal.var()
        seasonal_strength = max(0, 1 - var_residual / (var_seasonal + var_residual))
        
        # Trend strength
        detrended = d.observed - d.trend
        var_detrended = detrended.var()
        trend_strength = max(0, 1 - var_residual / var_detrended) if var_detrended > 0 else 0
        
        return {
            "model": d.model,
            "period": d.period,
            "trend": {
                "start_value": float(trend_clean.iloc[0]),
                "end_value": float(trend_clean.iloc[-1]),
                "change_percent": float(trend_change),
                "direction": "increasing" if trend_change > 0 else "decreasing",
                "strength": float(trend_strength)
            },
            "seasonality": {
                "strength": float(seasonal_strength),
                "max_effect": float(d.seasonal.max()),
                "min_effect": float(d.seasonal.min()),
                "range": float(d.seasonal.max() - d.seasonal.min())
            },
            "residual": {
                "mean": float(d.residual.mean()),
                "std": float(d.residual.std()),
                "max": float(d.residual.max()),
                "min": float(d.residual.min())
            }
        }


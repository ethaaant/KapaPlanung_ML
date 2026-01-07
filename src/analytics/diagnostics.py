"""
Model Diagnostics Module.
Provides residual analysis, error breakdown, and model performance diagnostics.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class DiagnosticResults:
    """Container for diagnostic results."""
    residuals: pd.DataFrame
    predictions: pd.DataFrame
    actuals: pd.DataFrame
    metrics_by_hour: pd.DataFrame
    metrics_by_day: pd.DataFrame
    normality_test: Dict[str, float]
    heteroscedasticity: Dict[str, bool]


class ModelDiagnostics:
    """
    Comprehensive model diagnostics for time series forecasting.
    """
    
    def __init__(self):
        self.results: Optional[DiagnosticResults] = None
    
    def analyze(
        self,
        actuals: pd.DataFrame,
        predictions: pd.DataFrame,
        timestamps: pd.Series,
        target_columns: List[str]
    ) -> DiagnosticResults:
        """
        Perform comprehensive diagnostic analysis.
        
        Args:
            actuals: DataFrame with actual values
            predictions: DataFrame with predicted values
            timestamps: Series of timestamps
            target_columns: List of target column names
            
        Returns:
            DiagnosticResults with all analysis
        """
        # Calculate residuals
        residuals = actuals[target_columns] - predictions[target_columns]
        residuals["timestamp"] = timestamps.values
        
        # Add time features for breakdown analysis
        residuals["hour"] = pd.to_datetime(timestamps).dt.hour
        residuals["day_of_week"] = pd.to_datetime(timestamps).dt.dayofweek
        residuals["day_name"] = pd.to_datetime(timestamps).dt.day_name()
        
        # Metrics by hour
        metrics_by_hour = self._calculate_metrics_by_group(
            residuals, actuals, predictions, target_columns, "hour"
        )
        
        # Metrics by day of week
        metrics_by_day = self._calculate_metrics_by_group(
            residuals, actuals, predictions, target_columns, "day_of_week"
        )
        
        # Normality test on residuals
        normality_test = {}
        for col in target_columns:
            stat, p_value = stats.shapiro(residuals[col].dropna()[:5000])  # Limit for performance
            normality_test[col] = {"statistic": stat, "p_value": p_value}
        
        # Heteroscedasticity check (variance changing with prediction size)
        heteroscedasticity = {}
        for col in target_columns:
            # Simple check: compare variance in low vs high prediction ranges
            median_pred = predictions[col].median()
            low_residuals = residuals.loc[predictions[col] < median_pred, col]
            high_residuals = residuals.loc[predictions[col] >= median_pred, col]
            
            if len(low_residuals) > 10 and len(high_residuals) > 10:
                _, p_value = stats.levene(low_residuals, high_residuals)
                heteroscedasticity[col] = p_value < 0.05  # True = heteroscedastic
            else:
                heteroscedasticity[col] = None
        
        self.results = DiagnosticResults(
            residuals=residuals,
            predictions=predictions,
            actuals=actuals,
            metrics_by_hour=metrics_by_hour,
            metrics_by_day=metrics_by_day,
            normality_test=normality_test,
            heteroscedasticity=heteroscedasticity
        )
        
        return self.results
    
    def _calculate_metrics_by_group(
        self,
        residuals: pd.DataFrame,
        actuals: pd.DataFrame,
        predictions: pd.DataFrame,
        target_columns: List[str],
        group_col: str
    ) -> pd.DataFrame:
        """Calculate error metrics grouped by a column."""
        metrics = []
        
        for group_val in residuals[group_col].unique():
            mask = residuals[group_col] == group_val
            
            for col in target_columns:
                res = residuals.loc[mask, col]
                act = actuals.loc[mask, col] if col in actuals.columns else None
                
                mae = np.abs(res).mean()
                rmse = np.sqrt((res ** 2).mean())
                bias = res.mean()
                
                # MAPE
                if act is not None and (act > 0).any():
                    mape = np.abs(res / act.replace(0, np.nan)).mean() * 100
                else:
                    mape = np.nan
                
                metrics.append({
                    group_col: group_val,
                    "target": col,
                    "mae": mae,
                    "rmse": rmse,
                    "bias": bias,
                    "mape": mape,
                    "count": len(res)
                })
        
        return pd.DataFrame(metrics)
    
    def plot_residuals_distribution(self, target: str) -> go.Figure:
        """Plot residual distribution with normal curve overlay."""
        if self.results is None:
            raise ValueError("Run analyze() first")
        
        residuals = self.results.residuals[target].dropna()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Residual Distribution", "Q-Q Plot"),
            column_widths=[0.6, 0.4]
        )
        
        # Histogram with KDE
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name="Residuals",
                nbinsx=50,
                histnorm="probability density",
                marker_color="#667eea",
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Normal curve overlay
        x_range = np.linspace(residuals.min(), residuals.max(), 100)
        normal_pdf = stats.norm.pdf(x_range, residuals.mean(), residuals.std())
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_pdf,
                name="Normal Distribution",
                line=dict(color="#ef4444", width=2)
            ),
            row=1, col=1
        )
        
        # Q-Q Plot
        theoretical_quantiles = stats.norm.ppf(
            np.linspace(0.01, 0.99, len(residuals))
        )
        sample_quantiles = np.sort(residuals)[:len(theoretical_quantiles)]
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode="markers",
                name="Q-Q",
                marker=dict(color="#667eea", size=4)
            ),
            row=1, col=2
        )
        
        # Reference line for Q-Q
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val * residuals.std() + residuals.mean(), 
                   max_val * residuals.std() + residuals.mean()],
                mode="lines",
                name="Reference",
                line=dict(color="#ef4444", dash="dash")
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f"Residual Analysis: {target.replace('_', ' ').title()}",
            showlegend=True,
            height=400
        )
        
        fig.update_xaxes(title_text="Residual Value", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
        
        return fig
    
    def plot_prediction_vs_actual(self, target: str) -> go.Figure:
        """Plot predictions vs actuals scatter plot."""
        if self.results is None:
            raise ValueError("Run analyze() first")
        
        actuals = self.results.actuals[target]
        predictions = self.results.predictions[target]
        
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=actuals,
            y=predictions,
            mode="markers",
            name="Predictions",
            marker=dict(
                color="#667eea",
                size=5,
                opacity=0.5
            )
        ))
        
        # Perfect prediction line
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Perfect Prediction",
            line=dict(color="#ef4444", dash="dash", width=2)
        ))
        
        # Calculate R²
        correlation = np.corrcoef(actuals, predictions)[0, 1]
        r_squared = correlation ** 2
        
        fig.update_layout(
            title=f"Prediction vs Actual: {target.replace('_', ' ').title()} (R² = {r_squared:.3f})",
            xaxis_title="Actual Value",
            yaxis_title="Predicted Value",
            height=450,
            showlegend=True
        )
        
        return fig
    
    def plot_residuals_over_time(self, target: str) -> go.Figure:
        """Plot residuals over time to detect patterns."""
        if self.results is None:
            raise ValueError("Run analyze() first")
        
        residuals = self.results.residuals
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=residuals["timestamp"],
            y=residuals[target],
            mode="markers",
            name="Residuals",
            marker=dict(color="#667eea", size=3, opacity=0.5)
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="#ef4444")
        
        # Add rolling mean
        rolling_mean = residuals[target].rolling(24).mean()
        fig.add_trace(go.Scatter(
            x=residuals["timestamp"],
            y=rolling_mean,
            mode="lines",
            name="24h Rolling Mean",
            line=dict(color="#10b981", width=2)
        ))
        
        fig.update_layout(
            title=f"Residuals Over Time: {target.replace('_', ' ').title()}",
            xaxis_title="Time",
            yaxis_title="Residual (Actual - Predicted)",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def plot_error_by_hour(self, target: str = None) -> go.Figure:
        """Plot error metrics by hour of day."""
        if self.results is None:
            raise ValueError("Run analyze() first")
        
        df = self.results.metrics_by_hour
        
        if target:
            df = df[df["target"] == target]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("MAE by Hour", "Bias by Hour")
        )
        
        for t in df["target"].unique():
            target_df = df[df["target"] == t].sort_values("hour")
            
            fig.add_trace(
                go.Bar(
                    x=target_df["hour"],
                    y=target_df["mae"],
                    name=f"{t} MAE",
                    marker_color="#667eea" if t == df["target"].unique()[0] else "#10b981"
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=target_df["hour"],
                    y=target_df["bias"],
                    name=f"{t} Bias",
                    marker_color="#667eea" if t == df["target"].unique()[0] else "#10b981"
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title="Error Analysis by Hour of Day",
            height=400,
            showlegend=True,
            barmode="group"
        )
        
        fig.update_xaxes(title_text="Hour", row=1, col=1)
        fig.update_xaxes(title_text="Hour", row=1, col=2)
        fig.update_yaxes(title_text="MAE", row=1, col=1)
        fig.update_yaxes(title_text="Bias", row=1, col=2)
        
        return fig
    
    def plot_error_by_day(self, target: str = None) -> go.Figure:
        """Plot error metrics by day of week."""
        if self.results is None:
            raise ValueError("Run analyze() first")
        
        df = self.results.metrics_by_day
        
        if target:
            df = df[df["target"] == target]
        
        day_order = [0, 1, 2, 3, 4, 5, 6]
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        
        fig = go.Figure()
        
        for t in df["target"].unique():
            target_df = df[df["target"] == t].sort_values("day_of_week")
            
            fig.add_trace(go.Bar(
                x=[day_names[d] for d in target_df["day_of_week"]],
                y=target_df["mape"],
                name=t.replace("_", " ").title(),
                text=target_df["mape"].round(1).astype(str) + "%",
                textposition="outside"
            ))
        
        fig.update_layout(
            title="MAPE by Day of Week",
            xaxis_title="Day",
            yaxis_title="MAPE (%)",
            height=400,
            barmode="group",
            showlegend=True
        )
        
        return fig
    
    def get_diagnostic_summary(self) -> Dict:
        """Get a summary of diagnostic results."""
        if self.results is None:
            raise ValueError("Run analyze() first")
        
        summary = {
            "overall_metrics": {},
            "normality": {},
            "heteroscedasticity": {},
            "worst_hours": {},
            "worst_days": {}
        }
        
        # Overall metrics
        for col in self.results.residuals.columns:
            if col in ["timestamp", "hour", "day_of_week", "day_name"]:
                continue
            
            res = self.results.residuals[col]
            summary["overall_metrics"][col] = {
                "mae": np.abs(res).mean(),
                "rmse": np.sqrt((res ** 2).mean()),
                "bias": res.mean(),
                "std": res.std()
            }
        
        # Normality
        for col, result in self.results.normality_test.items():
            summary["normality"][col] = {
                "is_normal": result["p_value"] > 0.05,
                "p_value": result["p_value"]
            }
        
        # Heteroscedasticity
        summary["heteroscedasticity"] = self.results.heteroscedasticity
        
        # Worst performing hours/days
        for col in self.results.metrics_by_hour["target"].unique():
            hour_data = self.results.metrics_by_hour[
                self.results.metrics_by_hour["target"] == col
            ]
            worst_hour = hour_data.loc[hour_data["mape"].idxmax()]
            summary["worst_hours"][col] = {
                "hour": int(worst_hour["hour"]),
                "mape": worst_hour["mape"]
            }
            
            day_data = self.results.metrics_by_day[
                self.results.metrics_by_day["target"] == col
            ]
            worst_day = day_data.loc[day_data["mape"].idxmax()]
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", 
                        "Friday", "Saturday", "Sunday"]
            summary["worst_days"][col] = {
                "day": day_names[int(worst_day["day_of_week"])],
                "mape": worst_day["mape"]
            }
        
        return summary


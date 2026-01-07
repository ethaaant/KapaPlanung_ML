"""
What-If Scenario Analysis Module.
Provides interactive scenario modeling and sensitivity analysis.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class Scenario:
    """A single scenario definition."""
    name: str
    description: str
    volume_multiplier: float = 1.0  # Multiplier for workload
    aht_multiplier: float = 1.0  # Multiplier for handle time
    service_level: float = 0.80
    shrinkage: float = 0.30
    custom_adjustments: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Results from a scenario analysis."""
    scenario: Scenario
    staffing_plan: pd.DataFrame
    total_fte: float
    peak_agents: int
    avg_agents: float
    cost_estimate: float  # If cost per agent is provided


class ScenarioAnalyzer:
    """
    What-if scenario analysis for workforce planning.
    Allows testing different assumptions and comparing outcomes.
    """
    
    def __init__(
        self,
        base_forecast: pd.DataFrame,
        capacity_planner,  # CapacityPlanner instance
        cost_per_agent_hour: float = 25.0  # Default cost
    ):
        """
        Initialize scenario analyzer.
        
        Args:
            base_forecast: Base forecast DataFrame
            capacity_planner: CapacityPlanner instance for calculations
            cost_per_agent_hour: Cost per agent per hour for cost estimation
        """
        self.base_forecast = base_forecast.copy()
        self.capacity_planner = capacity_planner
        self.cost_per_agent_hour = cost_per_agent_hour
        self.scenarios: Dict[str, ScenarioResult] = {}
        
        # Calculate base scenario
        self._calculate_base()
    
    def _calculate_base(self):
        """Calculate the base scenario."""
        base = Scenario(
            name="Base",
            description="Current forecast without modifications"
        )
        
        result = self._run_scenario(base)
        self.scenarios["Base"] = result
        self.base_result = result
    
    def _run_scenario(self, scenario: Scenario) -> ScenarioResult:
        """Run a single scenario."""
        # Adjust forecast based on scenario
        adjusted = self.base_forecast.copy()
        
        # Apply volume multiplier to workload columns
        workload_cols = [c for c in adjusted.columns 
                        if c not in ["timestamp", "date", "hour"]]
        
        for col in workload_cols:
            adjusted[col] = adjusted[col] * scenario.volume_multiplier
            
            # Apply custom adjustments if any
            if col in scenario.custom_adjustments:
                adjusted[col] = adjusted[col] * scenario.custom_adjustments[col]
        
        # Calculate staffing using create_staffing_plan (for DataFrames)
        staffing = self.capacity_planner.create_staffing_plan(adjusted)
        
        # Apply AHT multiplier effect (more handle time = more agents needed)
        if scenario.aht_multiplier != 1.0:
            if "total_agents" in staffing.columns:
                staffing["total_agents"] = staffing["total_agents"] * scenario.aht_multiplier
            agent_cols = [c for c in staffing.columns if c.endswith("_agents")]
            for col in agent_cols:
                staffing[col] = staffing[col] * scenario.aht_multiplier
        
        # Apply shrinkage adjustment (relative to default 30%)
        if scenario.shrinkage != 0.30 and "total_agents" in staffing.columns:
            shrinkage_factor = (1 - 0.30) / (1 - scenario.shrinkage)
            staffing["total_agents"] = staffing["total_agents"] * shrinkage_factor
            agent_cols = [c for c in staffing.columns if c.endswith("_agents") and c != "total_agents"]
            for col in agent_cols:
                staffing[col] = staffing[col] * shrinkage_factor
        
        # Calculate metrics
        if "total_agents" in staffing.columns:
            total_fte = staffing["total_agents"].sum()
            peak_agents = int(staffing["total_agents"].max())
            avg_agents = staffing["total_agents"].mean()
        else:
            # Fallback if total_agents not present
            agent_cols = [c for c in staffing.columns if c.endswith("_agents")]
            if agent_cols:
                total_fte = staffing[agent_cols].sum().sum()
                peak_agents = int(staffing[agent_cols].sum(axis=1).max())
                avg_agents = staffing[agent_cols].sum(axis=1).mean()
            else:
                total_fte = 0
                peak_agents = 0
                avg_agents = 0
        
        cost_estimate = total_fte * self.cost_per_agent_hour
        
        return ScenarioResult(
            scenario=scenario,
            staffing_plan=staffing,
            total_fte=total_fte,
            peak_agents=peak_agents,
            avg_agents=avg_agents,
            cost_estimate=cost_estimate
        )
    
    def add_scenario(
        self,
        name: str,
        description: str,
        volume_change_pct: float = 0,
        aht_change_pct: float = 0,
        service_level: float = 0.80,
        shrinkage: float = 0.30,
        custom_adjustments: Dict[str, float] = None
    ) -> ScenarioResult:
        """
        Add and calculate a new scenario.
        
        Args:
            name: Scenario name
            description: Description of the scenario
            volume_change_pct: Percentage change in volume (-50 to +100)
            aht_change_pct: Percentage change in AHT (-50 to +100)
            service_level: Target service level (0.5 to 0.99)
            shrinkage: Shrinkage factor (0.1 to 0.5)
            custom_adjustments: Dict of {column: multiplier} for specific adjustments
            
        Returns:
            ScenarioResult
        """
        scenario = Scenario(
            name=name,
            description=description,
            volume_multiplier=1 + (volume_change_pct / 100),
            aht_multiplier=1 + (aht_change_pct / 100),
            service_level=service_level,
            shrinkage=shrinkage,
            custom_adjustments=custom_adjustments or {}
        )
        
        result = self._run_scenario(scenario)
        self.scenarios[name] = result
        
        return result
    
    def compare_scenarios(self, scenario_names: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple scenarios.
        
        Args:
            scenario_names: List of scenario names to compare (default: all)
            
        Returns:
            DataFrame with comparison metrics
        """
        if scenario_names is None:
            scenario_names = list(self.scenarios.keys())
        
        comparisons = []
        base = self.base_result
        
        for name in scenario_names:
            if name not in self.scenarios:
                continue
            
            result = self.scenarios[name]
            
            comparisons.append({
                "Scenario": name,
                "Description": result.scenario.description,
                "Volume Change": f"{(result.scenario.volume_multiplier - 1) * 100:+.0f}%",
                "AHT Change": f"{(result.scenario.aht_multiplier - 1) * 100:+.0f}%",
                "Total FTE Hours": f"{result.total_fte:,.0f}",
                "Peak Agents": result.peak_agents,
                "Avg Agents": f"{result.avg_agents:.1f}",
                "Est. Cost": f"€{result.cost_estimate:,.0f}",
                "vs Base FTE": f"{((result.total_fte / base.total_fte) - 1) * 100:+.1f}%" if name != "Base" else "-",
                "vs Base Cost": f"{((result.cost_estimate / base.cost_estimate) - 1) * 100:+.1f}%" if name != "Base" else "-"
            })
        
        return pd.DataFrame(comparisons)
    
    def plot_scenario_comparison(self, scenario_names: List[str] = None) -> go.Figure:
        """Plot bar chart comparing scenarios."""
        if scenario_names is None:
            scenario_names = list(self.scenarios.keys())
        
        names = []
        fte_values = []
        cost_values = []
        peak_values = []
        
        for name in scenario_names:
            if name not in self.scenarios:
                continue
            result = self.scenarios[name]
            names.append(name)
            fte_values.append(result.total_fte)
            cost_values.append(result.cost_estimate)
            peak_values.append(result.peak_agents)
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Total FTE Hours", "Estimated Cost (€)", "Peak Agents")
        )
        
        colors = ["#667eea", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]
        
        fig.add_trace(
            go.Bar(
                x=names, y=fte_values,
                name="FTE Hours",
                marker_color=colors[:len(names)],
                text=[f"{v:,.0f}" for v in fte_values],
                textposition="outside"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=names, y=cost_values,
                name="Cost",
                marker_color=colors[:len(names)],
                text=[f"€{v:,.0f}" for v in cost_values],
                textposition="outside"
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=names, y=peak_values,
                name="Peak",
                marker_color=colors[:len(names)],
                text=[str(v) for v in peak_values],
                textposition="outside"
            ),
            row=1, col=3
        )
        
        fig.update_layout(
            title="Scenario Comparison",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def plot_staffing_comparison(
        self,
        scenario_names: List[str] = None,
        aggregate: str = "hour"
    ) -> go.Figure:
        """
        Plot staffing patterns for multiple scenarios.
        
        Args:
            scenario_names: Scenarios to compare
            aggregate: 'hour' for hourly pattern, 'day' for daily
        """
        if scenario_names is None:
            scenario_names = list(self.scenarios.keys())
        
        fig = go.Figure()
        colors = ["#667eea", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]
        
        for i, name in enumerate(scenario_names):
            if name not in self.scenarios:
                continue
            
            result = self.scenarios[name]
            df = result.staffing_plan.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            if aggregate == "hour":
                df["group"] = df["timestamp"].dt.hour
                grouped = df.groupby("group")["total_agents"].mean()
                x_vals = [f"{h}:00" for h in grouped.index]
            else:
                df["group"] = df["timestamp"].dt.dayofweek
                grouped = df.groupby("group")["total_agents"].mean()
                x_vals = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=grouped.values,
                name=name,
                mode="lines+markers",
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title=f"Staffing Pattern by {aggregate.title()} - Scenario Comparison",
            xaxis_title="Hour" if aggregate == "hour" else "Day",
            yaxis_title="Average Agents Required",
            height=450,
            showlegend=True
        )
        
        return fig
    
    def sensitivity_analysis(
        self,
        parameter: str,
        values: List[float],
        base_value: float
    ) -> Tuple[go.Figure, pd.DataFrame]:
        """
        Perform sensitivity analysis on a parameter.
        
        Args:
            parameter: 'volume', 'aht', or 'shrinkage'
            values: List of values to test (as percentages for volume/aht, absolute for shrinkage)
            base_value: The base/default value
            
        Returns:
            Tuple of (Plotly figure, DataFrame with results)
        """
        results = []
        
        for val in values:
            if parameter == "volume":
                scenario = Scenario(
                    name=f"Volume {val:+.0f}%",
                    description=f"Volume change: {val:+.0f}%",
                    volume_multiplier=1 + (val / 100)
                )
            elif parameter == "aht":
                scenario = Scenario(
                    name=f"AHT {val:+.0f}%",
                    description=f"AHT change: {val:+.0f}%",
                    aht_multiplier=1 + (val / 100)
                )
            elif parameter == "shrinkage":
                scenario = Scenario(
                    name=f"Shrinkage {val:.0%}",
                    description=f"Shrinkage: {val:.0%}",
                    shrinkage=val
                )
            else:
                raise ValueError(f"Unknown parameter: {parameter}")
            
            result = self._run_scenario(scenario)
            
            results.append({
                "value": val,
                "total_fte": result.total_fte,
                "peak_agents": result.peak_agents,
                "avg_agents": result.avg_agents,
                "cost": result.cost_estimate
            })
        
        df = pd.DataFrame(results)
        
        # Create plot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Impact on Total FTE Hours", "Impact on Cost")
        )
        
        fig.add_trace(
            go.Scatter(
                x=df["value"], y=df["total_fte"],
                mode="lines+markers",
                name="FTE Hours",
                line=dict(color="#667eea", width=2)
            ),
            row=1, col=1
        )
        
        # Mark base value
        base_idx = df["value"].sub(base_value).abs().idxmin()
        fig.add_trace(
            go.Scatter(
                x=[df.loc[base_idx, "value"]],
                y=[df.loc[base_idx, "total_fte"]],
                mode="markers",
                name="Base",
                marker=dict(color="#ef4444", size=12, symbol="star")
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df["value"], y=df["cost"],
                mode="lines+markers",
                name="Cost (€)",
                line=dict(color="#10b981", width=2)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=[df.loc[base_idx, "value"]],
                y=[df.loc[base_idx, "cost"]],
                mode="markers",
                name="Base",
                marker=dict(color="#ef4444", size=12, symbol="star"),
                showlegend=False
            ),
            row=1, col=2
        )
        
        x_title = {
            "volume": "Volume Change (%)",
            "aht": "AHT Change (%)",
            "shrinkage": "Shrinkage Factor"
        }[parameter]
        
        fig.update_layout(
            title=f"Sensitivity Analysis: {parameter.upper()}",
            height=400,
            showlegend=True
        )
        
        fig.update_xaxes(title_text=x_title, row=1, col=1)
        fig.update_xaxes(title_text=x_title, row=1, col=2)
        fig.update_yaxes(title_text="FTE Hours", row=1, col=1)
        fig.update_yaxes(title_text="Cost (€)", row=1, col=2)
        
        return fig, df
    
    def service_level_cost_tradeoff(
        self,
        service_levels: List[float] = None
    ) -> go.Figure:
        """
        Analyze the trade-off between service level and cost.
        
        Args:
            service_levels: List of service levels to test (default: 0.5 to 0.95)
            
        Returns:
            Plotly figure showing the trade-off curve
        """
        if service_levels is None:
            service_levels = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        
        results = []
        
        for sl in service_levels:
            scenario = Scenario(
                name=f"SL {sl:.0%}",
                description=f"Service Level: {sl:.0%}",
                service_level=sl
            )
            
            result = self._run_scenario(scenario)
            
            results.append({
                "service_level": sl * 100,
                "total_fte": result.total_fte,
                "cost": result.cost_estimate,
                "avg_agents": result.avg_agents
            })
        
        df = pd.DataFrame(results)
        
        fig = go.Figure()
        
        # Cost vs Service Level
        fig.add_trace(go.Scatter(
            x=df["service_level"],
            y=df["cost"],
            mode="lines+markers",
            name="Cost (€)",
            line=dict(color="#667eea", width=3),
            marker=dict(size=10)
        ))
        
        # Add annotations for key points
        for _, row in df.iterrows():
            if row["service_level"] in [80, 90, 95]:
                fig.add_annotation(
                    x=row["service_level"],
                    y=row["cost"],
                    text=f"€{row['cost']:,.0f}",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-30
                )
        
        fig.update_layout(
            title="Service Level vs Cost Trade-off",
            xaxis_title="Service Level (%)",
            yaxis_title="Estimated Cost (€)",
            height=450,
            showlegend=False
        )
        
        # Add reference line at 80%
        fig.add_vline(x=80, line_dash="dash", line_color="#ef4444", 
                     annotation_text="Industry Standard (80%)")
        
        return fig
    
    def get_predefined_scenarios(self) -> List[Dict]:
        """Get a list of predefined scenario templates."""
        return [
            {
                "name": "Peak Season (+30%)",
                "description": "30% increase in all volumes (holiday season, promotions)",
                "volume_change_pct": 30,
                "aht_change_pct": 0
            },
            {
                "name": "Low Season (-20%)",
                "description": "20% decrease in volumes (summer, quiet period)",
                "volume_change_pct": -20,
                "aht_change_pct": 0
            },
            {
                "name": "New Product Launch",
                "description": "50% more calls, 20% longer handle time",
                "volume_change_pct": 50,
                "aht_change_pct": 20
            },
            {
                "name": "Efficiency Improvement",
                "description": "10% reduction in handle time through training/tools",
                "volume_change_pct": 0,
                "aht_change_pct": -10
            },
            {
                "name": "Crisis Response",
                "description": "Double volume, 50% longer calls",
                "volume_change_pct": 100,
                "aht_change_pct": 50
            },
            {
                "name": "Email Shift",
                "description": "20% more emails, 20% fewer calls",
                "volume_change_pct": 0,
                "aht_change_pct": 0,
                "custom_adjustments": {"calls": 0.8, "emails": 1.2}
            }
        ]


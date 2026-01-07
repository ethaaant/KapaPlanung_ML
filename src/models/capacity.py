"""
Agent capacity planning using Erlang-C formula.

The Erlang-C formula is the industry standard for call center staffing,
calculating the probability of waiting and required agents to meet service levels.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math
from scipy import special
import logging

from src.utils.config import (
    DEFAULT_CAPACITY_CONFIG,
    CapacityConfig,
    TaskConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StaffingRequirement:
    """Container for staffing calculation results."""
    timestamp: pd.Timestamp
    task_type: str
    forecasted_volume: float
    traffic_intensity: float  # Erlang
    required_agents: int
    occupancy: float
    service_level: float
    avg_wait_time: float


@dataclass
class HourlyStaffingPlan:
    """Staffing plan for a single hour."""
    timestamp: pd.Timestamp
    task_requirements: Dict[str, StaffingRequirement]
    total_agents: int
    total_volume: float


class ErlangC:
    """
    Erlang-C calculator for call center staffing.
    
    The Erlang-C formula calculates:
    - Probability that a call must wait (not immediately answered)
    - Average waiting time
    - Service level (% of calls answered within target time)
    - Required number of agents to meet service level target
    """
    
    @staticmethod
    def factorial(n: int) -> float:
        """Calculate factorial, handling large numbers."""
        if n <= 0:
            return 1
        return math.factorial(n)
    
    @staticmethod
    def erlang_c_probability(agents: int, traffic_intensity: float) -> float:
        """
        Calculate Erlang-C probability (probability of waiting).
        
        Args:
            agents: Number of agents.
            traffic_intensity: Traffic intensity in Erlangs (volume * AHT / interval).
            
        Returns:
            Probability that a call must wait.
        """
        if agents <= 0 or traffic_intensity <= 0:
            return 0.0
        
        if traffic_intensity >= agents:
            # System is overloaded
            return 1.0
        
        # Calculate using logarithms to avoid overflow
        a = traffic_intensity
        n = agents
        
        # Sum term: sum of (a^k / k!) for k=0 to n-1
        sum_term = 0.0
        for k in range(n):
            sum_term += (a ** k) / ErlangC.factorial(k)
        
        # Final term: (a^n / n!) * (n / (n - a))
        final_term = ((a ** n) / ErlangC.factorial(n)) * (n / (n - a))
        
        # Erlang-C probability
        ec = final_term / (sum_term + final_term)
        
        return min(1.0, max(0.0, ec))
    
    @staticmethod
    def average_wait_time(
        agents: int, 
        traffic_intensity: float, 
        avg_handling_time: float
    ) -> float:
        """
        Calculate average wait time for calls that must wait.
        
        Args:
            agents: Number of agents.
            traffic_intensity: Traffic intensity in Erlangs.
            avg_handling_time: Average handling time in seconds.
            
        Returns:
            Average wait time in seconds.
        """
        if agents <= traffic_intensity:
            return float("inf")
        
        ec = ErlangC.erlang_c_probability(agents, traffic_intensity)
        
        # Average wait = EC * AHT / (agents - traffic_intensity)
        avg_wait = (ec * avg_handling_time) / (agents - traffic_intensity)
        
        return max(0.0, avg_wait)
    
    @staticmethod
    def service_level(
        agents: int,
        traffic_intensity: float,
        avg_handling_time: float,
        target_wait_time: float
    ) -> float:
        """
        Calculate service level (% of calls answered within target time).
        
        Args:
            agents: Number of agents.
            traffic_intensity: Traffic intensity in Erlangs.
            avg_handling_time: Average handling time in seconds.
            target_wait_time: Target wait time in seconds.
            
        Returns:
            Service level as a fraction (0-1).
        """
        if agents <= traffic_intensity:
            return 0.0
        
        ec = ErlangC.erlang_c_probability(agents, traffic_intensity)
        
        # SL = 1 - EC * exp(-(agents - intensity) * target / AHT)
        exponent = -1 * (agents - traffic_intensity) * target_wait_time / avg_handling_time
        sl = 1 - ec * math.exp(exponent)
        
        return min(1.0, max(0.0, sl))
    
    @staticmethod
    def required_agents(
        traffic_intensity: float,
        avg_handling_time: float,
        target_service_level: float = 0.80,
        target_wait_time: float = 20.0,
        max_agents: int = 500
    ) -> int:
        """
        Calculate required number of agents to meet service level target.
        
        Args:
            traffic_intensity: Traffic intensity in Erlangs.
            avg_handling_time: Average handling time in seconds.
            target_service_level: Target service level (e.g., 0.80 = 80%).
            target_wait_time: Target wait time in seconds.
            max_agents: Maximum agents to consider.
            
        Returns:
            Required number of agents.
        """
        if traffic_intensity <= 0:
            return 0
        
        # Minimum agents is ceil of traffic intensity
        min_agents = max(1, int(math.ceil(traffic_intensity)))
        
        for agents in range(min_agents, max_agents + 1):
            sl = ErlangC.service_level(
                agents, traffic_intensity, avg_handling_time, target_wait_time
            )
            if sl >= target_service_level:
                return agents
        
        # If we can't meet service level, return max
        logger.warning(
            f"Cannot meet service level {target_service_level} "
            f"with {max_agents} agents for intensity {traffic_intensity}"
        )
        return max_agents


class CapacityPlanner:
    """
    Plan agent capacity based on forecasted workload.
    
    Combines Erlang-C calculations with task-specific handling times
    and shrinkage factors to determine staffing requirements.
    """
    
    def __init__(self, config: CapacityConfig = None):
        """
        Initialize capacity planner.
        
        Args:
            config: Capacity planning configuration.
        """
        self.config = config or DEFAULT_CAPACITY_CONFIG
        self.erlang = ErlangC()
        
    def calculate_traffic_intensity(
        self,
        volume: float,
        avg_handling_time_minutes: float,
        interval_minutes: float = 60.0
    ) -> float:
        """
        Calculate traffic intensity (Erlang).
        
        Traffic intensity = (volume * AHT) / interval
        
        Args:
            volume: Number of tasks in the interval.
            avg_handling_time_minutes: Average handling time in minutes.
            interval_minutes: Length of interval in minutes.
            
        Returns:
            Traffic intensity in Erlangs.
        """
        if volume <= 0 or avg_handling_time_minutes <= 0:
            return 0.0
        
        # Convert to same time units
        return (volume * avg_handling_time_minutes) / interval_minutes
    
    def calculate_staffing(
        self,
        volume: float,
        task_config: TaskConfig,
        interval_minutes: float = 60.0
    ) -> StaffingRequirement:
        """
        Calculate staffing requirement for a single task type.
        
        Args:
            volume: Forecasted volume.
            task_config: Configuration for the task type.
            interval_minutes: Interval length in minutes.
            
        Returns:
            StaffingRequirement with calculated values.
        """
        # Calculate traffic intensity
        intensity = self.calculate_traffic_intensity(
            volume,
            task_config.avg_handling_time_minutes,
            interval_minutes
        )
        
        # Adjust for concurrency (e.g., emails can be handled in parallel)
        effective_intensity = intensity / task_config.concurrency
        
        # Convert AHT to seconds for Erlang calculations
        aht_seconds = task_config.avg_handling_time_minutes * 60
        target_wait = self.config.service_level_time_seconds
        target_sl = self.config.service_level_target
        
        # Calculate required agents
        if effective_intensity > 0:
            raw_agents = self.erlang.required_agents(
                effective_intensity,
                aht_seconds,
                target_sl,
                target_wait
            )
            
            # Apply shrinkage factor
            required_agents = int(math.ceil(
                raw_agents / (1 - self.config.shrinkage_factor)
            ))
            
            # Calculate achieved metrics
            occupancy = effective_intensity / raw_agents if raw_agents > 0 else 0
            service_level = self.erlang.service_level(
                raw_agents, effective_intensity, aht_seconds, target_wait
            )
            avg_wait = self.erlang.average_wait_time(
                raw_agents, effective_intensity, aht_seconds
            )
        else:
            required_agents = 0
            occupancy = 0.0
            service_level = 1.0
            avg_wait = 0.0
        
        return StaffingRequirement(
            timestamp=None,  # Set by caller
            task_type=task_config.name,
            forecasted_volume=volume,
            traffic_intensity=effective_intensity,
            required_agents=required_agents,
            occupancy=occupancy,
            service_level=service_level,
            avg_wait_time=avg_wait
        )
    
    def calculate_hourly_plan(
        self,
        timestamp: pd.Timestamp,
        volumes: Dict[str, float]
    ) -> HourlyStaffingPlan:
        """
        Calculate staffing plan for a single hour.
        
        Args:
            timestamp: Hour timestamp.
            volumes: Dictionary of task type to volume.
            
        Returns:
            HourlyStaffingPlan with all task requirements.
        """
        task_requirements = {}
        total_agents = 0
        total_volume = 0
        
        for task_key, task_config in self.config.tasks.items():
            volume = volumes.get(task_key, 0)
            
            if volume > 0:
                req = self.calculate_staffing(volume, task_config)
                req.timestamp = timestamp
                task_requirements[task_key] = req
                total_agents += req.required_agents
                total_volume += volume
        
        return HourlyStaffingPlan(
            timestamp=timestamp,
            task_requirements=task_requirements,
            total_agents=total_agents,
            total_volume=total_volume
        )
    
    def create_staffing_plan(
        self,
        forecast_df: pd.DataFrame,
        volume_mapping: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        Create a complete staffing plan from forecasts.
        
        Args:
            forecast_df: DataFrame with timestamp and forecasted volumes.
            volume_mapping: Mapping of forecast column names to task keys.
            
        Returns:
            DataFrame with staffing requirements by hour.
        """
        # Default mapping based on common column names
        if volume_mapping is None:
            volume_mapping = {
                "call_volume": "calls",
                "email_count": "emails",
                "outbound_ook": "outbound_ook",
                "outbound_omk": "outbound_omk",
                "outbound_nb": "outbound_nb",
                "outbound_total": None,  # Skip total, use components
            }
        
        results = []
        
        for _, row in forecast_df.iterrows():
            timestamp = row["timestamp"]
            
            # Check if within working hours
            hour = timestamp.hour
            if not (self.config.working_hours_start <= hour < self.config.working_hours_end):
                continue
            
            # Gather volumes for each task
            volumes = {}
            for col, task_key in volume_mapping.items():
                if task_key and col in row.index:
                    volumes[task_key] = row[col]
            
            # Calculate hourly plan
            plan = self.calculate_hourly_plan(timestamp, volumes)
            
            # Flatten to row
            row_data = {
                "timestamp": timestamp,
                "hour": hour,
                "day_of_week": timestamp.dayofweek,
                "date": timestamp.date(),
                "total_volume": plan.total_volume,
                "total_agents": plan.total_agents,
            }
            
            # Add per-task details
            for task_key, req in plan.task_requirements.items():
                row_data[f"{task_key}_volume"] = req.forecasted_volume
                row_data[f"{task_key}_agents"] = req.required_agents
                row_data[f"{task_key}_occupancy"] = req.occupancy
                row_data[f"{task_key}_service_level"] = req.service_level
            
            results.append(row_data)
        
        result_df = pd.DataFrame(results)
        
        if len(result_df) > 0:
            result_df = result_df.sort_values("timestamp").reset_index(drop=True)
        
        return result_df
    
    def aggregate_daily(self, staffing_plan: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate staffing plan to daily level.
        
        Args:
            staffing_plan: Hourly staffing plan.
            
        Returns:
            Daily aggregated plan.
        """
        if len(staffing_plan) == 0:
            return pd.DataFrame()
        
        daily = staffing_plan.groupby("date").agg({
            "total_volume": "sum",
            "total_agents": "max",  # Peak agents needed
        }).reset_index()
        
        # Add average agents
        daily_avg = staffing_plan.groupby("date")["total_agents"].mean().reset_index()
        daily_avg.columns = ["date", "avg_agents"]
        daily = daily.merge(daily_avg, on="date")
        
        # Add per-task summaries
        volume_cols = [c for c in staffing_plan.columns if c.endswith("_volume") and c != "total_volume"]
        agent_cols = [c for c in staffing_plan.columns if c.endswith("_agents") and c != "total_agents"]
        
        for col in volume_cols:
            task_daily = staffing_plan.groupby("date")[col].sum().reset_index()
            daily = daily.merge(task_daily, on="date", how="left")
        
        for col in agent_cols:
            task_daily = staffing_plan.groupby("date")[col].max().reset_index()
            task_daily.columns = ["date", f"{col}_peak"]
            daily = daily.merge(task_daily, on="date", how="left")
        
        return daily
    
    def create_shift_plan(
        self,
        staffing_plan: pd.DataFrame,
        shift_length_hours: int = 8
    ) -> pd.DataFrame:
        """
        Create shift-based staffing plan.
        
        Groups hourly requirements into shifts and calculates
        headcount per shift.
        
        Args:
            staffing_plan: Hourly staffing plan.
            shift_length_hours: Length of shifts in hours.
            
        Returns:
            Shift-based staffing plan.
        """
        if len(staffing_plan) == 0:
            return pd.DataFrame()
        
        df = staffing_plan.copy()
        
        # Define shift based on hour
        def get_shift(hour):
            start = self.config.working_hours_start
            shift_num = (hour - start) // shift_length_hours
            shift_start = start + shift_num * shift_length_hours
            shift_end = min(shift_start + shift_length_hours, self.config.working_hours_end)
            return f"{shift_start:02d}:00-{shift_end:02d}:00"
        
        df["shift"] = df["hour"].apply(get_shift)
        
        # Aggregate by date and shift
        shift_plan = df.groupby(["date", "shift"]).agg({
            "total_volume": "sum",
            "total_agents": "max",  # Peak within shift
        }).reset_index()
        
        shift_plan = shift_plan.rename(columns={
            "total_agents": "peak_agents"
        })
        
        # Calculate average agents per shift
        shift_avg = df.groupby(["date", "shift"])["total_agents"].mean().reset_index()
        shift_avg.columns = ["date", "shift", "avg_agents"]
        shift_plan = shift_plan.merge(shift_avg, on=["date", "shift"])
        
        # Round up average to get recommended headcount
        shift_plan["recommended_headcount"] = np.ceil(
            (shift_plan["peak_agents"] + shift_plan["avg_agents"]) / 2
        ).astype(int)
        
        return shift_plan


def demonstrate_erlang_c():
    """Demonstrate Erlang-C calculations."""
    erlang = ErlangC()
    
    print("Erlang-C Demonstration")
    print("=" * 50)
    
    # Example: 100 calls/hour, 5 min AHT
    volume = 100
    aht_minutes = 5
    interval_minutes = 60
    
    # Traffic intensity
    intensity = (volume * aht_minutes) / interval_minutes
    print(f"\nScenario: {volume} calls/hour, {aht_minutes} min AHT")
    print(f"Traffic Intensity: {intensity:.2f} Erlangs")
    
    # Calculate for different agent counts
    print(f"\n{'Agents':<10}{'Wait Prob':<15}{'Avg Wait':<15}{'SL (80/20)':<15}")
    print("-" * 55)
    
    for agents in range(int(intensity), int(intensity) + 10):
        wait_prob = erlang.erlang_c_probability(agents, intensity)
        avg_wait = erlang.average_wait_time(agents, intensity, aht_minutes * 60)
        sl = erlang.service_level(agents, intensity, aht_minutes * 60, 20)
        
        print(f"{agents:<10}{wait_prob:<15.2%}{avg_wait:<15.1f}s{sl:<15.2%}")
    
    # Find required agents for 80/20 service level
    required = erlang.required_agents(intensity, aht_minutes * 60, 0.80, 20)
    print(f"\nRequired agents for 80/20 SL: {required}")


if __name__ == "__main__":
    # Demonstrate Erlang-C
    demonstrate_erlang_c()
    
    # Test capacity planner
    print("\n\nCapacity Planner Test")
    print("=" * 50)
    
    planner = CapacityPlanner()
    
    # Create sample forecast
    import pandas as pd
    from datetime import datetime, timedelta
    
    start = datetime(2024, 6, 1, 8, 0)
    hours = 12 * 5  # 5 days of working hours
    
    forecast_data = []
    for i in range(hours):
        ts = start + timedelta(hours=i)
        if ts.hour < 8 or ts.hour >= 20:
            continue
            
        forecast_data.append({
            "timestamp": ts,
            "call_volume": 50 + np.random.randint(-10, 20),
            "email_count": 30 + np.random.randint(-5, 10),
            "outbound_ook": 15 + np.random.randint(-3, 5),
            "outbound_omk": 10 + np.random.randint(-2, 5),
            "outbound_nb": 20 + np.random.randint(-5, 8),
        })
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Create staffing plan
    staffing_plan = planner.create_staffing_plan(forecast_df)
    
    print(f"\nStaffing Plan:")
    print(staffing_plan[["timestamp", "total_volume", "total_agents"]].head(10))
    
    # Daily summary
    daily = planner.aggregate_daily(staffing_plan)
    print(f"\nDaily Summary:")
    print(daily[["date", "total_volume", "total_agents", "avg_agents"]])
    
    # Shift plan
    shift_plan = planner.create_shift_plan(staffing_plan)
    print(f"\nShift Plan:")
    print(shift_plan.head(15))


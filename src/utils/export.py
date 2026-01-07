"""
Excel export functionality for forecasts and staffing plans.

Generates professional Excel reports with formatting, charts, and multiple sheets.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import io
import logging

try:
    import xlsxwriter
    XLSXWRITER_AVAILABLE = True
except ImportError:
    XLSXWRITER_AVAILABLE = False

from src.utils.config import OUTPUTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkforceReportExporter:
    """
    Export workforce planning results to Excel with professional formatting.
    
    Creates multi-sheet reports with:
    - Forecast data
    - Hourly staffing requirements
    - Daily summary
    - Weekly summary
    - Charts and visualizations
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize exporter.
        
        Args:
            output_dir: Directory for output files.
        """
        self.output_dir = output_dir or OUTPUTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _add_header_format(self, workbook) -> dict:
        """Create header format for Excel."""
        return workbook.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": "#4472C4",
            "border": 1,
            "align": "center",
            "valign": "vcenter"
        })
    
    def _add_number_format(self, workbook) -> dict:
        """Create number format for Excel."""
        return workbook.add_format({
            "num_format": "#,##0",
            "border": 1,
            "align": "center"
        })
    
    def _add_decimal_format(self, workbook) -> dict:
        """Create decimal format for Excel."""
        return workbook.add_format({
            "num_format": "#,##0.0",
            "border": 1,
            "align": "center"
        })
    
    def _add_percent_format(self, workbook) -> dict:
        """Create percentage format for Excel."""
        return workbook.add_format({
            "num_format": "0.0%",
            "border": 1,
            "align": "center"
        })
    
    def _add_date_format(self, workbook) -> dict:
        """Create date format for Excel."""
        return workbook.add_format({
            "num_format": "yyyy-mm-dd",
            "border": 1,
            "align": "center"
        })
    
    def _add_datetime_format(self, workbook) -> dict:
        """Create datetime format for Excel."""
        return workbook.add_format({
            "num_format": "yyyy-mm-dd hh:mm",
            "border": 1,
            "align": "center"
        })
    
    def export_forecast(
        self,
        forecast_df: pd.DataFrame,
        filename: str = None
    ) -> Path:
        """
        Export forecast data to Excel.
        
        Args:
            forecast_df: DataFrame with forecast data.
            filename: Output filename.
            
        Returns:
            Path to the exported file.
        """
        if filename is None:
            filename = f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        filepath = self.output_dir / filename
        
        with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
            workbook = writer.book
            
            # Formats
            header_fmt = self._add_header_format(workbook)
            number_fmt = self._add_number_format(workbook)
            datetime_fmt = self._add_datetime_format(workbook)
            
            # Write forecast data
            forecast_df.to_excel(writer, sheet_name="Forecast", index=False)
            
            worksheet = writer.sheets["Forecast"]
            
            # Format headers
            for col_num, value in enumerate(forecast_df.columns):
                worksheet.write(0, col_num, value, header_fmt)
            
            # Set column widths
            worksheet.set_column("A:A", 18)  # Timestamp
            worksheet.set_column("B:Z", 12)  # Other columns
            
            # Add chart
            chart = workbook.add_chart({"type": "line"})
            
            # Find numeric columns for chart
            numeric_cols = [c for c in forecast_df.columns if c != "timestamp"]
            
            for i, col in enumerate(numeric_cols[:5]):  # Limit to 5 series
                col_idx = forecast_df.columns.get_loc(col)
                chart.add_series({
                    "name": col,
                    "categories": f"='Forecast'!$A$2:$A${len(forecast_df)+1}",
                    "values": f"='Forecast'!${chr(65+col_idx)}$2:${chr(65+col_idx)}${len(forecast_df)+1}",
                })
            
            chart.set_title({"name": "Volume Forecast"})
            chart.set_x_axis({"name": "Date/Time"})
            chart.set_y_axis({"name": "Volume"})
            chart.set_size({"width": 720, "height": 400})
            
            worksheet.insert_chart("H2", chart)
        
        logger.info(f"Exported forecast to: {filepath}")
        return filepath
    
    def export_staffing_plan(
        self,
        staffing_plan: pd.DataFrame,
        filename: str = None
    ) -> Path:
        """
        Export staffing plan to Excel.
        
        Args:
            staffing_plan: DataFrame with staffing requirements.
            filename: Output filename.
            
        Returns:
            Path to the exported file.
        """
        if filename is None:
            filename = f"staffing_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        filepath = self.output_dir / filename
        
        with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
            workbook = writer.book
            
            # Formats
            header_fmt = self._add_header_format(workbook)
            number_fmt = self._add_number_format(workbook)
            
            # Sheet 1: Hourly Plan
            staffing_plan.to_excel(writer, sheet_name="Hourly Plan", index=False)
            
            ws_hourly = writer.sheets["Hourly Plan"]
            for col_num, value in enumerate(staffing_plan.columns):
                ws_hourly.write(0, col_num, value, header_fmt)
            ws_hourly.set_column("A:A", 18)
            ws_hourly.set_column("B:Z", 12)
            
            # Sheet 2: Daily Summary
            if "date" in staffing_plan.columns:
                daily = staffing_plan.groupby("date").agg({
                    "total_volume": "sum",
                    "total_agents": ["max", "mean"]
                }).reset_index()
                daily.columns = ["Date", "Total Volume", "Peak Agents", "Avg Agents"]
                daily["Avg Agents"] = daily["Avg Agents"].round(1)
                
                daily.to_excel(writer, sheet_name="Daily Summary", index=False)
                
                ws_daily = writer.sheets["Daily Summary"]
                for col_num, value in enumerate(daily.columns):
                    ws_daily.write(0, col_num, value, header_fmt)
                ws_daily.set_column("A:A", 12)
                ws_daily.set_column("B:D", 15)
                
                # Add daily chart
                chart = workbook.add_chart({"type": "column"})
                chart.add_series({
                    "name": "Peak Agents",
                    "categories": f"='Daily Summary'!$A$2:$A${len(daily)+1}",
                    "values": f"='Daily Summary'!$C$2:$C${len(daily)+1}",
                })
                chart.set_title({"name": "Daily Peak Agent Requirements"})
                chart.set_x_axis({"name": "Date"})
                chart.set_y_axis({"name": "Agents"})
                ws_daily.insert_chart("F2", chart)
            
            # Sheet 3: Weekly Summary
            if "timestamp" in staffing_plan.columns:
                staffing_plan_copy = staffing_plan.copy()
                staffing_plan_copy["week"] = pd.to_datetime(
                    staffing_plan_copy["timestamp"]
                ).dt.isocalendar().week
                
                weekly = staffing_plan_copy.groupby("week").agg({
                    "total_volume": "sum",
                    "total_agents": ["max", "mean"]
                }).reset_index()
                weekly.columns = ["Week", "Total Volume", "Peak Agents", "Avg Agents"]
                weekly["Avg Agents"] = weekly["Avg Agents"].round(1)
                
                weekly.to_excel(writer, sheet_name="Weekly Summary", index=False)
                
                ws_weekly = writer.sheets["Weekly Summary"]
                for col_num, value in enumerate(weekly.columns):
                    ws_weekly.write(0, col_num, value, header_fmt)
        
        logger.info(f"Exported staffing plan to: {filepath}")
        return filepath
    
    def export_complete_report(
        self,
        forecast_df: pd.DataFrame,
        staffing_plan: pd.DataFrame,
        historical_data: pd.DataFrame = None,
        metrics: Dict = None,
        filename: str = None
    ) -> Path:
        """
        Export complete workforce planning report.
        
        Args:
            forecast_df: Forecast data.
            staffing_plan: Staffing requirements.
            historical_data: Optional historical data for comparison.
            metrics: Optional model metrics.
            filename: Output filename.
            
        Returns:
            Path to the exported file.
        """
        if filename is None:
            filename = f"workforce_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        filepath = self.output_dir / filename
        
        with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
            workbook = writer.book
            
            # Formats
            header_fmt = self._add_header_format(workbook)
            title_fmt = workbook.add_format({
                "bold": True,
                "font_size": 14,
                "font_color": "#4472C4"
            })
            
            # Sheet 1: Summary/Dashboard
            summary_ws = workbook.add_worksheet("Summary")
            
            summary_ws.write("A1", "Workforce Planning Report", title_fmt)
            summary_ws.write("A2", f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            
            summary_ws.write("A4", "Report Contents:")
            summary_ws.write("A5", "• Forecast - Predicted volumes by hour")
            summary_ws.write("A6", "• Hourly Plan - Required agents by hour")
            summary_ws.write("A7", "• Daily Summary - Daily aggregates")
            summary_ws.write("A8", "• Weekly Summary - Weekly aggregates")
            
            if metrics:
                summary_ws.write("A10", "Model Performance:", title_fmt)
                row = 11
                for target, m in metrics.items():
                    summary_ws.write(row, 0, f"{target}:")
                    summary_ws.write(row, 1, f"RMSE: {m.get('rmse', 'N/A'):.2f}")
                    summary_ws.write(row, 2, f"MAE: {m.get('mae', 'N/A'):.2f}")
                    summary_ws.write(row, 3, f"R²: {m.get('r2', 'N/A'):.3f}")
                    row += 1
            
            summary_ws.set_column("A:A", 30)
            summary_ws.set_column("B:D", 15)
            
            # Sheet 2: Forecast
            forecast_df.to_excel(writer, sheet_name="Forecast", index=False)
            
            ws_forecast = writer.sheets["Forecast"]
            for col_num, value in enumerate(forecast_df.columns):
                ws_forecast.write(0, col_num, value, header_fmt)
            ws_forecast.set_column("A:A", 18)
            
            # Add forecast chart
            chart1 = workbook.add_chart({"type": "line"})
            numeric_cols = [c for c in forecast_df.columns if c != "timestamp"]
            
            for i, col in enumerate(numeric_cols[:3]):
                col_idx = forecast_df.columns.get_loc(col)
                chart1.add_series({
                    "name": col,
                    "categories": f"='Forecast'!$A$2:$A${len(forecast_df)+1}",
                    "values": f"='Forecast'!${chr(65+col_idx)}$2:${chr(65+col_idx)}${len(forecast_df)+1}",
                })
            
            chart1.set_title({"name": "Volume Forecast"})
            chart1.set_size({"width": 720, "height": 400})
            ws_forecast.insert_chart("H2", chart1)
            
            # Sheet 3: Hourly Plan
            staffing_plan.to_excel(writer, sheet_name="Hourly Plan", index=False)
            
            ws_hourly = writer.sheets["Hourly Plan"]
            for col_num, value in enumerate(staffing_plan.columns):
                ws_hourly.write(0, col_num, value, header_fmt)
            
            # Sheet 4: Daily Summary
            if "date" in staffing_plan.columns:
                daily = staffing_plan.groupby("date").agg({
                    "total_volume": "sum",
                    "total_agents": ["max", "mean"]
                }).reset_index()
                daily.columns = ["Date", "Total Volume", "Peak Agents", "Avg Agents"]
                daily["Avg Agents"] = daily["Avg Agents"].round(1)
                
                daily.to_excel(writer, sheet_name="Daily Summary", index=False)
                
                ws_daily = writer.sheets["Daily Summary"]
                for col_num, value in enumerate(daily.columns):
                    ws_daily.write(0, col_num, value, header_fmt)
                
                # Staffing chart
                chart2 = workbook.add_chart({"type": "column"})
                chart2.add_series({
                    "name": "Peak Agents",
                    "categories": f"='Daily Summary'!$A$2:$A${len(daily)+1}",
                    "values": f"='Daily Summary'!$C$2:$C${len(daily)+1}",
                    "fill": {"color": "#4472C4"}
                })
                chart2.add_series({
                    "name": "Avg Agents",
                    "categories": f"='Daily Summary'!$A$2:$A${len(daily)+1}",
                    "values": f"='Daily Summary'!$D$2:$D${len(daily)+1}",
                    "fill": {"color": "#70AD47"}
                })
                chart2.set_title({"name": "Daily Agent Requirements"})
                chart2.set_size({"width": 600, "height": 350})
                ws_daily.insert_chart("F2", chart2)
            
            # Sheet 5: Weekly Summary
            if "timestamp" in staffing_plan.columns:
                sp_copy = staffing_plan.copy()
                sp_copy["week"] = pd.to_datetime(sp_copy["timestamp"]).dt.isocalendar().week
                
                weekly = sp_copy.groupby("week").agg({
                    "total_volume": "sum",
                    "total_agents": ["max", "mean"]
                }).reset_index()
                weekly.columns = ["Week", "Total Volume", "Peak Agents", "Avg Agents"]
                weekly["Avg Agents"] = weekly["Avg Agents"].round(1)
                
                weekly.to_excel(writer, sheet_name="Weekly Summary", index=False)
                
                ws_weekly = writer.sheets["Weekly Summary"]
                for col_num, value in enumerate(weekly.columns):
                    ws_weekly.write(0, col_num, value, header_fmt)
            
            # Sheet 6: Historical (if provided)
            if historical_data is not None and len(historical_data) > 0:
                # Resample to daily
                hist_daily = historical_data.set_index("timestamp").resample("D").sum().reset_index()
                hist_daily.to_excel(writer, sheet_name="Historical Data", index=False)
                
                ws_hist = writer.sheets["Historical Data"]
                for col_num, value in enumerate(hist_daily.columns):
                    ws_hist.write(0, col_num, value, header_fmt)
        
        logger.info(f"Exported complete report to: {filepath}")
        return filepath
    
    def export_to_buffer(
        self,
        forecast_df: pd.DataFrame,
        staffing_plan: pd.DataFrame,
        metrics: Dict = None
    ) -> io.BytesIO:
        """
        Export report to a BytesIO buffer (for web download).
        
        Args:
            forecast_df: Forecast data.
            staffing_plan: Staffing requirements.
            metrics: Optional model metrics.
            
        Returns:
            BytesIO buffer with Excel file.
        """
        buffer = io.BytesIO()
        
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            workbook = writer.book
            header_fmt = self._add_header_format(workbook)
            
            # Forecast
            forecast_df.to_excel(writer, sheet_name="Forecast", index=False)
            ws_forecast = writer.sheets["Forecast"]
            for col_num, value in enumerate(forecast_df.columns):
                ws_forecast.write(0, col_num, value, header_fmt)
            
            # Staffing Plan
            staffing_plan.to_excel(writer, sheet_name="Hourly Plan", index=False)
            ws_hourly = writer.sheets["Hourly Plan"]
            for col_num, value in enumerate(staffing_plan.columns):
                ws_hourly.write(0, col_num, value, header_fmt)
            
            # Daily Summary
            if "date" in staffing_plan.columns:
                daily = staffing_plan.groupby("date").agg({
                    "total_volume": "sum",
                    "total_agents": ["max", "mean"]
                }).reset_index()
                daily.columns = ["Date", "Total Volume", "Peak Agents", "Avg Agents"]
                daily["Avg Agents"] = daily["Avg Agents"].round(1)
                
                daily.to_excel(writer, sheet_name="Daily Summary", index=False)
        
        buffer.seek(0)
        return buffer


def export_forecast_csv(
    forecast_df: pd.DataFrame,
    output_dir: Path = None,
    filename: str = None
) -> Path:
    """
    Simple CSV export for forecast data.
    
    Args:
        forecast_df: Forecast DataFrame.
        output_dir: Output directory.
        filename: Output filename.
        
    Returns:
        Path to exported file.
    """
    output_dir = output_dir or OUTPUTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    filepath = output_dir / filename
    forecast_df.to_csv(filepath, index=False)
    
    logger.info(f"Exported forecast CSV to: {filepath}")
    return filepath


def export_staffing_csv(
    staffing_plan: pd.DataFrame,
    output_dir: Path = None,
    filename: str = None
) -> Path:
    """
    Simple CSV export for staffing plan.
    
    Args:
        staffing_plan: Staffing plan DataFrame.
        output_dir: Output directory.
        filename: Output filename.
        
    Returns:
        Path to exported file.
    """
    output_dir = output_dir or OUTPUTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"staffing_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    filepath = output_dir / filename
    staffing_plan.to_csv(filepath, index=False)
    
    logger.info(f"Exported staffing CSV to: {filepath}")
    return filepath


if __name__ == "__main__":
    # Test export functionality
    import numpy as np
    from datetime import timedelta
    
    # Create sample forecast
    start = datetime(2024, 6, 1, 8, 0)
    hours = 168  # 1 week
    
    forecast_data = []
    for i in range(hours):
        ts = start + timedelta(hours=i)
        forecast_data.append({
            "timestamp": ts,
            "call_volume": 50 + np.random.randint(-10, 20),
            "email_count": 30 + np.random.randint(-5, 10),
            "outbound_total": 40 + np.random.randint(-8, 15),
        })
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Create sample staffing plan
    staffing_data = []
    for i in range(hours):
        ts = start + timedelta(hours=i)
        if ts.hour >= 8 and ts.hour < 20:
            staffing_data.append({
                "timestamp": ts,
                "hour": ts.hour,
                "date": ts.date(),
                "total_volume": forecast_data[i]["call_volume"] + forecast_data[i]["email_count"],
                "total_agents": np.random.randint(10, 25),
                "calls_agents": np.random.randint(5, 12),
                "emails_agents": np.random.randint(3, 8),
            })
    
    staffing_df = pd.DataFrame(staffing_data)
    
    # Export
    exporter = WorkforceReportExporter()
    
    # Export individual files
    exporter.export_forecast(forecast_df)
    exporter.export_staffing_plan(staffing_df)
    
    # Export complete report
    metrics = {
        "call_volume": {"rmse": 8.5, "mae": 6.2, "r2": 0.87},
        "email_count": {"rmse": 5.3, "mae": 4.1, "r2": 0.82},
    }
    exporter.export_complete_report(forecast_df, staffing_df, metrics=metrics)
    
    print("Export tests completed!")


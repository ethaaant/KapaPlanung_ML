"""
Reusable UI components for Streamlit.
Provides consistent styling and improved UX across the application.
"""
import streamlit as st
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time


class AlertType(Enum):
    """Types of alerts for the UI."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class TooltipConfig:
    """Configuration for tooltips."""
    text: str
    position: str = "top"  # top, bottom, left, right
    icon: str = "ℹ️"


# ===========================================
# ALERT & NOTIFICATION COMPONENTS
# ===========================================

def show_alert(
    message: str,
    alert_type: AlertType = AlertType.INFO,
    icon: str = None,
    dismissible: bool = False
):
    """
    Show a styled alert message.
    
    Args:
        message: The message to display
        alert_type: Type of alert (success, error, warning, info)
        icon: Optional custom icon
        dismissible: Whether the alert can be dismissed
    """
    icons = {
        AlertType.SUCCESS: "✅",
        AlertType.ERROR: "❌",
        AlertType.WARNING: "⚠️",
        AlertType.INFO: "ℹ️"
    }
    
    colors = {
        AlertType.SUCCESS: ("#ecfdf5", "#059669", "#d1fae5"),
        AlertType.ERROR: ("#fef2f2", "#dc2626", "#fecaca"),
        AlertType.WARNING: ("#fffbeb", "#d97706", "#fde68a"),
        AlertType.INFO: ("#eff6ff", "#2563eb", "#dbeafe")
    }
    
    bg, text_color, border = colors[alert_type]
    display_icon = icon or icons[alert_type]
    
    st.markdown(f"""
    <div style="
        background: {bg};
        border-left: 4px solid {text_color};
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    ">
        <span style="font-size: 1.25rem;">{display_icon}</span>
        <span style="color: {text_color}; font-weight: 500;">{message}</span>
    </div>
    """, unsafe_allow_html=True)


def show_toast(message: str, duration: int = 3):
    """
    Show a temporary toast notification.
    
    Args:
        message: The message to display
        duration: Seconds to show the toast
    """
    # Using Streamlit's native toast
    st.toast(message, icon="✨")


# ===========================================
# PROGRESS INDICATORS
# ===========================================

class ProgressTracker:
    """
    A progress tracker with status updates.
    
    Usage:
        tracker = ProgressTracker(total_steps=5)
        tracker.start("Loading data...")
        tracker.update(1, "Processing...")
        tracker.complete("Done!")
    """
    
    def __init__(self, total_steps: int, title: str = "Progress"):
        self.total_steps = total_steps
        self.title = title
        self.current_step = 0
        self.progress_bar = None
        self.status_text = None
        self.container = None
    
    def start(self, message: str = "Starting..."):
        """Start the progress tracker."""
        self.container = st.container()
        with self.container:
            st.markdown(f"**{self.title}**")
            self.progress_bar = st.progress(0)
            self.status_text = st.empty()
            self.status_text.markdown(f"🔄 {message}")
    
    def update(self, step: int = None, message: str = None):
        """Update progress."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        progress = min(self.current_step / self.total_steps, 1.0)
        
        if self.progress_bar:
            self.progress_bar.progress(progress)
        
        if message and self.status_text:
            self.status_text.markdown(f"🔄 {message}")
    
    def complete(self, message: str = "Complete!"):
        """Mark as complete."""
        if self.progress_bar:
            self.progress_bar.progress(1.0)
        if self.status_text:
            self.status_text.markdown(f"✅ {message}")


def with_progress(message: str = "Processing..."):
    """
    Decorator to show a spinner during function execution.
    
    Usage:
        @with_progress("Training model...")
        def train_model():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with st.spinner(message):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# ===========================================
# TOOLTIPS & HELP
# ===========================================

def info_tooltip(text: str, label: str = None):
    """
    Display an inline info tooltip.
    
    Args:
        text: Tooltip text
        label: Optional label before the tooltip
    """
    if label:
        st.markdown(f"{label} ℹ️", help=text)
    else:
        st.markdown("ℹ️", help=text)


def help_section(title: str, content: str, expanded: bool = False):
    """
    Create an expandable help section.
    
    Args:
        title: Section title
        content: Help content (supports markdown)
        expanded: Whether to expand by default
    """
    with st.expander(f"❓ {title}", expanded=expanded):
        st.markdown(content)


# Field help texts for common fields
FIELD_HELP = {
    "service_level": "Target percentage of calls answered within the service time. Industry standard is typically 80%.",
    "service_time": "Maximum acceptable wait time in seconds for calls. Common values: 20s, 30s, 60s.",
    "shrinkage": "Accounts for agent unavailability (breaks, training, sick leave). Typical range: 25-35%.",
    "avg_handle_time": "Average time to handle one interaction (talk time + after-call work).",
    "forecast_horizon": "Number of days to forecast ahead. Longer horizons have higher uncertainty.",
    "confidence_level": "Statistical confidence for prediction intervals. 95% means we're 95% confident the actual value falls within the range."
}


def get_field_help(field: str) -> str:
    """Get help text for a field."""
    return FIELD_HELP.get(field, "")


# ===========================================
# METRIC CARDS
# ===========================================

def metric_card(
    title: str,
    value: str,
    subtitle: str = None,
    change: float = None,
    icon: str = None
):
    """
    Display a styled metric card.
    
    Args:
        title: Metric title
        value: Main value to display
        subtitle: Optional subtitle
        change: Optional percentage change (positive = green, negative = red)
        icon: Optional icon
    """
    change_html = ""
    if change is not None:
        color = "#059669" if change >= 0 else "#dc2626"
        arrow = "↑" if change >= 0 else "↓"
        change_html = f"""
        <div style="color: {color}; font-size: 0.875rem; font-weight: 500;">
            {arrow} {abs(change):.1f}%
        </div>
        """
    
    icon_html = f'<span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>' if icon else ""
    
    st.markdown(f"""
    <div style="
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem;
        height: 100%;
    ">
        <div style="color: #6b7280; font-size: 0.875rem; margin-bottom: 0.5rem;">
            {icon_html}{title}
        </div>
        <div style="font-size: 1.75rem; font-weight: 600; color: #1f2937;">
            {value}
        </div>
        {f'<div style="color: #9ca3af; font-size: 0.75rem; margin-top: 0.25rem;">{subtitle}</div>' if subtitle else ''}
        {change_html}
    </div>
    """, unsafe_allow_html=True)


def metric_row(metrics: List[Dict[str, Any]]):
    """
    Display a row of metric cards.
    
    Args:
        metrics: List of dicts with keys: title, value, subtitle, change, icon
    """
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            metric_card(**metric)


# ===========================================
# ERROR HANDLING UI
# ===========================================

def show_error(
    title: str,
    message: str,
    error_code: str = None,
    details: str = None,
    show_support: bool = True
):
    """
    Display a formatted error message.
    
    Args:
        title: Error title
        message: User-friendly error message
        error_code: Optional error code for support
        details: Optional technical details
        show_support: Whether to show support info
    """
    st.error(f"**{title}**")
    st.markdown(message)
    
    if error_code:
        st.caption(f"Error code: `{error_code}`")
    
    if details:
        with st.expander("Technical Details"):
            st.code(details)
    
    if show_support:
        st.info("💡 If this problem persists, please contact support.")


def handle_exception(e: Exception, context: str = ""):
    """
    Handle an exception and display user-friendly error.
    
    Args:
        e: The exception
        context: Optional context about what was happening
    """
    from src.utils.exceptions import WorkforcePlanningError
    
    if isinstance(e, WorkforcePlanningError):
        show_error(
            title="An error occurred",
            message=e.user_message,
            error_code=e.error_code,
            details=str(e.details) if e.details else None
        )
    else:
        show_error(
            title="Unexpected Error",
            message=f"An unexpected error occurred{f' while {context}' if context else ''}.",
            details=str(e)
        )


# ===========================================
# LOADING STATES
# ===========================================

def loading_placeholder(message: str = "Loading..."):
    """
    Display a loading placeholder.
    
    Returns a placeholder that can be replaced with content.
    """
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            color: #6b7280;
        ">
            <div style="
                width: 24px;
                height: 24px;
                border: 3px solid #e5e7eb;
                border-top-color: #3b82f6;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 0.75rem;
            "></div>
            {message}
        </div>
        <style>
            @keyframes spin {{
                to {{ transform: rotate(360deg); }}
            }}
        </style>
        """, unsafe_allow_html=True)
    return placeholder


def skeleton_loader(height: int = 100, count: int = 1):
    """
    Display skeleton loading placeholders.
    
    Args:
        height: Height of each skeleton in pixels
        count: Number of skeletons to show
    """
    for _ in range(count):
        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
            height: {height}px;
            border-radius: 8px;
            margin-bottom: 0.5rem;
        "></div>
        <style>
            @keyframes shimmer {{
                0% {{ background-position: 200% 0; }}
                100% {{ background-position: -200% 0; }}
            }}
        </style>
        """, unsafe_allow_html=True)


# ===========================================
# CONFIRMATION DIALOGS
# ===========================================

def confirm_action(
    message: str,
    confirm_text: str = "Confirm",
    cancel_text: str = "Cancel",
    key: str = None
) -> bool:
    """
    Show a confirmation dialog.
    
    Args:
        message: Confirmation message
        confirm_text: Text for confirm button
        cancel_text: Text for cancel button
        key: Unique key for the widget
        
    Returns:
        True if confirmed, False otherwise
    """
    col1, col2 = st.columns(2)
    with col1:
        confirmed = st.button(confirm_text, type="primary", key=f"{key}_confirm" if key else None, use_container_width=True)
    with col2:
        cancelled = st.button(cancel_text, key=f"{key}_cancel" if key else None, use_container_width=True)
    
    if confirmed:
        return True
    return False


# ===========================================
# DATA DISPLAY HELPERS
# ===========================================

def formatted_dataframe(
    df,
    title: str = None,
    show_index: bool = False,
    highlight_max: List[str] = None,
    highlight_min: List[str] = None
):
    """
    Display a formatted dataframe with optional highlighting.
    
    Args:
        df: DataFrame to display
        title: Optional title
        show_index: Whether to show index
        highlight_max: Columns to highlight max values
        highlight_min: Columns to highlight min values
    """
    if title:
        st.markdown(f"**{title}**")
    
    styled_df = df.style
    
    if highlight_max:
        styled_df = styled_df.highlight_max(subset=highlight_max, color='#dcfce7')
    
    if highlight_min:
        styled_df = styled_df.highlight_min(subset=highlight_min, color='#fee2e2')
    
    st.dataframe(styled_df, use_container_width=True, hide_index=not show_index)


def empty_state(
    title: str,
    message: str,
    icon: str = "📭",
    action_text: str = None,
    action_callback: Callable = None
):
    """
    Display an empty state message.
    
    Args:
        title: Title text
        message: Description
        icon: Icon to display
        action_text: Optional action button text
        action_callback: Optional callback when action clicked
    """
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 3rem;
        color: #6b7280;
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
        <div style="font-size: 1.25rem; font-weight: 600; color: #374151; margin-bottom: 0.5rem;">
            {title}
        </div>
        <div style="font-size: 0.95rem;">{message}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if action_text and action_callback:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(action_text, use_container_width=True):
                action_callback()


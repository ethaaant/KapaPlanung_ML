"""
Streamlit dashboard for workforce planning ML system.

Run with: streamlit run src/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import io

# Add project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import (
    DATA_RAW, DATA_PROCESSED, OUTPUTS_DIR, MODELS_DIR,
    DEFAULT_CAPACITY_CONFIG, DEFAULT_FORECAST_CONFIG,
    CapacityConfig, TaskConfig
)
from src.data.loader import DataLoader, create_sample_data
from src.data.preprocessor import Preprocessor
from src.models.forecaster import WorkloadForecaster
from src.models.capacity import CapacityPlanner

# Try to import Prophet forecaster
try:
    from src.models.prophet_forecaster import ProphetForecaster, ProphetConfig
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    ProphetForecaster = None

# Authentication
from src.auth import (
    init_auth_state, is_authenticated, is_admin, is_dienstleister,
    render_login_page, render_user_info_sidebar, get_current_user,
    logout
)

# Page configuration
st.set_page_config(
    page_title="Workforce Planning ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean, modern tile design
st.markdown("""
<style>
    /* ============================================
       GLOBAL TEXT COLORS - MAKE ALL TEXT VISIBLE
       ============================================ */
    
    /* Base text color for entire app */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label {
        color: #1f2937 !important;
    }
    
    /* Markdown text */
    .stMarkdown, .stMarkdown p, .stMarkdown span {
        color: #1f2937 !important;
    }
    
    /* All paragraphs */
    p {
        color: #1f2937 !important;
    }
    
    /* Labels */
    label, .stTextInput label, .stSelectbox label, .stSlider label,
    .stNumberInput label, .stDateInput label {
        color: #374151 !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #1f2937 !important;
    }
    
    /* ============================================
       MAIN APP BACKGROUND
       ============================================ */
    .stApp {
        background-color: #f5f7fa;
    }
    
    /* Main container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* ============================================
       HEADERS
       ============================================ */
    .main-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1a1a2e !important;
        margin-bottom: 0.25rem;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #6b7280 !important;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a2e !important;
    }
    
    h2 {
        font-weight: 600;
        font-size: 1.25rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    h3 {
        font-weight: 600;
        font-size: 1rem;
        color: #374151 !important;
    }
    
    /* ============================================
       TABS
       ============================================ */
    .stTabs [data-baseweb="tab-panel"] {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: white;
        border-radius: 10px;
        padding: 0.5rem;
        gap: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #4b5563 !important;
        font-weight: 500;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4f46e5 !important;
        color: white !important;
    }
    
    /* ============================================
       METRICS
       ============================================ */
    [data-testid="stMetric"] {
        background-color: white;
        padding: 1rem 1.25rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
    }
    
    [data-testid="stMetricLabel"] {
        color: #6b7280 !important;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    [data-testid="stMetricValue"] {
        color: #1a1a2e !important;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    [data-testid="stMetricDelta"] {
        color: #059669 !important;
    }
    
    /* ============================================
       TOOLTIP / HELP ICONS
       ============================================ */
    /* Style the help icon (ℹ) to be more visible */
    [data-testid="stTooltipIcon"] {
        color: #4f46e5 !important;
        opacity: 0.8;
    }
    
    [data-testid="stTooltipIcon"]:hover {
        opacity: 1;
        color: #4338ca !important;
    }
    
    /* Help icon in metric labels */
    [data-testid="stMetricLabel"] svg {
        color: #4f46e5 !important;
        width: 14px;
        height: 14px;
    }
    
    /* Tooltip content styling - comprehensive fix */
    [data-testid="stTooltipContent"] {
        background-color: #1f2937 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        font-size: 0.85rem !important;
        max-width: 280px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        min-height: auto !important;
        line-height: 1.4 !important;
    }
    
    [data-testid="stTooltipContent"] p,
    [data-testid="stTooltipContent"] span,
    [data-testid="stTooltipContent"] div {
        color: white !important;
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
        line-height: 1.4 !important;
    }
    
    /* BaseWeb tooltip body */
    [data-baseweb="tooltip"] {
        background-color: #1f2937 !important;
        border-radius: 8px !important;
        padding: 0 !important;
    }
    
    [data-baseweb="tooltip"] > div {
        background-color: #1f2937 !important;
        padding: 8px 12px !important;
    }
    
    /* Markdown inside tooltip */
    [data-testid="stTooltipContent"] [data-testid="stMarkdownContainer"],
    [data-testid="stTooltipContent"] [data-testid="stMarkdownContainer"] p {
        color: white !important;
        margin: 0 !important;
        background: transparent !important;
    }
    
    /* Role tooltip */
    div[role="tooltip"] {
        background-color: #1f2937 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
    }
    
    div[role="tooltip"] * {
        color: white !important;
        background: transparent !important;
    }
    
    /* Slider help icons */
    .stSlider label svg {
        color: #4f46e5 !important;
    }
    
    /* ============================================
       BUTTONS
       ============================================ */
    .stButton > button {
        background-color: #4f46e5;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #4338ca;
        color: white !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    
    .stDownloadButton > button {
        background-color: #10b981;
        color: white !important;
        border: none;
        border-radius: 8px;
    }
    
    .stDownloadButton > button:hover {
        background-color: #059669;
        color: white !important;
    }
    
    /* ============================================
       HIDE DEFAULT SIDEBAR
       ============================================ */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }
    
    /* ============================================
       STICKY TOP BAR
       ============================================ */
    .top-bar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999;
        background: linear-gradient(135deg, #1a1a2e 0%, #2d2d44 100%);
        padding: 0.5rem 2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        display: flex;
        align-items: center;
        gap: 2rem;
    }
    
    .top-bar-logo {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: white !important;
        font-weight: 600;
        font-size: 1.1rem;
        white-space: nowrap;
    }
    
    .top-bar-logo span {
        color: white !important;
    }
    
    .top-bar-config {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        flex-wrap: wrap;
    }
    
    .top-bar-item {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        color: rgba(255,255,255,0.9) !important;
        font-size: 0.8rem;
    }
    
    .top-bar-item label {
        color: rgba(255,255,255,0.7) !important;
        font-size: 0.75rem !important;
        font-weight: 400 !important;
        margin: 0 !important;
    }
    
    .top-bar-item .value {
        background: rgba(255,255,255,0.15);
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        color: white !important;
        font-weight: 500;
    }
    
    .top-bar-user {
        margin-left: auto;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .top-bar-user-info {
        color: rgba(255,255,255,0.9) !important;
        font-size: 0.85rem;
    }
    
    .top-bar-user-info .role {
        background: #4f46e5;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.7rem;
        margin-left: 0.5rem;
    }
    
    /* Add padding to main content to account for top bar */
    .main .block-container {
        padding-top: 4.5rem !important;
    }
    
    /* Settings expander in top bar style */
    .config-expander {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    
    .config-expander-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        cursor: pointer;
    }
    
    .config-inline {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        flex-wrap: wrap;
        background: #f8fafc;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    .config-inline label {
        font-size: 0.8rem !important;
        color: #64748b !important;
        margin-bottom: 0 !important;
    }
    
    /* Compact inputs for top config */
    .compact-input input, .compact-input select {
        height: 32px !important;
        font-size: 0.85rem !important;
        padding: 0.25rem 0.5rem !important;
    }
    
    /* ============================================
       INPUT FIELDS
       ============================================ */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stDateInput > div > div > input,
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        background-color: white;
        color: #1f2937 !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div:focus {
        border-color: #4f46e5;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background-color: #4f46e5;
    }
    
    .stSlider p {
        color: #1f2937 !important;
    }
    
    /* ============================================
       TABLES & DATAFRAMES
       ============================================ */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
    }
    
    /* ============================================
       ALERTS (Info, Warning, Error, Success)
       ============================================ */
    .stAlert, [data-testid="stAlert"] {
        border-radius: 10px;
        border: none;
    }
    
    /* ============================================
       FILE UPLOADER
       ============================================ */
    [data-testid="stFileUploader"] {
        background-color: white;
        border-radius: 10px;
        border: 2px dashed #d1d5db;
        padding: 1rem;
    }
    
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span {
        color: #4b5563 !important;
    }
    
    /* ============================================
       EXPANDER
       ============================================ */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        color: #1f2937 !important;
    }
    
    .streamlit-expanderHeader p {
        color: #1f2937 !important;
    }
    
    /* ============================================
       CHARTS
       ============================================ */
    [data-testid="stPlotlyChart"] {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
    }
        margin-top: 1rem;
    }
    
    /* Form styling */
    [data-testid="stForm"] {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
    }
    
    /* Number input */
    .stNumberInput > div > div > input {
        border-radius: 8px;
    }
    
    /* Horizontal rule */
    hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 1.5rem 0;
    }
    
    /* Login page specific */
    .login-box {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
    }
    
    /* Accent colors */
    .accent-primary { color: #4f46e5; }
    .accent-success { color: #10b981; }
    .accent-warning { color: #f59e0b; }
    .accent-danger { color: #ef4444; }
    
    /* Card container helper class */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a1a1a1;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False
    if "forecast_generated" not in st.session_state:
        st.session_state.forecast_generated = False
    if "combined_data" not in st.session_state:
        st.session_state.combined_data = None
    if "forecaster" not in st.session_state:
        st.session_state.forecaster = None
    if "forecast_df" not in st.session_state:
        st.session_state.forecast_df = None
    if "staffing_plan" not in st.session_state:
        st.session_state.staffing_plan = None


def render_top_bar():
    """Render the sticky top navigation bar with user info."""
    user = get_current_user()
    role_display = "Admin" if is_admin() else "Viewer"
    
    st.markdown(f"""
    <div class="top-bar">
        <div class="top-bar-logo">
            <span>📊</span>
            <span>Workforce Planning ML</span>
        </div>
        <div class="top-bar-user">
            <div class="top-bar-user-info">
                👤 {user.display_name if user else 'User'}
                <span class="role">{role_display}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def compact_config():
    """Render compact inline configuration panel."""
    # Capacity Planning Config
    with st.expander("⚙️ **Capacity Planning** — Service levels & handling times", expanded=False):
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            service_level_pct = st.number_input(
                "Service Level %",
                min_value=50,
                max_value=99,
                value=80,
                step=5,
                help="Target % of calls answered within wait time"
            )
        
        with col2:
            wait_time = st.number_input(
                "Wait Time (sec)",
                min_value=5,
                max_value=120,
                value=20,
                step=5,
                help="Max acceptable wait time"
            )
        
        with col3:
            shrinkage_pct = st.number_input(
                "Shrinkage %",
                min_value=10,
                max_value=50,
                value=30,
                step=5,
                help="Agent unavailability factor"
            )
        
        with col4:
            aht_calls = st.number_input(
                "Calls AHT (min)",
                value=5.0,
                min_value=1.0,
                max_value=30.0,
                step=0.5,
                help="Avg handle time for calls"
            )
        
        with col5:
            aht_emails = st.number_input(
                "Emails AHT (min)",
                value=8.0,
                min_value=1.0,
                max_value=30.0,
                step=0.5,
                help="Avg handle time for emails"
            )
        
        with col6:
            aht_outbound = st.number_input(
                "Outbound AHT (min)",
                value=6.0,
                min_value=1.0,
                max_value=30.0,
                step=0.5,
                help="Avg handle time for outbound"
            )
    
    # Business Metrics Config - Monthly Values
    with st.expander("📊 **Business Metrics** — Monthly leads & growth targets", expanded=False):
        st.markdown("##### Monthly Business Targets")
        st.caption("Set targets for each month. Contact Rate = Leads / Total Contacts (Calls + Emails + Outbound)")
        
        # Initialize business metrics in session state if not present
        if "business_metrics" not in st.session_state:
            st.session_state.business_metrics = {}
        
        # Get current year and create month options
        current_year = datetime.now().year
        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        
        # Year selector
        col_year, col_spacer = st.columns([1, 3])
        with col_year:
            selected_year = st.selectbox(
                "Year",
                options=[current_year, current_year + 1],
                index=0,
                help="Select year for business metrics"
            )
        
        # Create tabs for each month
        month_tabs = st.tabs([m[:3] for m in months])
        
        for i, (month_tab, month_name) in enumerate(zip(month_tabs, months)):
            month_key = f"{selected_year}-{i+1:02d}"
            
            with month_tab:
                # Get existing values or defaults
                existing = st.session_state.business_metrics.get(month_key, {})
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    leads = st.number_input(
                        "Created Leads",
                        min_value=0,
                        max_value=100000,
                        value=existing.get("leads", 0),
                        step=100,
                        key=f"leads_{month_key}",
                        help=f"Number of leads created in {month_name}"
                    )
                
                with col_b:
                    growth_pct = st.number_input(
                        "Expected Growth %",
                        min_value=-50.0,
                        max_value=200.0,
                        value=existing.get("growth_pct", 0.0),
                        step=1.0,
                        key=f"growth_{month_key}",
                        help="Growth adjustment on top of leads forecast"
                    )
                
                with col_c:
                    contact_rate = st.number_input(
                        "Contact Rate %",
                        min_value=0.0,
                        max_value=100.0,
                        value=existing.get("contact_rate", 0.0),
                        step=0.5,
                        key=f"contact_rate_{month_key}",
                        help="Leads / Total Contacts × 100"
                    )
                
                # Calculate adjusted leads
                adjusted_leads = int(leads * (1 + growth_pct / 100))
                if leads > 0:
                    st.caption(f"📈 Adjusted Leads (with growth): **{adjusted_leads:,}**")
                
                # Store in session state
                st.session_state.business_metrics[month_key] = {
                    "leads": leads,
                    "growth_pct": growth_pct,
                    "contact_rate": contact_rate,
                    "adjusted_leads": adjusted_leads
                }
        
        # Summary row
        st.markdown("---")
        st.markdown("##### Annual Summary")
        
        total_leads = sum(
            m.get("leads", 0) 
            for k, m in st.session_state.business_metrics.items() 
            if k.startswith(str(selected_year))
        )
        total_adjusted = sum(
            m.get("adjusted_leads", 0) 
            for k, m in st.session_state.business_metrics.items() 
            if k.startswith(str(selected_year))
        )
        avg_contact_rate = np.mean([
            m.get("contact_rate", 0) 
            for k, m in st.session_state.business_metrics.items() 
            if k.startswith(str(selected_year)) and m.get("contact_rate", 0) > 0
        ]) if any(
            m.get("contact_rate", 0) > 0 
            for k, m in st.session_state.business_metrics.items() 
            if k.startswith(str(selected_year))
        ) else 0
        
        sum_col1, sum_col2, sum_col3 = st.columns(3)
        with sum_col1:
            st.metric("Total Leads", f"{total_leads:,}")
        with sum_col2:
            st.metric("Adjusted Leads", f"{total_adjusted:,}")
        with sum_col3:
            st.metric("Avg Contact Rate", f"{avg_contact_rate:.1f}%")
    
    service_level = service_level_pct / 100.0
    shrinkage = shrinkage_pct / 100.0
    
    # Create config object
    tasks = {
        "calls": TaskConfig(name="Inbound Calls", avg_handling_time_minutes=aht_calls),
        "emails": TaskConfig(name="E-Mails", avg_handling_time_minutes=aht_emails, concurrency=2.0),
        "outbound_ook": TaskConfig(name="Outbound OOK", avg_handling_time_minutes=aht_outbound),
        "outbound_omk": TaskConfig(name="Outbound OMK", avg_handling_time_minutes=aht_outbound),
        "outbound_nb": TaskConfig(name="Outbound NB", avg_handling_time_minutes=aht_outbound),
    }
    
    config = CapacityConfig(
        service_level_target=service_level,
        service_level_time_seconds=wait_time,
        shrinkage_factor=shrinkage,
        tasks=tasks
    )
    
    return config


def render_logout_button():
    """Render logout button in top area."""
    col1, col2, col3 = st.columns([8, 1, 1])
    with col3:
        if st.button("🚪 Sign Out", key="logout_btn", use_container_width=True):
            logout()
            st.rerun()


def data_upload_section():
    """Render data upload section."""
    st.markdown("## 📁 Data Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Upload Your Data")
        
        uploaded_files = st.file_uploader(
            "Upload CSV or Excel files",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            help="Upload historical data for calls, emails, and outbound tasks"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Save to raw data directory
                save_path = DATA_RAW / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ Saved: {uploaded_file.name}")
    
    with col2:
        st.markdown("### Or Use Sample Data")
        
        if st.button("🎲 Generate Sample Data", type="primary"):
            with st.spinner("Generating sample data..."):
                create_sample_data(DATA_RAW)
            st.success("✅ Sample data generated!")
            st.session_state.data_loaded = False
    
    # Load data button
    st.markdown("---")
    
    if st.button("📊 Load Data", type="primary"):
        with st.spinner("Loading data..."):
            try:
                loader = DataLoader()
                data = loader.load_all()
                
                if not data:
                    st.error("No data files found. Please upload data or generate sample data.")
                    return
                
                combined = loader.combine_data()
                st.session_state.combined_data = combined
                st.session_state.data_loaded = True
                st.session_state.loader = loader
                st.success(f"✅ Loaded {len(combined)} hourly records")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")


def data_exploration_section():
    """Render data exploration section."""
    if not st.session_state.data_loaded:
        st.info("👆 Please load data first")
        return
    
    st.markdown("## 🔍 Data Exploration")
    
    data = st.session_state.combined_data
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Records", 
            f"{len(data):,}",
            help="Total number of hourly data points in the loaded dataset"
        )
    with col2:
        date_range = (data["timestamp"].max() - data["timestamp"].min()).days
        st.metric(
            "Date Range", 
            f"{date_range} days",
            help="Time span covered by the data, from first to last timestamp"
        )
    with col3:
        if "call_volume" in data.columns:
            st.metric(
                "Avg Daily Calls", 
                f"{data['call_volume'].sum() / date_range:.0f}",
                help="Average number of inbound calls received per day"
            )
    with col4:
        if "email_count" in data.columns:
            st.metric(
                "Avg Daily Emails", 
                f"{data['email_count'].sum() / date_range:.0f}",
                help="Average number of customer emails received per day"
            )
    
    # Time series plot
    st.markdown("### Volume Over Time")
    
    # Resample to daily for cleaner visualization
    daily = data.set_index("timestamp").resample("D").sum().reset_index()
    
    fig = go.Figure()
    
    if "call_volume" in daily.columns:
        fig.add_trace(go.Scatter(
            x=daily["timestamp"],
            y=daily["call_volume"],
            name="Calls",
            line=dict(color="#667eea")
        ))
    
    if "email_count" in daily.columns:
        fig.add_trace(go.Scatter(
            x=daily["timestamp"],
            y=daily["email_count"],
            name="Emails",
            line=dict(color="#764ba2")
        ))
    
    if "outbound_total" in daily.columns:
        fig.add_trace(go.Scatter(
            x=daily["timestamp"],
            y=daily["outbound_total"],
            name="Outbound",
            line=dict(color="#f093fb")
        ))
    
    fig.update_layout(
        title="Daily Volume Trends",
        xaxis_title="Date",
        yaxis_title="Volume",
        hovermode="x unified",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Hourly patterns
    st.markdown("### Hourly Patterns")
    
    data_with_hour = data.copy()
    data_with_hour["hour"] = data_with_hour["timestamp"].dt.hour
    
    hourly_avg = data_with_hour.groupby("hour").mean(numeric_only=True).reset_index()
    
    fig2 = go.Figure()
    
    for col in ["call_volume", "email_count", "outbound_total"]:
        if col in hourly_avg.columns:
            fig2.add_trace(go.Bar(
                x=hourly_avg["hour"],
                y=hourly_avg[col],
                name=col.replace("_", " ").title()
            ))
    
    fig2.update_layout(
        title="Average Volume by Hour of Day",
        xaxis_title="Hour",
        yaxis_title="Average Volume",
        barmode="group",
        template="plotly_white"
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Day of week patterns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Day of Week Patterns")
        
        data_with_dow = data.copy()
        data_with_dow["day_of_week"] = data_with_dow["timestamp"].dt.day_name()
        
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_avg = data_with_dow.groupby("day_of_week").mean(numeric_only=True).reindex(dow_order).reset_index()
        
        fig3 = px.bar(
            dow_avg,
            x="day_of_week",
            y=[c for c in ["call_volume", "email_count"] if c in dow_avg.columns],
            barmode="group",
            title="Average Volume by Day of Week"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.markdown("### Data Preview")
        st.dataframe(data.head(20), use_container_width=True)


def training_section():
    """Render model training section."""
    if not st.session_state.data_loaded:
        st.info("👆 Please load data first")
        return
    
    st.markdown("## 🧠 Model Training")
    
    # Model selection
    st.markdown("### Model Selection")
    
    model_options = ["Prophet (Recommended)"]
    if not PROPHET_AVAILABLE:
        model_options = ["Gradient Boosting"]
        st.warning("⚠️ Prophet not available. Using Gradient Boosting fallback.")
    else:
        model_options.append("Gradient Boosting (Legacy)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_type = st.selectbox(
            "Select Model",
            model_options,
            help="Prophet is recommended for complex seasonality (daily, weekly, yearly patterns)"
        )
    
    with col2:
        # Show model info
        if "Prophet" in model_type:
            st.info("🔮 **Prophet**: Best for seasonal patterns, holidays, and long-term forecasts")
        else:
            st.info("🌲 **Gradient Boosting**: Fast training, good for short-term forecasts")
    
    # Prophet-specific settings
    if "Prophet" in model_type and PROPHET_AVAILABLE:
        with st.expander("⚙️ Prophet Settings", expanded=False):
            prophet_col1, prophet_col2 = st.columns(2)
            
            with prophet_col1:
                seasonality_mode = st.selectbox(
                    "Seasonality Mode",
                    ["multiplicative", "additive"],
                    help="Multiplicative: seasonal effects scale with trend. Additive: constant seasonal effects."
                )
                
                yearly_seasonality = st.checkbox("Yearly Seasonality", value=True,
                    help="Capture yearly patterns (e.g., Christmas, summer)")
                weekly_seasonality = st.checkbox("Weekly Seasonality", value=True,
                    help="Capture day-of-week patterns")
            
            with prophet_col2:
                changepoint_scale = st.slider(
                    "Trend Flexibility",
                    min_value=0.01,
                    max_value=0.5,
                    value=0.05,
                    help="Higher = more flexible trend (may overfit)"
                )
                
                daily_seasonality = st.checkbox("Daily Seasonality", value=True,
                    help="Capture hour-of-day patterns")
                include_holidays = st.checkbox("German Holidays", value=True,
                    help="Include German public holidays and special events (Black Friday, Christmas)")
    else:
        # Gradient Boosting settings
        test_days = st.slider(
            "Validation Period (days)",
            min_value=7,
            max_value=30,
            value=14,
            help="Number of days to use for model validation"
        )
    
    # Train button
    st.markdown("---")
    
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training models... This may take a few minutes."):
            try:
                data = st.session_state.combined_data
                
                if "Prophet" in model_type and PROPHET_AVAILABLE:
                    # Train Prophet model
                    config = ProphetConfig(
                        seasonality_mode=seasonality_mode,
                        yearly_seasonality=yearly_seasonality,
                        weekly_seasonality=weekly_seasonality,
                        daily_seasonality=daily_seasonality,
                        changepoint_prior_scale=changepoint_scale
                    )
                    
                    forecaster = ProphetForecaster(config)
                    
                    # Determine target columns
                    target_cols = [c for c in data.columns 
                                  if c not in ['timestamp', 'date', 'hour']
                                  and pd.api.types.is_numeric_dtype(data[c])]
                    
                    metrics = forecaster.fit(data, target_columns=target_cols)
                    
                    # Save to session state
                    st.session_state.forecaster = forecaster
                    st.session_state.model_trained = True
                    st.session_state.training_metrics = metrics
                    st.session_state.model_type = "Prophet"
                    
                    # Save model
                    forecaster.save()
                    
                    st.success("✅ Prophet model trained successfully!")
                    
                else:
                    # Train Gradient Boosting model (legacy)
                    preprocessor = Preprocessor()
                    feature_set = preprocessor.fit_transform(data)
                    
                    forecaster = WorkloadForecaster()
                    metrics = forecaster.fit(feature_set, test_size_days=test_days)
                    
                    st.session_state.forecaster = forecaster
                    st.session_state.preprocessor = preprocessor
                    st.session_state.feature_set = feature_set
                    st.session_state.model_trained = True
                    st.session_state.training_metrics = metrics
                    st.session_state.model_type = "GradientBoosting"
                    
                    forecaster.save()
                    
                    st.success("✅ Gradient Boosting model trained successfully!")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Show training metrics
    if st.session_state.model_trained and "training_metrics" in st.session_state:
        model_name = st.session_state.get("model_type", "Unknown")
        st.markdown(f"### Training Metrics ({model_name})")
        
        metrics = st.session_state.training_metrics
        
        # Limit columns to 4 max for display
        n_cols = min(len(metrics), 4)
        cols = st.columns(n_cols)
        
        for i, (target, m) in enumerate(metrics.items()):
            with cols[i % n_cols]:
                st.markdown(f"**{target.replace('_', ' ').title()}**")
                st.metric(
                    "RMSE", 
                    f"{m['rmse']:.2f}",
                    help="Root Mean Square Error: Average prediction error magnitude. Lower is better."
                )
                st.metric(
                    "MAE", 
                    f"{m['mae']:.2f}",
                    help="Mean Absolute Error: Average absolute difference between predicted and actual values."
                )
                st.metric(
                    "R²", 
                    f"{m['r2']:.3f}",
                    help="R-squared: Proportion of variance explained. 1.0 = perfect prediction."
                )
                st.metric(
                    "MAPE", 
                    f"{m['mape']:.1f}%",
                    help="Mean Absolute Percentage Error. <10% excellent, 10-20% good."
                )
        
        # Model-specific visualizations
        if st.session_state.get("model_type") == "Prophet" and PROPHET_AVAILABLE:
            st.markdown("### 📊 Prophet Components")
            st.info("Prophet automatically decomposes your data into trend, weekly, and yearly patterns.")
            
            # Show seasonality info
            forecaster = st.session_state.forecaster
            
            with st.expander("View Seasonality Patterns", expanded=False):
                try:
                    target = list(forecaster.models.keys())[0]
                    model = forecaster.models[target]
                    
                    # Create figure for components
                    future = model.make_future_dataframe(periods=24*7, freq='H')
                    future['is_weekday'] = future['ds'].dt.dayofweek < 5
                    future['is_weekend'] = ~future['is_weekday']
                    forecast = model.predict(future)
                    
                    # Weekly pattern
                    weekly_pattern = forecast.groupby(forecast['ds'].dt.dayofweek)['weekly'].mean()
                    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                    
                    fig_weekly = go.Figure()
                    fig_weekly.add_trace(go.Bar(
                        x=days,
                        y=weekly_pattern.values,
                        marker_color='#667eea'
                    ))
                    fig_weekly.update_layout(
                        title=f"Weekly Pattern: {target.replace('_', ' ').title()}",
                        xaxis_title="Day of Week",
                        yaxis_title="Effect",
                        height=300
                    )
                    st.plotly_chart(fig_weekly, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Could not display components: {e}")
        
        elif st.session_state.get("model_type") == "GradientBoosting":
            # Feature importance for gradient boosting
            if hasattr(st.session_state.forecaster, 'get_feature_importance'):
                st.markdown("### Feature Importance")
                
                importance = st.session_state.forecaster.get_feature_importance()
                if len(importance) > 0:
                    top_n = min(15, len(importance))
                    
                    fig = px.bar(
                        importance.head(top_n),
                        x="avg_importance",
                        y="feature",
                        orientation="h",
                        title=f"Top {top_n} Most Important Features"
                    )
                    fig.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)


def forecast_section(capacity_config: CapacityConfig):
    """Render forecast generation section."""
    if not st.session_state.model_trained:
        st.info("👆 Please train the model first")
        return
    
    st.markdown("## 🔮 Generate Forecast")
    
    model_type = st.session_state.get("model_type", "Unknown")
    st.caption(f"Using: **{model_type}** model")
    
    # Get the last date in the training data
    data = st.session_state.combined_data
    last_data_date = data["timestamp"].max()
    
    # Default dates: start from day after last data, forecast 7 days
    default_start = (last_data_date + timedelta(days=1)).date()
    default_end = (last_data_date + timedelta(days=7)).date()
    
    # For Prophet, we can forecast further ahead
    max_forecast_days = 60 if model_type == "Prophet" else 30
    
    st.markdown("### Select Forecast Period")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        forecast_start = st.date_input(
            "Start Date",
            value=default_start,
            min_value=default_start,
            max_value=default_start + timedelta(days=90),
            help="First day to forecast (must be after training data ends)"
        )
    
    with col2:
        forecast_end = st.date_input(
            "End Date",
            value=default_end,
            min_value=forecast_start,
            max_value=forecast_start + timedelta(days=max_forecast_days),
            help=f"Last day to forecast (max {max_forecast_days} days from start)"
        )
    
    with col3:
        # Calculate and display forecast duration
        forecast_days = (forecast_end - forecast_start).days + 1
        st.metric(
            "Forecast Duration", 
            f"{forecast_days} days",
            help="Number of days the forecast covers. Longer forecasts have higher uncertainty."
        )
    
    # Show date range info
    st.info(f"📅 **Forecast Period:** {forecast_start.strftime('%A, %B %d, %Y')} to {forecast_end.strftime('%A, %B %d, %Y')} ({forecast_days} days, {forecast_days * 24} hours)")
    
    # Generate button
    if st.button("📈 Generate Forecast", type="primary", use_container_width=True):
        with st.spinner(f"Generating {forecast_days}-day forecast... This may take a while."):
            try:
                forecaster = st.session_state.forecaster
                
                # Calculate hours from last data point to end date
                start_datetime = datetime.combine(forecast_start, datetime.min.time())
                end_datetime = datetime.combine(forecast_end, datetime.max.time().replace(hour=23, minute=0, second=0, microsecond=0))
                
                # Hours to forecast
                hours_to_forecast = forecast_days * 24
                
                # Generate forecast based on model type
                if model_type == "Prophet":
                    # Prophet forecast
                    result = forecaster.forecast(
                        horizon_hours=hours_to_forecast,
                        start_date=start_datetime,
                        include_history=False
                    )
                else:
                    # Gradient Boosting forecast (legacy)
                    preprocessor = st.session_state.get("preprocessor")
                    
                    # Hours from end of data to end of forecast period
                    hours_from_data_end = int((end_datetime - last_data_date).total_seconds() / 3600)
                    
                    result = forecaster.forecast_horizon(
                        data,
                        horizon_hours=hours_from_data_end,
                        preprocessor=preprocessor,
                        start_date=pd.Timestamp(start_datetime),
                        confidence_level=0.95
                    )
                
                # Create forecast dataframe
                forecast_df = result.predictions.copy()
                forecast_df["timestamp"] = result.timestamps.values
                
                # Filter to only the requested date range
                forecast_df = forecast_df[
                    (forecast_df["timestamp"] >= pd.Timestamp(start_datetime)) &
                    (forecast_df["timestamp"] <= pd.Timestamp(end_datetime))
                ].reset_index(drop=True)
                
                # Reorder columns
                cols = ["timestamp"] + [c for c in forecast_df.columns if c != "timestamp"]
                forecast_df = forecast_df[cols]
                
                # Calculate staffing plan
                planner = CapacityPlanner(capacity_config)
                staffing_plan = planner.create_staffing_plan(forecast_df)
                
                # Save to session state
                st.session_state.forecast_df = forecast_df
                st.session_state.staffing_plan = staffing_plan
                st.session_state.forecast_generated = True
                st.session_state.forecast_start = forecast_start
                st.session_state.forecast_end = forecast_end
                st.session_state.forecast_result = result  # Store for confidence intervals
                
                # Reset scenario analyzer when new forecast generated
                if "scenario_analyzer" in st.session_state:
                    del st.session_state.scenario_analyzer
                
                st.success(f"✅ Generated forecast for {forecast_start} to {forecast_end} ({len(forecast_df)} hours)")
                
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def results_section():
    """Render forecast results and staffing plan."""
    if not st.session_state.forecast_generated:
        st.info("👆 Please generate a forecast first")
        return
    
    st.markdown("## 📊 Forecast Results")
    
    forecast_df = st.session_state.forecast_df
    staffing_plan = st.session_state.staffing_plan
    
    # Forecast visualization
    st.markdown("### Forecasted Volumes")
    
    fig = go.Figure()
    
    for col in forecast_df.columns:
        if col != "timestamp" and not col.startswith("outbound_"):
            fig.add_trace(go.Scatter(
                x=forecast_df["timestamp"],
                y=forecast_df[col],
                name=col.replace("_", " ").title(),
                mode="lines"
            ))
    
    fig.update_layout(
        title="Hourly Volume Forecast",
        xaxis_title="Date/Time",
        yaxis_title="Volume",
        hovermode="x unified",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Staffing plan visualization
    st.markdown("### Staffing Requirements")
    
    if len(staffing_plan) > 0:
        # Heatmap of agents needed
        staffing_plan["hour"] = pd.to_datetime(staffing_plan["timestamp"]).dt.hour
        staffing_plan["date"] = pd.to_datetime(staffing_plan["timestamp"]).dt.date
        
        pivot = staffing_plan.pivot_table(
            index="date",
            columns="hour",
            values="total_agents",
            aggfunc="max"
        )
        
        fig2 = px.imshow(
            pivot,
            labels=dict(x="Hour", y="Date", color="Agents"),
            title="Required Agents Heatmap",
            color_continuous_scale="Blues"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Daily summary
        st.markdown("### Daily Summary")
        
        daily_summary = staffing_plan.groupby("date").agg({
            "total_volume": "sum",
            "total_agents": ["max", "mean"]
        }).reset_index()
        daily_summary.columns = ["Date", "Total Volume", "Peak Agents", "Avg Agents"]
        daily_summary["Avg Agents"] = daily_summary["Avg Agents"].round(1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(daily_summary, use_container_width=True)
        
        with col2:
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=daily_summary["Date"].astype(str),
                y=daily_summary["Peak Agents"],
                name="Peak Agents"
            ))
            fig3.add_trace(go.Scatter(
                x=daily_summary["Date"].astype(str),
                y=daily_summary["Avg Agents"],
                name="Avg Agents",
                mode="lines+markers"
            ))
            fig3.update_layout(
                title="Daily Agent Requirements",
                xaxis_title="Date",
                yaxis_title="Agents"
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        # Weekly Staffing Schedule Section
        st.markdown("---")
        st.markdown("## 📅 Weekly Staffing Schedule")
        st.markdown("Average required agents per hour, broken down by day of week")
        
        # Prepare weekly data
        staffing_copy = staffing_plan.copy()
        staffing_copy["day_of_week"] = pd.to_datetime(staffing_copy["timestamp"]).dt.dayofweek
        staffing_copy["day_name"] = pd.to_datetime(staffing_copy["timestamp"]).dt.day_name()
        staffing_copy["hour"] = pd.to_datetime(staffing_copy["timestamp"]).dt.hour
        
        # Create weekly pivot table (average agents per day/hour)
        weekly_schedule = staffing_copy.pivot_table(
            index="day_of_week",
            columns="hour",
            values="total_agents",
            aggfunc="mean"
        ).round(1)
        
        # Sort by day of week and rename index
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekly_schedule.index = [day_names[i] for i in weekly_schedule.index]
        
        # Weekly Heatmap
        fig_weekly = go.Figure(data=go.Heatmap(
            z=weekly_schedule.values,
            x=[f"{h}:00" for h in weekly_schedule.columns],
            y=weekly_schedule.index,
            colorscale="Blues",
            text=weekly_schedule.values.round(0).astype(int),
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate="Day: %{y}<br>Hour: %{x}<br>Agents: %{z:.1f}<extra></extra>"
        ))
        
        fig_weekly.update_layout(
            title="Weekly Staffing Heatmap (Average Agents Required)",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            height=400,
            yaxis=dict(autorange="reversed")  # Monday at top
        )
        
        st.plotly_chart(fig_weekly, use_container_width=True)
        
        # Weekly Schedule Table
        st.markdown("### 📋 Weekly Schedule Table")
        
        # Format the table nicely
        weekly_table = weekly_schedule.copy()
        weekly_table.columns = [f"{h}:00" for h in weekly_table.columns]
        
        # Add row totals (average agents per day)
        weekly_table["Daily Avg"] = weekly_table.mean(axis=1).round(1)
        weekly_table["Peak Hour"] = weekly_schedule.max(axis=1).round(0).astype(int)
        
        st.dataframe(
            weekly_table.style.background_gradient(cmap="Blues", subset=weekly_table.columns[:-2]),
            use_container_width=True
        )
        
        # Summary metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric(
                "Peak Staffing",
                f"{int(weekly_schedule.values.max())} agents",
                help="Maximum agents needed at any single hour"
            )
        
        with col_m2:
            st.metric(
                "Avg Staffing",
                f"{weekly_schedule.values.mean():.1f} agents",
                help="Average agents needed across all hours"
            )
        
        with col_m3:
            # Find busiest day
            busiest_day = weekly_schedule.mean(axis=1).idxmax()
            st.metric(
                "Busiest Day",
                busiest_day,
                help="Day with highest average staffing need"
            )
        
        with col_m4:
            # Find peak hour
            peak_hour = weekly_schedule.mean(axis=0).idxmax()
            st.metric(
                "Peak Hour",
                f"{peak_hour}:00",
                help="Hour with highest average staffing need"
            )
        
        # Breakdown by task type if available
        agent_cols = [c for c in staffing_plan.columns if c.endswith("_agents") and c != "total_agents"]
        
        if agent_cols:
            st.markdown("### 📊 Staffing by Task Type")
            
            # Create tabs for each task type
            task_tabs = st.tabs([col.replace("_agents", "").replace("_", " ").title() for col in agent_cols])
            
            for tab, col in zip(task_tabs, agent_cols):
                with tab:
                    task_schedule = staffing_copy.pivot_table(
                        index="day_of_week",
                        columns="hour",
                        values=col,
                        aggfunc="mean"
                    ).round(1)
                    
                    task_schedule.index = [day_names[i] for i in task_schedule.index]
                    
                    fig_task = go.Figure(data=go.Heatmap(
                        z=task_schedule.values,
                        x=[f"{h}:00" for h in task_schedule.columns],
                        y=task_schedule.index,
                        colorscale="Greens",
                        text=task_schedule.values.round(0).astype(int),
                        texttemplate="%{text}",
                        textfont={"size": 11},
                    ))
                    
                    fig_task.update_layout(
                        title=f"{col.replace('_agents', '').replace('_', ' ').title()} - Agents Required",
                        xaxis_title="Hour",
                        yaxis_title="Day",
                        height=350,
                        yaxis=dict(autorange="reversed")
                    )
                    
                    st.plotly_chart(fig_task, use_container_width=True)


def export_section():
    """Render export section."""
    if not st.session_state.forecast_generated:
        st.info("👆 Please generate a forecast first")
        return
    
    st.markdown("## 💾 Export Results")
    
    forecast_df = st.session_state.forecast_df
    staffing_plan = st.session_state.staffing_plan
    
    # Format selector
    st.markdown("### Select Export Format")
    export_format = st.radio(
        "Choose format:",
        ["Excel (.xlsx)", "CSV (.csv)"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    if export_format == "Excel (.xlsx)":
        # Excel exports
        with col1:
            st.markdown("### 📊 Forecast Data")
            st.caption("Hourly volume predictions")
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                forecast_df.to_excel(writer, sheet_name="Forecast", index=False)
            
            st.download_button(
                label="📥 Download Forecast",
                data=buffer.getvalue(),
                file_name=f"forecast_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col2:
            st.markdown("### 👥 Staffing Plan")
            st.caption("Hourly agent requirements")
            
            if len(staffing_plan) > 0:
                buffer2 = io.BytesIO()
                with pd.ExcelWriter(buffer2, engine="xlsxwriter") as writer:
                    staffing_plan.to_excel(writer, sheet_name="Hourly Plan", index=False)
                    
                    daily = staffing_plan.groupby("date").agg({
                        "total_volume": "sum",
                        "total_agents": ["max", "mean"]
                    }).reset_index()
                    daily.columns = ["Date", "Total Volume", "Peak Agents", "Avg Agents"]
                    daily.to_excel(writer, sheet_name="Daily Summary", index=False)
                
                st.download_button(
                    label="📥 Download Staffing Plan",
                    data=buffer2.getvalue(),
                    file_name=f"staffing_plan_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        with col3:
            st.markdown("### 📋 Complete Report")
            st.caption("All data in one file")
            
            buffer3 = io.BytesIO()
            with pd.ExcelWriter(buffer3, engine="xlsxwriter") as writer:
                forecast_df.to_excel(writer, sheet_name="Forecast", index=False)
                if len(staffing_plan) > 0:
                    staffing_plan.to_excel(writer, sheet_name="Hourly Staffing", index=False)
                    
                    daily = staffing_plan.groupby("date").agg({
                        "total_volume": "sum",
                        "total_agents": ["max", "mean"]
                    }).reset_index()
                    daily.columns = ["Date", "Total Volume", "Peak Agents", "Avg Agents"]
                    daily.to_excel(writer, sheet_name="Daily Summary", index=False)
            
            st.download_button(
                label="📥 Download Complete Report",
                data=buffer3.getvalue(),
                file_name=f"workforce_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    else:
        # CSV exports
        with col1:
            st.markdown("### 📊 Forecast Data")
            st.caption("Hourly volume predictions")
            
            csv_forecast = forecast_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Forecast",
                data=csv_forecast,
                file_name=f"forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.markdown("### 👥 Staffing Plan")
            st.caption("Hourly agent requirements")
            
            if len(staffing_plan) > 0:
                csv_staffing = staffing_plan.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Staffing Plan",
                    data=csv_staffing,
                    file_name=f"staffing_plan_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col3:
            st.markdown("### 📆 Daily Summary")
            st.caption("Aggregated daily data")
            
            if len(staffing_plan) > 0:
                daily = staffing_plan.groupby("date").agg({
                    "total_volume": "sum",
                    "total_agents": ["max", "mean"]
                }).reset_index()
                daily.columns = ["Date", "Total Volume", "Peak Agents", "Avg Agents"]
                daily["Avg Agents"] = daily["Avg Agents"].round(1)
                
                csv_daily = daily.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Daily Summary",
                    data=csv_daily,
                    file_name=f"daily_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # Data preview section
    st.markdown("---")
    st.markdown("### 👁️ Data Preview")
    
    preview_tab1, preview_tab2 = st.tabs(["Forecast Data", "Staffing Plan"])
    
    with preview_tab1:
        st.dataframe(forecast_df.head(20), use_container_width=True)
        st.caption(f"Showing first 20 of {len(forecast_df)} rows")
    
    with preview_tab2:
        if len(staffing_plan) > 0:
            st.dataframe(staffing_plan.head(20), use_container_width=True)
            st.caption(f"Showing first 20 of {len(staffing_plan)} rows")


def analytics_section(capacity_config):
    """Data Science Analytics section with advanced analysis tools."""
    st.markdown("## 📈 Data Science Analytics")
    
    if not st.session_state.get("data_loaded", False):
        st.info("👆 Please load data first to access analytics")
        return
    
    # Sub-tabs for different analytics
    analytics_tabs = st.tabs([
        "🔬 Model Diagnostics",
        "📊 Time Series Analysis",
        "📉 Forecast Confidence",
        "🎯 What-If Scenarios"
    ])
    
    # Tab 1: Model Diagnostics
    with analytics_tabs[0]:
        render_model_diagnostics()
    
    # Tab 2: Time Series Decomposition
    with analytics_tabs[1]:
        render_time_series_analysis()
    
    # Tab 3: Forecast Confidence Visualization
    with analytics_tabs[2]:
        render_forecast_confidence()
    
    # Tab 4: What-If Scenario Analysis
    with analytics_tabs[3]:
        render_scenario_analysis(capacity_config)


def render_model_diagnostics():
    """Render model diagnostics sub-section."""
    st.markdown("### 🔬 Model Diagnostics")
    st.markdown("Analyze model performance with residual analysis and error breakdown.")
    
    if not st.session_state.get("model_trained", False):
        st.warning("⚠️ Train a model first to see diagnostics")
        return
    
    if not st.session_state.get("forecast_generated", False):
        st.warning("⚠️ Generate a forecast first to analyze residuals")
        return
    
    try:
        from src.analytics.diagnostics import ModelDiagnostics
        
        forecaster = st.session_state.forecaster
        forecast_df = st.session_state.forecast_df
        
        # For diagnostics, we need validation data (predictions vs actuals)
        # Using the training validation split results
        if hasattr(forecaster, '_training_residuals') and forecaster._training_residuals:
            st.success("✅ Model diagnostics available from training validation")
            
            # Get target columns
            target_cols = forecaster.target_columns
            
            # Select target to analyze
            selected_target = st.selectbox(
                "Select target to analyze",
                target_cols,
                help="Choose which workload type to analyze"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Residual distribution
                st.markdown("#### Residual Distribution")
                residuals = forecaster._training_residuals.get(selected_target, [])
                
                if len(residuals) > 0:
                    import plotly.graph_objects as go
                    from scipy import stats
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=residuals,
                        nbinsx=50,
                        name="Residuals",
                        marker_color="#667eea",
                        opacity=0.7
                    ))
                    
                    # Add normal curve
                    x_range = np.linspace(min(residuals), max(residuals), 100)
                    normal_pdf = stats.norm.pdf(x_range, np.mean(residuals), np.std(residuals))
                    scale_factor = len(residuals) * (max(residuals) - min(residuals)) / 50
                    
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=normal_pdf * scale_factor,
                        name="Normal Dist",
                        line=dict(color="#ef4444", width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"Residual Distribution: {selected_target.replace('_', ' ').title()}",
                        xaxis_title="Residual (Actual - Predicted)",
                        yaxis_title="Frequency",
                        height=350,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    st.markdown("**Residual Statistics:**")
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    with stats_col1:
                        st.metric("Mean", f"{np.mean(residuals):.2f}", help="Should be close to 0 for unbiased model")
                    with stats_col2:
                        st.metric("Std Dev", f"{np.std(residuals):.2f}", help="Lower is better")
                    with stats_col3:
                        # Shapiro-Wilk test for normality
                        if len(residuals) > 3:
                            _, p_val = stats.shapiro(residuals[:5000])
                            st.metric("Normal?", "Yes" if p_val > 0.05 else "No", 
                                     help=f"Shapiro-Wilk p-value: {p_val:.4f}")
            
            with col2:
                # Training metrics
                st.markdown("#### Training Metrics")
                if hasattr(forecaster, '_training_metrics') and selected_target in forecaster._training_metrics:
                    metrics = forecaster._training_metrics[selected_target]
                    
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}",
                                 help="Root Mean Square Error")
                        st.metric("R²", f"{metrics.get('r2', 0):.3f}",
                                 help="Coefficient of determination")
                    with m_col2:
                        st.metric("MAE", f"{metrics.get('mae', 0):.2f}",
                                 help="Mean Absolute Error")
                        st.metric("MAPE", f"{metrics.get('mape', 0):.1f}%",
                                 help="Mean Absolute Percentage Error")
            
            # Error by hour/day breakdown
            st.markdown("---")
            st.markdown("#### Error Pattern Analysis")
            st.info("💡 If errors are higher at certain hours or days, the model may need more features for those patterns.")
            
        else:
            st.info("📊 Detailed diagnostics require training data with validation split.")
            
    except ImportError as e:
        st.error(f"Analytics module not available: {e}")
    except Exception as e:
        st.error(f"Error in diagnostics: {str(e)}")


def render_time_series_analysis():
    """Render time series decomposition analysis."""
    st.markdown("### 📊 Time Series Decomposition")
    st.markdown("Analyze trend, seasonality, and patterns in your data.")
    
    data = st.session_state.combined_data
    
    if data is None or len(data) == 0:
        st.warning("No data available")
        return
    
    try:
        from src.analytics.decomposition import TimeSeriesDecomposer, STATSMODELS_AVAILABLE
        
        if not STATSMODELS_AVAILABLE:
            st.error("statsmodels library required. Install with: pip install statsmodels")
            return
        
        decomposer = TimeSeriesDecomposer()
        
        # Column selection
        numeric_cols = [c for c in data.columns 
                       if c not in ["timestamp", "date"] and pd.api.types.is_numeric_dtype(data[c])]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_col = st.selectbox(
                "Select column to analyze",
                numeric_cols,
                help="Choose a workload column to decompose"
            )
        
        with col2:
            period = st.selectbox(
                "Seasonality period",
                [24, 168, 24*7],
                format_func=lambda x: {24: "Daily (24h)", 168: "Weekly (168h)", 24*7: "Weekly"}[x],
                help="Expected seasonal pattern length"
            )
        
        if st.button("🔄 Run Decomposition", type="primary"):
            with st.spinner("Decomposing time series..."):
                result = decomposer.decompose(data, selected_col, period=period)
                
                # Store in session for reuse
                st.session_state.decomposition_result = result
                st.session_state.decomposer = decomposer
        
        # Display results if available
        if hasattr(st.session_state, 'decomposition_result'):
            decomposer = st.session_state.decomposer
            
            # Decomposition plot
            st.markdown("#### Decomposition Components")
            fig = decomposer.plot_decomposition()
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            summary = decomposer.get_decomposition_summary()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📈 Trend**")
                trend_dir = "📈" if summary["trend"]["direction"] == "increasing" else "📉"
                st.metric(
                    "Trend Direction", 
                    f"{trend_dir} {summary['trend']['change_percent']:+.1f}%",
                    help="Overall change from start to end"
                )
                st.metric(
                    "Trend Strength",
                    f"{summary['trend']['strength']:.2f}",
                    help="0=no trend, 1=strong trend"
                )
            
            with col2:
                st.markdown("**🔄 Seasonality**")
                st.metric(
                    "Seasonal Strength",
                    f"{summary['seasonality']['strength']:.2f}",
                    help="0=no seasonality, 1=strong seasonality"
                )
                st.metric(
                    "Seasonal Range",
                    f"{summary['seasonality']['range']:.1f}",
                    help="Difference between peak and trough"
                )
            
            with col3:
                st.markdown("**📊 Residual**")
                st.metric(
                    "Residual Std",
                    f"{summary['residual']['std']:.2f}",
                    help="Unexplained variation"
                )
                st.metric(
                    "Mean",
                    f"{summary['residual']['mean']:.2f}",
                    help="Should be ~0"
                )
            
            # Seasonal patterns
            st.markdown("---")
            st.markdown("#### Seasonal Patterns")
            
            pattern_col1, pattern_col2 = st.columns(2)
            
            with pattern_col1:
                fig_hourly = decomposer.plot_seasonal_pattern(aggregate_by="hour")
                st.plotly_chart(fig_hourly, use_container_width=True)
            
            with pattern_col2:
                fig_daily = decomposer.plot_seasonal_pattern(aggregate_by="dayofweek")
                st.plotly_chart(fig_daily, use_container_width=True)
            
            # ACF/PACF
            st.markdown("---")
            st.markdown("#### Autocorrelation Analysis")
            st.markdown("Helps identify significant lags and patterns in the data.")
            
            fig_acf = decomposer.plot_acf_pacf(data, selected_col, n_lags=48)
            st.plotly_chart(fig_acf, use_container_width=True)
            
            # Stationarity test
            st.markdown("---")
            st.markdown("#### Stationarity Test (ADF)")
            
            stationarity = decomposer.test_stationarity(data, selected_col)
            
            if stationarity.is_stationary:
                st.success(f"✅ Series is **stationary** (p-value: {stationarity.p_value:.4f})")
            else:
                st.warning(f"⚠️ Series is **non-stationary** (p-value: {stationarity.p_value:.4f})")
                st.info("Consider differencing or detrending for some modeling approaches.")
            
    except ImportError as e:
        st.error(f"Required library not available: {e}")
        st.info("Install with: pip install statsmodels")
    except Exception as e:
        st.error(f"Error in time series analysis: {str(e)}")


def render_forecast_confidence():
    """Render forecast confidence visualization."""
    st.markdown("### 📉 Forecast Confidence Intervals")
    st.markdown("Visualize prediction uncertainty and confidence bands.")
    
    if not st.session_state.get("forecast_generated", False):
        st.warning("⚠️ Generate a forecast first to see confidence intervals")
        return
    
    forecast_df = st.session_state.forecast_df
    forecaster = st.session_state.forecaster
    
    # Check if we have confidence intervals
    if not hasattr(st.session_state, 'forecast_result') or st.session_state.forecast_result is None:
        st.info("Confidence intervals are calculated during forecast generation.")
        st.info("Re-run the forecast to see confidence bands.")
        return
    
    result = st.session_state.forecast_result
    
    if result.confidence_intervals is None:
        st.warning("No confidence intervals available for this forecast.")
        return
    
    # Select target to visualize
    target_cols = list(result.confidence_intervals.keys())
    
    selected_target = st.selectbox(
        "Select target to visualize",
        target_cols,
        help="Choose which workload type to display"
    )
    
    # Get confidence data
    ci_data = result.confidence_intervals[selected_target]
    timestamps = result.timestamps
    
    # Create plot
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([timestamps, timestamps[::-1]]),
        y=pd.concat([ci_data['upper'], ci_data['lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval',
        showlegend=True
    ))
    
    # Prediction line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=ci_data['prediction'],
        name='Forecast',
        line=dict(color='#667eea', width=2)
    ))
    
    # Upper bound
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=ci_data['upper'],
        name='Upper Bound',
        line=dict(color='#667eea', width=1, dash='dash'),
        showlegend=False
    ))
    
    # Lower bound
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=ci_data['lower'],
        name='Lower Bound',
        line=dict(color='#667eea', width=1, dash='dash'),
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"Forecast with 95% Confidence Interval: {selected_target.replace('_', ' ').title()}",
        xaxis_title="Time",
        yaxis_title="Predicted Value",
        height=450,
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Confidence interval statistics
    st.markdown("#### Confidence Interval Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    avg_width = (ci_data['upper'] - ci_data['lower']).mean()
    max_width = (ci_data['upper'] - ci_data['lower']).max()
    min_width = (ci_data['upper'] - ci_data['lower']).min()
    avg_pred = ci_data['prediction'].mean()
    
    with col1:
        st.metric(
            "Avg Interval Width",
            f"{avg_width:.1f}",
            help="Average size of the confidence band"
        )
    
    with col2:
        st.metric(
            "Max Interval Width",
            f"{max_width:.1f}",
            help="Widest confidence band (highest uncertainty)"
        )
    
    with col3:
        st.metric(
            "Relative Uncertainty",
            f"{(avg_width / avg_pred * 100):.1f}%",
            help="Average uncertainty as % of predicted value"
        )
    
    with col4:
        st.metric(
            "Avg Prediction",
            f"{avg_pred:.1f}",
            help="Mean predicted value"
        )
    
    # Uncertainty over time
    st.markdown("---")
    st.markdown("#### Uncertainty Growth Over Time")
    st.info("💡 Confidence intervals typically widen as we forecast further into the future.")
    
    # Plot interval width over time
    interval_width = ci_data['upper'] - ci_data['lower']
    
    fig_width = go.Figure()
    fig_width.add_trace(go.Scatter(
        x=timestamps,
        y=interval_width,
        mode='lines',
        name='Interval Width',
        line=dict(color='#f59e0b', width=2)
    ))
    
    fig_width.update_layout(
        title="Confidence Interval Width Over Forecast Horizon",
        xaxis_title="Time",
        yaxis_title="Interval Width",
        height=300
    )
    
    st.plotly_chart(fig_width, use_container_width=True)


def render_scenario_analysis(capacity_config):
    """Render what-if scenario analysis."""
    st.markdown("### 🎯 What-If Scenario Analysis")
    st.markdown("Explore different scenarios and their impact on staffing requirements.")
    
    if not st.session_state.get("forecast_generated", False):
        st.warning("⚠️ Generate a forecast first to run scenario analysis")
        return
    
    forecast_df = st.session_state.forecast_df
    
    try:
        from src.analytics.scenarios import ScenarioAnalyzer
        from src.models.capacity import CapacityPlanner
        
        # Initialize capacity planner and scenario analyzer
        planner = CapacityPlanner(capacity_config)
        
        # Store or retrieve analyzer
        if "scenario_analyzer" not in st.session_state:
            st.session_state.scenario_analyzer = ScenarioAnalyzer(
                base_forecast=forecast_df,
                capacity_planner=planner,
                cost_per_agent_hour=25.0
            )
        
        analyzer = st.session_state.scenario_analyzer
        
        # Cost configuration
        with st.expander("💰 Cost Settings", expanded=False):
            cost_per_hour = st.number_input(
                "Cost per agent hour (€)",
                value=25.0,
                min_value=10.0,
                max_value=100.0,
                help="Hourly cost per agent for cost estimates"
            )
            analyzer.cost_per_agent_hour = cost_per_hour
        
        st.markdown("---")
        
        # Scenario creation
        st.markdown("#### Create Custom Scenario")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scenario_name = st.text_input(
                "Scenario Name",
                value="My Scenario",
                help="Give your scenario a descriptive name"
            )
            
            volume_change = st.slider(
                "Volume Change (%)",
                min_value=-50,
                max_value=100,
                value=0,
                step=5,
                help="Increase or decrease workload volume"
            )
        
        with col2:
            scenario_desc = st.text_input(
                "Description",
                value="Custom scenario",
                help="Brief description of this scenario"
            )
            
            aht_change = st.slider(
                "Handle Time Change (%)",
                min_value=-30,
                max_value=50,
                value=0,
                step=5,
                help="Increase or decrease average handling time"
            )
        
        if st.button("➕ Add Scenario", type="primary"):
            result = analyzer.add_scenario(
                name=scenario_name,
                description=scenario_desc,
                volume_change_pct=volume_change,
                aht_change_pct=aht_change,
                service_level=capacity_config.service_level_target,
                shrinkage=capacity_config.shrinkage_factor
            )
            st.success(f"✅ Added scenario: {scenario_name}")
            st.rerun()
        
        # Predefined scenarios
        st.markdown("---")
        st.markdown("#### Quick Scenarios")
        
        predefined = analyzer.get_predefined_scenarios()
        
        quick_cols = st.columns(3)
        for i, scenario in enumerate(predefined[:6]):
            with quick_cols[i % 3]:
                if st.button(
                    f"📌 {scenario['name']}", 
                    key=f"quick_{i}",
                    help=scenario['description'],
                    use_container_width=True
                ):
                    analyzer.add_scenario(
                        name=scenario['name'],
                        description=scenario['description'],
                        volume_change_pct=scenario.get('volume_change_pct', 0),
                        aht_change_pct=scenario.get('aht_change_pct', 0),
                        custom_adjustments=scenario.get('custom_adjustments')
                    )
                    st.rerun()
        
        # Scenario comparison
        if len(analyzer.scenarios) > 1:
            st.markdown("---")
            st.markdown("#### 📊 Scenario Comparison")
            
            # Comparison table
            comparison_df = analyzer.compare_scenarios()
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Visual comparison
            fig_comparison = analyzer.plot_scenario_comparison()
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Staffing pattern comparison
            st.markdown("#### Staffing Patterns by Scenario")
            fig_staffing = analyzer.plot_staffing_comparison(aggregate="hour")
            st.plotly_chart(fig_staffing, use_container_width=True)
        
        # Sensitivity analysis
        st.markdown("---")
        st.markdown("#### 🔍 Sensitivity Analysis")
        st.markdown("See how changes in key parameters affect staffing needs.")
        
        sens_param = st.selectbox(
            "Parameter to analyze",
            ["volume", "aht", "shrinkage"],
            format_func=lambda x: {"volume": "Volume Change", "aht": "Handle Time", "shrinkage": "Shrinkage Factor"}[x]
        )
        
        if st.button("📈 Run Sensitivity Analysis"):
            with st.spinner("Running analysis..."):
                if sens_param == "volume":
                    values = [-30, -20, -10, 0, 10, 20, 30, 50]
                    base_val = 0
                elif sens_param == "aht":
                    values = [-20, -10, 0, 10, 20, 30]
                    base_val = 0
                else:
                    values = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
                    base_val = 0.30
                
                fig_sens, sens_df = analyzer.sensitivity_analysis(sens_param, values, base_val)
                st.plotly_chart(fig_sens, use_container_width=True)
        
        # Service level trade-off
        st.markdown("---")
        st.markdown("#### ⚖️ Service Level vs Cost Trade-off")
        
        if st.button("📊 Analyze Trade-off"):
            with st.spinner("Calculating trade-offs..."):
                fig_tradeoff = analyzer.service_level_cost_tradeoff()
                st.plotly_chart(fig_tradeoff, use_container_width=True)
                
                st.info("💡 Higher service levels require disproportionately more agents. "
                       "The 'sweet spot' is often around 80-85%.")
        
    except ImportError as e:
        st.error(f"Scenario analysis module not available: {e}")
    except Exception as e:
        st.error(f"Error in scenario analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def dienstleister_view():
    """Simplified view for Dienstleister users - shows only staffing schedule."""
    # Render top navigation bar
    render_top_bar()
    
    # Header with logout
    col_header, col_logout = st.columns([9, 1])
    with col_header:
        st.markdown('<h1 class="main-header">📅 Agent Staffing Schedule</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sub-header">View the weekly staffing requirements</p>',
            unsafe_allow_html=True
        )
    with col_logout:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 Sign Out", key="logout_dienstleister", use_container_width=True):
            logout()
            st.rerun()
    
    # Check if we have a staffing plan
    if not st.session_state.get("forecast_generated", False):
        st.warning("⏳ No staffing plan available yet. Please wait for an administrator to generate the forecast.")
        
        # Try to load existing forecast if available
        st.info("💡 If a forecast was previously generated, click below to reload it.")
        if st.button("🔄 Check for Available Forecast"):
            st.rerun()
        return
    
    staffing_plan = st.session_state.staffing_plan
    
    if staffing_plan is None or len(staffing_plan) == 0:
        st.warning("No staffing data available.")
        return
    
    # Display forecast period
    forecast_start = st.session_state.get("forecast_start", staffing_plan["timestamp"].min())
    forecast_end = st.session_state.get("forecast_end", staffing_plan["timestamp"].max())
    
    st.info(f"📅 **Forecast Period:** {forecast_start} to {forecast_end}")
    
    # Prepare weekly data
    staffing_copy = staffing_plan.copy()
    staffing_copy["day_of_week"] = pd.to_datetime(staffing_copy["timestamp"]).dt.dayofweek
    staffing_copy["day_name"] = pd.to_datetime(staffing_copy["timestamp"]).dt.day_name()
    staffing_copy["hour"] = pd.to_datetime(staffing_copy["timestamp"]).dt.hour
    
    # Create weekly pivot table
    weekly_schedule = staffing_copy.pivot_table(
        index="day_of_week",
        columns="hour",
        values="total_agents",
        aggfunc="mean"
    ).round(1)
    
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly_schedule.index = [day_names[i] for i in weekly_schedule.index]
    
    # Summary metrics at the top
    st.markdown("## 📊 Quick Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🔝 Peak Staffing",
            f"{int(weekly_schedule.values.max())} agents",
            help="Maximum agents needed at any hour"
        )
    
    with col2:
        st.metric(
            "📈 Average Staffing",
            f"{weekly_schedule.values.mean():.1f} agents",
            help="Average agents across all hours"
        )
    
    with col3:
        busiest_day = weekly_schedule.mean(axis=1).idxmax()
        st.metric(
            "📅 Busiest Day",
            busiest_day,
            help="Day with highest staffing need"
        )
    
    with col4:
        peak_hour = weekly_schedule.mean(axis=0).idxmax()
        st.metric(
            "⏰ Peak Hour",
            f"{peak_hour}:00",
            help="Hour with highest staffing need"
        )
    
    st.markdown("---")
    
    # Weekly Heatmap
    st.markdown("## 🗓️ Weekly Staffing Heatmap")
    st.markdown("This shows the average number of agents needed for each hour of each day.")
    
    fig_weekly = go.Figure(data=go.Heatmap(
        z=weekly_schedule.values,
        x=[f"{h}:00" for h in weekly_schedule.columns],
        y=weekly_schedule.index,
        colorscale="Blues",
        text=weekly_schedule.values.round(0).astype(int),
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Agents: %{z:.1f}<extra></extra>"
    ))
    
    fig_weekly.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        height=450,
        yaxis=dict(autorange="reversed"),
        font=dict(size=14)
    )
    
    st.plotly_chart(fig_weekly, use_container_width=True)
    
    # Detailed Table
    st.markdown("## 📋 Detailed Weekly Schedule")
    st.markdown("Number of agents required per hour:")
    
    weekly_table = weekly_schedule.copy()
    weekly_table.columns = [f"{h}:00" for h in weekly_table.columns]
    weekly_table["Daily Avg"] = weekly_schedule.mean(axis=1).round(1)
    weekly_table["Peak"] = weekly_schedule.max(axis=1).round(0).astype(int)
    
    # Style the table
    st.dataframe(
        weekly_table.style.background_gradient(cmap="Blues", subset=weekly_table.columns[:-2])
            .format("{:.0f}", subset=weekly_table.columns[:-2])
            .format("{:.1f}", subset=["Daily Avg"]),
        use_container_width=True,
        height=320
    )
    
    # Daily breakdown
    st.markdown("## 📆 Daily Staffing Summary")
    
    if "date" in staffing_plan.columns:
        daily_summary = staffing_plan.groupby("date").agg({
            "total_volume": "sum",
            "total_agents": ["max", "mean"]
        }).reset_index()
        daily_summary.columns = ["Date", "Total Tasks", "Peak Agents", "Avg Agents"]
        daily_summary["Avg Agents"] = daily_summary["Avg Agents"].round(1)
        daily_summary["Date"] = pd.to_datetime(daily_summary["Date"]).dt.strftime("%a, %b %d")
        
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Bar(
            x=daily_summary["Date"],
            y=daily_summary["Peak Agents"],
            name="Peak Agents",
            marker_color="#4472C4"
        ))
        fig_daily.add_trace(go.Scatter(
            x=daily_summary["Date"],
            y=daily_summary["Avg Agents"],
            name="Avg Agents",
            mode="lines+markers",
            line=dict(color="#ED7D31", width=3),
            marker=dict(size=8)
        ))
        
        fig_daily.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Agents",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_daily, use_container_width=True)
    
    # Export option for Dienstleister
    st.markdown("---")
    st.markdown("## 💾 Download Schedule")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.markdown("##### Excel Format")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            weekly_table.to_excel(writer, sheet_name="Weekly Schedule", index=True)
            if "date" in staffing_plan.columns:
                daily_summary.to_excel(writer, sheet_name="Daily Summary", index=False)
        
        st.download_button(
            label="📥 Download Schedule (Excel)",
            data=buffer.getvalue(),
            file_name=f"staffing_schedule_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col_exp2:
        st.markdown("##### CSV Format")
        csv_weekly = weekly_table.to_csv(index=True)
        
        st.download_button(
            label="📥 Download Schedule (CSV)",
            data=csv_weekly,
            file_name=f"staffing_schedule_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )


def admin_view():
    """Full admin view with all features."""
    # Render top navigation bar
    render_top_bar()
    
    # Header with logout
    col_header, col_logout = st.columns([9, 1])
    with col_header:
        st.markdown('<h1 class="main-header">📊 Workforce Planning ML</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sub-header">Forecast customer service workload and plan agent capacity</p>',
            unsafe_allow_html=True
        )
    with col_logout:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 Sign Out", key="logout_top", use_container_width=True):
            logout()
            st.rerun()
    
    # Compact configuration panel
    capacity_config = compact_config()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📁 Data",
        "🔍 Explore",
        "🧠 Train",
        "🔮 Forecast",
        "📈 Analytics",
        "💾 Export"
    ])
    
    with tab1:
        data_upload_section()
    
    with tab2:
        data_exploration_section()
    
    with tab3:
        training_section()
    
    with tab4:
        forecast_section(capacity_config)
        results_section()
    
    with tab5:
        analytics_section(capacity_config)
    
    with tab6:
        export_section()


def main():
    """Main application entry point."""
    # Initialize authentication
    init_auth_state()
    init_session_state()
    
    # Check authentication
    if not is_authenticated():
        render_login_page()
        return
    
    # Route to appropriate view based on role
    if is_admin():
        admin_view()
    else:
        # Dienstleister view
        dienstleister_view()
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #888; font-size: 0.85rem;'>
            Workforce Planning ML System | CHECK24 Mobilfunk | © 2026
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()


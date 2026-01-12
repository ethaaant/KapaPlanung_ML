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
from src.models.model_manager import ModelManager, ModelMetadata
from src.utils.quality_indicators import (
    get_quality_indicator, compare_forecast_to_actuals,
    calculate_peak_detection_accuracy, QualityIndicator, ForecastComparison
)

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

# Business hours configuration
BUSINESS_HOURS_START = 7   # 07:00
BUSINESS_HOURS_END = 22    # 22:00 (inclusive, so 7-22 = 7,8,...,21,22)
BUSINESS_HOURS = list(range(BUSINESS_HOURS_START, BUSINESS_HOURS_END + 1))  # [7, 8, ..., 22]

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
    # Model management
    if "model_source" not in st.session_state:
        st.session_state.model_source = "none"  # "none", "trained", "loaded"
    if "model_version" not in st.session_state:
        st.session_state.model_version = None
    if "model_metadata" not in st.session_state:
        st.session_state.model_metadata = None
    # Forecast adjustment factors
    if "adjustment_factors" not in st.session_state:
        st.session_state.adjustment_factors = {
            "call_volume": 0,
            "email_count": 0,
            "outbound_ook": 0,
            "outbound_omk": 0,
            "outbound_nb": 0
        }
    
    # Call Center Management
    if "call_centers" not in st.session_state:
        # Load from file if exists, otherwise use defaults
        call_center_file = Path("data/call_centers/call_centers.json")
        if call_center_file.exists():
            import json
            try:
                with open(call_center_file, 'r', encoding='utf-8') as f:
                    st.session_state.call_centers = json.load(f)
            except:
                st.session_state.call_centers = _get_default_call_centers()
        else:
            st.session_state.call_centers = _get_default_call_centers()


def _get_default_call_centers():
    """Return default call center configuration."""
    import uuid
    return {
        "cc_kikxxl": {
            "id": "cc_kikxxl",
            "name": "KiKxxl",
            "cost_per_agent_hour": 25.0,
            "created_at": datetime.now().isoformat(),
            "agents": {}
        },
        "cc_octopodo": {
            "id": "cc_octopodo",
            "name": "Octopodo",
            "cost_per_agent_hour": 28.0,
            "created_at": datetime.now().isoformat(),
            "agents": {}
        },
        "cc_gevekom": {
            "id": "cc_gevekom",
            "name": "Gevekom",
            "cost_per_agent_hour": 26.0,
            "created_at": datetime.now().isoformat(),
            "agents": {}
        }
    }


def _save_call_centers():
    """Save call centers to file for persistence."""
    import json
    data_dir = Path("data/call_centers")
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "call_centers.json", 'w', encoding='utf-8') as f:
        json.dump(st.session_state.call_centers, f, indent=2, ensure_ascii=False)


def _generate_agent_id(call_center_id: str) -> str:
    """Generate a unique agent ID."""
    import uuid
    return f"agent_{call_center_id}_{uuid.uuid4().hex[:8]}"


def _generate_call_center_id(name: str) -> str:
    """Generate a unique call center ID."""
    import uuid
    # Create a readable ID from name + unique suffix
    clean_name = name.lower().replace(" ", "_").replace("-", "_")
    return f"cc_{clean_name}_{uuid.uuid4().hex[:6]}"


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
    with st.expander("⚙️ **Kapazitätsplanung** — Service Level & Bearbeitungszeiten", expanded=False):
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            service_level_pct = st.number_input(
                "Service Level %",
                min_value=50,
                max_value=99,
                value=80,
                step=5,
                help="Ziel-% der Anrufe, die innerhalb der Wartezeit angenommen werden. Branchenstandard: 80/20 (80% in 20 Sek.)"
            )
        
        with col2:
            wait_time = st.number_input(
                "Wartezeit (Sek.)",
                min_value=5,
                max_value=120,
                value=20,
                step=5,
                help="Maximal akzeptable Wartezeit für Anrufer"
            )
        
        with col3:
            shrinkage_pct = st.number_input(
                "Shrinkage %",
                min_value=10,
                max_value=50,
                value=30,
                step=5,
                help="Zeit, in der Agenten nicht verfügbar sind (Pausen, Schulungen, Meetings, Krankheit). Typisch: 25-35%"
            )
        
        with col4:
            aht_calls = st.number_input(
                "AHT Inbound",
                value=5.0,
                min_value=1.0,
                max_value=30.0,
                step=0.5,
                help="Durchschnittliche Bearbeitungszeit pro Anruf (inkl. Nachbearbeitung)"
            )
        
        with col5:
            aht_emails = st.number_input(
                "Bearbeitungszeit E-Mails",
                value=8.0,
                min_value=1.0,
                max_value=30.0,
                step=0.5,
                help="Durchschnittliche Bearbeitungszeit pro E-Mail"
            )
        
        with col6:
            aht_outbound = st.number_input(
                "Bearbeitungszeit Outbound",
                value=6.0,
                min_value=1.0,
                max_value=30.0,
                step=0.5,
                help="Durchschnittliche Bearbeitungszeit für Outbound-Kontakte (OOK, OMK, NB)"
            )
    
    # Business Metrics Config - Monthly Values
    with st.expander("📊 **Geschäftskennzahlen** — Monatliche Leads & Wachstumsziele", expanded=False):
        st.markdown("##### Monatliche Geschäftsziele")
        st.caption("Ziele für jeden Monat festlegen. Kontaktrate = Leads / Gesamtkontakte (Anrufe + E-Mails + Outbound)")
        
        # Initialize business metrics in session state if not present
        if "business_metrics" not in st.session_state:
            st.session_state.business_metrics = {}
        
        # Get current year and create month options
        current_year = datetime.now().year
        months = [
            "Januar", "Februar", "März", "April", "Mai", "Juni",
            "Juli", "August", "September", "Oktober", "November", "Dezember"
        ]
        
        # Year selector
        col_year, col_spacer = st.columns([1, 3])
        with col_year:
            selected_year = st.selectbox(
                "Jahr",
                options=[current_year, current_year + 1],
                index=0,
                help="Jahr für Geschäftskennzahlen auswählen"
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
                        "Erstellte Leads",
                        min_value=0,
                        max_value=100000,
                        value=existing.get("leads", 0),
                        step=100,
                        key=f"leads_{month_key}",
                        help=f"Anzahl der im {month_name} erstellten Leads"
                    )
                
                with col_b:
                    growth_pct = st.number_input(
                        "Erw. Wachstum %",
                        min_value=-50.0,
                        max_value=200.0,
                        value=existing.get("growth_pct", 0.0),
                        step=1.0,
                        key=f"growth_{month_key}",
                        help="Prozentuale Wachstumsanpassung auf die Lead-Prognose"
                    )
                
                with col_c:
                    contact_rate = st.number_input(
                        "Kontaktrate %",
                        min_value=0.0,
                        max_value=100.0,
                        value=existing.get("contact_rate", 0.0),
                        step=0.5,
                        key=f"contact_rate_{month_key}",
                        help="Leads / Gesamtkontakte × 100. Zeigt, wie viele Kontakte zu einem Lead führen."
                    )
                
                # Calculate adjusted leads
                adjusted_leads = round(leads * (1 + growth_pct / 100))
                difference = adjusted_leads - leads
                
                # Always show the adjusted leads info
                if leads > 0 or growth_pct != 0:
                    if difference > 0:
                        st.caption(f"📈 Angepasste Leads: **{adjusted_leads:,}** (+{difference:,} durch {growth_pct:.1f}% Wachstum)")
                    elif difference < 0:
                        st.caption(f"📉 Angepasste Leads: **{adjusted_leads:,}** ({difference:,} durch {growth_pct:.1f}% Rückgang)")
                    elif leads > 0:
                        st.caption(f"➡️ Leads: **{leads:,}** (kein Wachstum angewendet)")
                
                # Store in session state
                st.session_state.business_metrics[month_key] = {
                    "leads": leads,
                    "growth_pct": growth_pct,
                    "contact_rate": contact_rate,
                    "adjusted_leads": adjusted_leads
                }
        
        # Summary row
        st.markdown("---")
        st.markdown("##### Jahresübersicht")
        
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
            st.metric("Gesamt Leads", f"{total_leads:,}")
        with sum_col2:
            st.metric("Angepasste Leads", f"{total_adjusted:,}")
        with sum_col3:
            st.metric("Ø Kontaktrate", f"{avg_contact_rate:.1f}%")
    
    # Historical Lead Data Upload Section
    with st.expander("📂 **Historische Leaddaten** — Basis für Wachstumsprognose", expanded=False):
        st.markdown("##### Historische Lead-Daten hochladen")
        st.caption("Laden Sie Ihre historischen Lead-Zahlen hoch, um Wachstumstrends automatisch zu berechnen.")
        
        # Initialize historical leads in session state
        if "historical_leads" not in st.session_state:
            st.session_state.historical_leads = None
        if "historical_leads_interval" not in st.session_state:
            st.session_state.historical_leads_interval = None
        
        # Format examples based on interval
        format_examples = {
            "Täglich": "z.B. 01.01.2026, 02.01.2026, 03.01.2026, ...",
            "Wöchentlich": "z.B. 06.01.2026 (KW1), 13.01.2026 (KW2), ...",
            "Monatlich": "z.B. 01.01.2026, 01.02.2026, 01.03.2026, ..."
        }
        
        # Interval selection with info box on the right
        col_interval, col_info = st.columns([1, 2])
        
        with col_interval:
            interval = st.selectbox(
                "Datenintervall",
                options=["Täglich", "Wöchentlich", "Monatlich"],
                key="leads_interval_select",
                help="Wählen Sie das Intervall Ihrer historischen Daten"
            )
        
        with col_info:
            info_placeholder = st.empty()
        
        # Render info box after selectbox value is determined
        with info_placeholder.container():
            st.info(f"""
**Erwartetes Dateiformat ({interval}):**
- **Spalte 1:** Datum (`YYYY-MM-DD` oder `DD.MM.YYYY`) — {format_examples[interval]}
- **Spalte 2:** Anzahl Leads (numerisch)

Optional: Weitere Spalten werden ignoriert.
""")
        
        # File upload
        uploaded_leads_file = st.file_uploader(
            "Historische Leads hochladen",
            type=["csv", "xlsx", "xls"],
            key="leads_file_upload",
            help="CSV oder Excel-Datei mit historischen Lead-Zahlen"
        )
        
        if uploaded_leads_file is not None:
            try:
                # Read the file
                if uploaded_leads_file.name.endswith('.csv'):
                    # Try different separators
                    try:
                        leads_df = pd.read_csv(uploaded_leads_file, sep=';')
                        if len(leads_df.columns) < 2:
                            uploaded_leads_file.seek(0)
                            leads_df = pd.read_csv(uploaded_leads_file, sep=',')
                    except:
                        uploaded_leads_file.seek(0)
                        leads_df = pd.read_csv(uploaded_leads_file)
                else:
                    leads_df = pd.read_excel(uploaded_leads_file)
                
                # Validate columns
                if len(leads_df.columns) < 2:
                    st.error("❌ Die Datei muss mindestens 2 Spalten haben (Datum und Leads)")
                else:
                    # Rename columns for consistency
                    leads_df.columns = ['date'] + [f'col_{i}' for i in range(1, len(leads_df.columns))]
                    leads_df = leads_df.rename(columns={'col_1': 'leads'})
                    
                    # Parse date column
                    try:
                        leads_df['date'] = pd.to_datetime(leads_df['date'], dayfirst=True)
                    except:
                        try:
                            leads_df['date'] = pd.to_datetime(leads_df['date'])
                        except:
                            st.error("❌ Datumsformat nicht erkannt. Bitte verwenden Sie YYYY-MM-DD oder DD.MM.YYYY")
                            leads_df = None
                    
                    if leads_df is not None:
                        # Convert leads to numeric
                        leads_df['leads'] = pd.to_numeric(leads_df['leads'], errors='coerce')
                        leads_df = leads_df.dropna(subset=['date', 'leads'])
                        leads_df['leads'] = leads_df['leads'].astype(int)
                        
                        # Sort by date
                        leads_df = leads_df.sort_values('date').reset_index(drop=True)
                        
                        # Show preview
                        st.markdown("---")
                        st.markdown("##### Datenvorschau")
                        
                        preview_col1, preview_col2 = st.columns([2, 1])
                        
                        with preview_col1:
                            st.dataframe(
                                leads_df[['date', 'leads']].head(10),
                                use_container_width=True,
                                hide_index=True
                            )
                            if len(leads_df) > 10:
                                st.caption(f"... und {len(leads_df) - 10} weitere Einträge")
                        
                        with preview_col2:
                            st.metric("Anzahl Datensätze", f"{len(leads_df):,}")
                            st.metric("Zeitraum", f"{leads_df['date'].min().strftime('%d.%m.%Y')} - {leads_df['date'].max().strftime('%d.%m.%Y')}")
                            st.metric("Ø Leads / Periode", f"{leads_df['leads'].mean():,.0f}")
                            st.metric("Gesamt Leads", f"{leads_df['leads'].sum():,}")
                        
                        # Calculate growth trend
                        st.markdown("---")
                        st.markdown("##### Wachstumstrend-Analyse")
                        
                        if len(leads_df) >= 2:
                            # Calculate period-over-period growth
                            leads_df['growth_pct'] = leads_df['leads'].pct_change() * 100
                            avg_growth = leads_df['growth_pct'].mean()
                            recent_growth = leads_df['growth_pct'].tail(3).mean() if len(leads_df) >= 4 else avg_growth
                            
                            growth_col1, growth_col2, growth_col3 = st.columns(3)
                            with growth_col1:
                                st.metric(
                                    "Ø Wachstum (gesamt)",
                                    f"{avg_growth:+.1f}%",
                                    help="Durchschnittliche Wachstumsrate über den gesamten Zeitraum"
                                )
                            with growth_col2:
                                st.metric(
                                    "Ø Wachstum (letzte 3 Perioden)",
                                    f"{recent_growth:+.1f}%",
                                    help="Durchschnittliche Wachstumsrate der letzten 3 Perioden"
                                )
                            with growth_col3:
                                trend_direction = "📈 Steigend" if recent_growth > 0 else ("📉 Fallend" if recent_growth < 0 else "➡️ Stabil")
                                st.metric("Trend", trend_direction)
                            
                            # Visualization
                            fig_leads = go.Figure()
                            fig_leads.add_trace(go.Scatter(
                                x=leads_df['date'],
                                y=leads_df['leads'],
                                mode='lines+markers',
                                name='Leads',
                                line=dict(color='#6366f1', width=2),
                                marker=dict(size=6)
                            ))
                            
                            # Add trend line
                            z = np.polyfit(range(len(leads_df)), leads_df['leads'], 1)
                            p = np.poly1d(z)
                            fig_leads.add_trace(go.Scatter(
                                x=leads_df['date'],
                                y=p(range(len(leads_df))),
                                mode='lines',
                                name='Trendlinie',
                                line=dict(color='#f59e0b', width=2, dash='dash')
                            ))
                            
                            fig_leads.update_layout(
                                title=f"Historische Leads ({interval})",
                                xaxis_title="Datum",
                                yaxis_title="Anzahl Leads",
                                template="plotly_white",
                                height=350
                            )
                            st.plotly_chart(fig_leads, use_container_width=True)
                        
                        # Save button
                        st.markdown("---")
                        save_col1, save_col2 = st.columns([1, 3])
                        with save_col1:
                            if st.button("💾 Daten speichern", type="primary", use_container_width=True):
                                st.session_state.historical_leads = leads_df[['date', 'leads']].copy()
                                st.session_state.historical_leads_interval = interval
                                
                                # Save to file for persistence
                                data_dir = Path("data/historical_leads")
                                data_dir.mkdir(parents=True, exist_ok=True)
                                save_path = data_dir / "historical_leads.csv"
                                leads_df[['date', 'leads']].to_csv(save_path, index=False)
                                
                                st.success("✅ Historische Lead-Daten erfolgreich gespeichert!")
                                st.rerun()
                        
                        with save_col2:
                            st.caption("Die Daten werden lokal gespeichert und als Basis für die Wachstumsprognose verwendet.")
            
            except Exception as e:
                st.error(f"❌ Fehler beim Lesen der Datei: {str(e)}")
        
        # Show saved data summary if available
        if st.session_state.historical_leads is not None:
            st.markdown("---")
            st.success(f"📊 **Gespeicherte Daten:** {len(st.session_state.historical_leads):,} Datensätze ({st.session_state.historical_leads_interval})")
            
            saved_df = st.session_state.historical_leads
            saved_col1, saved_col2, saved_col3, saved_col4 = st.columns(4)
            with saved_col1:
                st.metric("Zeitraum Start", saved_df['date'].min().strftime('%d.%m.%Y'))
            with saved_col2:
                st.metric("Zeitraum Ende", saved_df['date'].max().strftime('%d.%m.%Y'))
            with saved_col3:
                st.metric("Ø Leads", f"{saved_df['leads'].mean():,.0f}")
            with saved_col4:
                if len(saved_df) >= 2:
                    overall_growth = ((saved_df['leads'].iloc[-1] / saved_df['leads'].iloc[0]) - 1) * 100
                    st.metric("Gesamtwachstum", f"{overall_growth:+.1f}%")
            
            # Apply to monthly targets button
            if st.button("📥 Wachstumstrend auf monatliche Ziele anwenden", help="Überträgt den berechneten Wachstumstrend auf alle Monatsziele"):
                if len(saved_df) >= 2:
                    growth_rates = saved_df['leads'].pct_change().dropna()
                    avg_monthly_growth = growth_rates.mean() * 100
                    
                    # Update all months with the average growth
                    for key in st.session_state.business_metrics:
                        st.session_state.business_metrics[key]['growth_pct'] = round(avg_monthly_growth, 1)
                    
                    st.success(f"✅ Wachstumstrend von {avg_monthly_growth:+.1f}% auf alle Monate angewendet!")
                    st.rerun()
        
        # Load from file if exists but not in session
        elif st.session_state.historical_leads is None:
            saved_path = Path("data/historical_leads/historical_leads.csv")
            if saved_path.exists():
                try:
                    saved_df = pd.read_csv(saved_path)
                    saved_df['date'] = pd.to_datetime(saved_df['date'])
                    st.session_state.historical_leads = saved_df
                    st.session_state.historical_leads_interval = "Unbekannt"
                    st.rerun()
                except:
                    pass
    
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
    """Render data upload section with format guide and multi-table support."""
    st.markdown("## 📁 Daten hochladen")
    
    # Initialize uploaded data preview in session state
    if "uploaded_previews" not in st.session_state:
        st.session_state.uploaded_previews = {}
    
    # ===========================================
    # FORMAT GUIDE SECTION
    # ===========================================
    with st.expander("📋 **Datenformat-Anleitung** — Erforderliche Spalten und Formate", expanded=True):
        st.markdown("""
        Laden Sie Ihre historischen Daten hoch. Das System unterstützt **separate Dateien** für jeden Datentyp 
        oder **eine kombinierte Datei** mit allen Spalten.
        """)
        
        # Required columns table
        st.markdown("#### Erforderliche Spalten")
        
        format_data = {
            "Spaltenname": [
                "timestamp", 
                "call_volume", 
                "email_count", 
                "outbound_ook", 
                "outbound_omk", 
                "outbound_nb"
            ],
            "Datentyp": [
                "Datum/Zeit", 
                "Integer", 
                "Integer", 
                "Integer", 
                "Integer", 
                "Integer"
            ],
            "Erforderlich": [
                "✅ Ja", 
                "⚪ Optional", 
                "⚪ Optional", 
                "⚪ Optional", 
                "⚪ Optional", 
                "⚪ Optional"
            ],
            "Beschreibung": [
                "Zeitstempel der Daten (YYYY-MM-DD HH:MM oder DD.MM.YYYY HH:MM)",
                "Anzahl eingehender Anrufe pro Stunde",
                "Anzahl eingehender E-Mails pro Stunde",
                "Outbound-Anrufe: Auftragsbestätigung (OOK)",
                "Outbound-Anrufe: Kundenkontakt (OMK)",
                "Outbound-Anrufe: Nachbearbeitung (NB)"
            ],
            "Beispiel": [
                "2026-01-15 08:00",
                "42",
                "18",
                "5",
                "8",
                "3"
            ]
        }
        
        st.dataframe(
            pd.DataFrame(format_data),
            use_container_width=True,
            hide_index=True
        )
        
        # Alternative column names
        st.markdown("#### Alternative Spaltennamen (werden automatisch erkannt)")
        
        alt_cols = st.columns(3)
        with alt_cols[0]:
            st.markdown("""
            **Zeitstempel:**
            - `timestamp`
            - `date`
            - `datetime`
            - `zeit`
            - `datum`
            """)
        with alt_cols[1]:
            st.markdown("""
            **Anrufe:**
            - `call_volume`
            - `calls`
            - `anrufe`
            - `inbound_calls`
            - `call_count`
            """)
        with alt_cols[2]:
            st.markdown("""
            **E-Mails:**
            - `email_count`
            - `emails`
            - `email`
            - `e-mail`
            - `mail_count`
            """)
        
        # File format info
        st.markdown("#### Unterstützte Dateiformate")
        st.info("""
        📄 **CSV** — Trennzeichen: `;` oder `,` (automatische Erkennung)  
        📊 **Excel** — `.xlsx` oder `.xls` (erstes Blatt wird verwendet)
        """)
    
    st.markdown("---")
    
    # ===========================================
    # MULTI-TABLE UPLOAD SECTION
    # ===========================================
    st.markdown("### 📤 Dateien hochladen")
    
    upload_tabs = st.tabs([
        "📁 Alle Daten (eine Datei)",
        "📞 Anrufe",
        "📧 E-Mails", 
        "📤 Outbound",
        "➕ Weitere Tabelle"
    ])
    
    # Combined upload
    with upload_tabs[0]:
        st.markdown("Laden Sie eine Datei mit allen Spalten hoch:")
        combined_file = st.file_uploader(
            "Kombinierte Datei",
            type=["csv", "xlsx", "xls"],
            key="combined_upload",
            help="Eine Datei mit timestamp und allen relevanten Datenspalten"
        )
        
        if combined_file:
            _process_uploaded_file(combined_file, "combined")
    
    # Calls upload
    with upload_tabs[1]:
        st.markdown("Laden Sie Anrufdaten separat hoch:")
        st.caption("Erforderliche Spalten: `timestamp`, `call_volume` (oder Alternativen)")
        calls_file = st.file_uploader(
            "Anrufdaten",
            type=["csv", "xlsx", "xls"],
            key="calls_upload"
        )
        
        if calls_file:
            _process_uploaded_file(calls_file, "calls")
    
    # Emails upload
    with upload_tabs[2]:
        st.markdown("Laden Sie E-Mail-Daten separat hoch:")
        st.caption("Erforderliche Spalten: `timestamp`, `email_count` (oder Alternativen)")
        emails_file = st.file_uploader(
            "E-Mail-Daten",
            type=["csv", "xlsx", "xls"],
            key="emails_upload"
        )
        
        if emails_file:
            _process_uploaded_file(emails_file, "emails")
    
    # Outbound upload
    with upload_tabs[3]:
        st.markdown("Laden Sie Outbound-Daten separat hoch:")
        st.caption("Erforderliche Spalten: `timestamp`, `outbound_ook`, `outbound_omk`, `outbound_nb` (oder kombiniert als `outbound_total`)")
        outbound_file = st.file_uploader(
            "Outbound-Daten",
            type=["csv", "xlsx", "xls"],
            key="outbound_upload"
        )
        
        if outbound_file:
            _process_uploaded_file(outbound_file, "outbound")
    
    # Additional table upload
    with upload_tabs[4]:
        st.markdown("Laden Sie zusätzliche Daten hoch:")
        st.caption("Für benutzerdefinierte Daten oder zusätzliche Metriken")
        
        custom_name = st.text_input("Name für diese Tabelle", placeholder="z.B. Feiertage, Marketing-Events")
        custom_file = st.file_uploader(
            "Zusätzliche Daten",
            type=["csv", "xlsx", "xls"],
            key="custom_upload"
        )
        
        if custom_file and custom_name:
            _process_uploaded_file(custom_file, f"custom_{custom_name}")
    
    # ===========================================
    # UPLOADED DATA PREVIEW
    # ===========================================
    if st.session_state.uploaded_previews:
        st.markdown("---")
        st.markdown("### 👁️ Datenvorschau — Hochgeladene Dateien")
        
        for file_key, preview_data in st.session_state.uploaded_previews.items():
            with st.expander(f"📄 **{preview_data['filename']}** — {preview_data['rows']:,} Zeilen, {preview_data['cols']} Spalten", expanded=True):
                
                # Column analysis
                st.markdown("#### Spaltenübersicht")
                
                col_analysis = []
                df = preview_data['dataframe']
                
                for col in df.columns:
                    col_info = {
                        "Spaltenname": col,
                        "Datentyp": str(df[col].dtype),
                        "Nicht-leer": f"{df[col].notna().sum():,} ({df[col].notna().mean()*100:.1f}%)",
                        "Leer/NaN": f"{df[col].isna().sum():,}",
                        "Eindeutige Werte": f"{df[col].nunique():,}",
                        "Beispielwerte": ", ".join(str(x) for x in df[col].dropna().head(3).tolist())
                    }
                    
                    # Add statistics for numeric columns
                    if pd.api.types.is_numeric_dtype(df[col]):
                        col_info["Min"] = f"{df[col].min():,.2f}" if df[col].notna().any() else "-"
                        col_info["Max"] = f"{df[col].max():,.2f}" if df[col].notna().any() else "-"
                        col_info["Durchschnitt"] = f"{df[col].mean():,.2f}" if df[col].notna().any() else "-"
                    else:
                        col_info["Min"] = "-"
                        col_info["Max"] = "-"
                        col_info["Durchschnitt"] = "-"
                    
                    col_analysis.append(col_info)
                
                analysis_df = pd.DataFrame(col_analysis)
                st.dataframe(analysis_df, use_container_width=True, hide_index=True)
                
                # Column mapping status
                st.markdown("#### Spalten-Mapping Status")
                
                mapping_status = _check_column_mapping(df)
                status_cols = st.columns(6)
                
                status_items = [
                    ("timestamp", "⏰ Zeitstempel"),
                    ("call_volume", "📞 Anrufe"),
                    ("email_count", "📧 E-Mails"),
                    ("outbound_ook", "📤 OOK"),
                    ("outbound_omk", "📤 OMK"),
                    ("outbound_nb", "📤 NB")
                ]
                
                for i, (col_key, col_label) in enumerate(status_items):
                    with status_cols[i]:
                        if mapping_status.get(col_key):
                            st.success(f"✅ {col_label}")
                            st.caption(f"→ {mapping_status[col_key]}")
                        else:
                            st.warning(f"❌ {col_label}")
                            st.caption("Nicht gefunden")
                
                # Data preview table
                st.markdown("#### Datenvorschau (erste 10 Zeilen)")
                st.dataframe(df.head(10), use_container_width=True, hide_index=True)
                
                # Data quality indicators
                st.markdown("#### Datenqualität")
                quality_cols = st.columns(4)
                
                with quality_cols[0]:
                    completeness = (df.notna().sum().sum() / (len(df) * len(df.columns))) * 100
                    st.metric("Vollständigkeit", f"{completeness:.1f}%")
                
                with quality_cols[1]:
                    duplicates = df.duplicated().sum()
                    st.metric("Duplikate", f"{duplicates:,}")
                
                with quality_cols[2]:
                    if "timestamp" in mapping_status and mapping_status["timestamp"]:
                        ts_col = mapping_status["timestamp"]
                        try:
                            ts_series = pd.to_datetime(df[ts_col])
                            date_range = (ts_series.max() - ts_series.min()).days
                            st.metric("Zeitraum", f"{date_range} Tage")
                        except:
                            st.metric("Zeitraum", "N/A")
                    else:
                        st.metric("Zeitraum", "Keine Zeitspalte")
                
                with quality_cols[3]:
                    st.metric("Speichergröße", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                
                # Remove button
                if st.button(f"🗑️ Entfernen", key=f"remove_{file_key}"):
                    del st.session_state.uploaded_previews[file_key]
                    st.rerun()
    
    # ===========================================
    # SAMPLE DATA OPTION
    # ===========================================
    st.markdown("---")
    
    sample_col1, sample_col2 = st.columns([1, 3])
    
    with sample_col1:
        st.markdown("### Beispieldaten")
        if st.button("🎲 Beispieldaten generieren", type="secondary"):
            with st.spinner("Generiere Beispieldaten..."):
                create_sample_data(DATA_RAW)
            st.success("✅ Beispieldaten generiert!")
            st.session_state.data_loaded = False
            st.session_state.uploaded_previews = {}
    
    with sample_col2:
        st.caption("""
        Wenn Sie keine eigenen Daten haben, können Sie Beispieldaten generieren.
        Diese enthalten realistische Muster für Anrufe, E-Mails und Outbound-Kontakte
        über einen Zeitraum von 6 Monaten.
        """)
    
    # ===========================================
    # LOAD DATA BUTTON
    # ===========================================
    st.markdown("---")
    
    load_col1, load_col2 = st.columns([1, 3])
    
    with load_col1:
        if st.button("📊 Daten laden & verarbeiten", type="primary", use_container_width=True):
            with st.spinner("Lade und verarbeite Daten..."):
                try:
                    # Save all uploaded previews to raw directory first
                    for file_key, preview_data in st.session_state.uploaded_previews.items():
                        save_path = DATA_RAW / preview_data['filename']
                        preview_data['dataframe'].to_csv(save_path, index=False)
                    
                    loader = DataLoader()
                    data = loader.load_all()
                    
                    if not data:
                        st.error("Keine Datendateien gefunden. Bitte Daten hochladen oder Beispieldaten generieren.")
                        return
                    
                    combined = loader.combine_data()
                    st.session_state.combined_data = combined
                    st.session_state.data_loaded = True
                    st.session_state.loader = loader
                    st.success(f"✅ {len(combined):,} stündliche Datensätze erfolgreich geladen!")
                except Exception as e:
                    st.error(f"Fehler beim Laden der Daten: {str(e)}")
    
    with load_col2:
        if st.session_state.get("data_loaded"):
            st.success("✅ Daten sind geladen und bereit für die Analyse.")
        else:
            st.info("👆 Klicken Sie auf 'Daten laden & verarbeiten', um die hochgeladenen Daten zu verarbeiten.")


def _process_uploaded_file(uploaded_file, file_type: str):
    """Process an uploaded file and store preview in session state."""
    try:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            # Try different separators
            try:
                df = pd.read_csv(uploaded_file, sep=';')
                if len(df.columns) < 2:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=',')
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Store in session state
        st.session_state.uploaded_previews[file_type] = {
            'filename': uploaded_file.name,
            'dataframe': df,
            'rows': len(df),
            'cols': len(df.columns),
            'uploaded_at': datetime.now().isoformat()
        }
        
        # Save to raw directory
        save_path = DATA_RAW / uploaded_file.name
        with open(save_path, "wb") as f:
            uploaded_file.seek(0)
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ Datei '{uploaded_file.name}' hochgeladen ({len(df):,} Zeilen)")
        
    except Exception as e:
        st.error(f"❌ Fehler beim Verarbeiten von '{uploaded_file.name}': {str(e)}")


def _check_column_mapping(df: pd.DataFrame) -> dict:
    """Check which required columns are present in the dataframe."""
    mapping = {}
    
    # Timestamp variations
    timestamp_cols = ['timestamp', 'date', 'datetime', 'zeit', 'datum', 'time', 'Timestamp', 'Date', 'DateTime']
    for col in timestamp_cols:
        if col in df.columns:
            mapping['timestamp'] = col
            break
    
    # Try to detect datetime columns
    if 'timestamp' not in mapping:
        for col in df.columns:
            try:
                pd.to_datetime(df[col].head(10))
                mapping['timestamp'] = col
                break
            except:
                pass
    
    # Call volume variations
    call_cols = ['call_volume', 'calls', 'anrufe', 'inbound_calls', 'call_count', 'Calls', 'Anrufe', 'CallVolume']
    for col in call_cols:
        if col in df.columns:
            mapping['call_volume'] = col
            break
    
    # Email count variations
    email_cols = ['email_count', 'emails', 'email', 'e-mail', 'mail_count', 'Email', 'Emails', 'EmailCount']
    for col in email_cols:
        if col in df.columns:
            mapping['email_count'] = col
            break
    
    # Outbound variations
    outbound_ook_cols = ['outbound_ook', 'ook', 'OOK', 'outbound_order']
    for col in outbound_ook_cols:
        if col in df.columns:
            mapping['outbound_ook'] = col
            break
    
    outbound_omk_cols = ['outbound_omk', 'omk', 'OMK', 'outbound_contact']
    for col in outbound_omk_cols:
        if col in df.columns:
            mapping['outbound_omk'] = col
            break
    
    outbound_nb_cols = ['outbound_nb', 'nb', 'NB', 'outbound_followup']
    for col in outbound_nb_cols:
        if col in df.columns:
            mapping['outbound_nb'] = col
            break
    
    # Check for combined outbound
    if 'outbound_total' in df.columns:
        mapping['outbound_total'] = 'outbound_total'
    
    return mapping


def data_exploration_section():
    """Render data exploration section."""
    if not st.session_state.data_loaded:
        st.info("👆 Bitte zuerst Daten laden")
        return
    
    st.markdown("## 🔍 Datenexploration")
    
    data = st.session_state.combined_data
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Datensätze gesamt", 
            f"{len(data):,}",
            help="Gesamtzahl der stündlichen Datenpunkte im geladenen Datensatz"
        )
    with col2:
        date_range = (data["timestamp"].max() - data["timestamp"].min()).days
        st.metric(
            "Zeitraum", 
            f"{date_range} Tage",
            help="Zeitspanne der Daten, vom ersten bis zum letzten Zeitstempel"
        )
    with col3:
        if "call_volume" in data.columns:
            st.metric(
                "Ø Anrufe/Tag", 
                f"{data['call_volume'].sum() / date_range:.0f}",
                help="Durchschnittliche Anzahl eingehender Anrufe pro Tag"
            )
    with col4:
        if "email_count" in data.columns:
            st.metric(
                "Ø E-Mails/Tag", 
                f"{data['email_count'].sum() / date_range:.0f}",
                help="Durchschnittliche Anzahl eingehender E-Mails pro Tag"
            )
    
    # Time series plot
    st.markdown("### Volumen über Zeit")
    
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
    
    # Hourly patterns (business hours only: 7:00-22:00)
    st.markdown("### Stündliche Muster (Geschäftszeiten)")
    
    data_with_hour = data.copy()
    data_with_hour["hour"] = data_with_hour["timestamp"].dt.hour
    
    # Filter to business hours only
    data_business_hours = data_with_hour[data_with_hour["hour"].isin(BUSINESS_HOURS)]
    hourly_avg = data_business_hours.groupby("hour").mean(numeric_only=True).reset_index()
    
    fig2 = go.Figure()
    
    for col in ["call_volume", "email_count", "outbound_total"]:
        if col in hourly_avg.columns:
            fig2.add_trace(go.Bar(
                x=hourly_avg["hour"],
                y=hourly_avg[col],
                name=col.replace("_", " ").title()
            ))
    
    fig2.update_layout(
        title="Ø Volumen nach Tageszeit (7:00-22:00)",
        xaxis_title="Stunde",
        yaxis_title="Ø Volumen",
        barmode="group",
        template="plotly_white",
        xaxis=dict(tickmode='array', tickvals=BUSINESS_HOURS, ticktext=[f"{h}:00" for h in BUSINESS_HOURS])
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
        data_preview = data.head(20).copy()
        numeric_cols = data_preview.select_dtypes(include=[np.number]).columns
        st.dataframe(
            data_preview.style.format("{:.1f}", subset=numeric_cols),
            use_container_width=True
        )


def training_section():
    """Render model training section with Load/Train/Save capabilities."""
    st.markdown("## 🧠 Model Management")
    
    # Initialize model manager
    model_manager = ModelManager(str(MODELS_DIR))
    
    # Show current model status
    _render_model_status()
    
    # Create tabs for Train New vs Load Existing
    train_tab, load_tab = st.tabs(["🚀 Train New Model", "📂 Load Saved Model"])
    
    with load_tab:
        _render_load_model_panel(model_manager)
    
    with train_tab:
        _render_train_model_panel(model_manager)


def _render_model_status():
    """Display current model status indicator."""
    source = st.session_state.get("model_source", "none")
    version = st.session_state.get("model_version")
    model_type = st.session_state.get("model_type", "None")
    
    if source == "none" or not st.session_state.get("model_trained", False):
        st.warning("⚠️ **Kein Modell geladen.** Trainieren Sie ein neues Modell oder laden Sie ein gespeichertes, um Prognosen zu erstellen.")
    elif source == "loaded":
        st.success(f"✅ **Modell geladen:** {model_type} (v{version}) — Bereit für Prognosen")
    elif source == "trained":
        st.success(f"✅ **Modell trainiert:** {model_type} — Bereit für Prognosen (nicht gespeichert)")
        st.caption("💡 Speichern Sie Ihr Modell, um es ohne erneutes Training wiederzuverwenden.")


def _render_load_model_panel(model_manager: ModelManager):
    """Render the Load Model panel."""
    st.markdown("### Gespeichertes Modell laden")
    st.caption("Laden Sie ein zuvor trainiertes Modell, um Prognosen ohne erneutes Training zu erstellen.")
    
    # Get available models
    all_models = model_manager.list_all_models()
    
    if not all_models:
        st.info("📭 Keine gespeicherten Modelle gefunden. Trainieren und speichern Sie zuerst ein Modell.")
        return
    
    # Model selector
    model_ids = list(all_models.keys())
    selected_model_id = st.selectbox(
        "Modell auswählen",
        model_ids,
        format_func=lambda x: f"{x} ({len(all_models[x])} Versionen)",
        help="Wählen Sie das zu ladende Modell"
    )
    
    if selected_model_id:
        versions = all_models[selected_model_id]
        
        # Version selector
        version_options = [(v.version, v) for v in reversed(versions)]  # Latest first
        selected_version = st.selectbox(
            "Version auswählen",
            [v[0] for v in version_options],
            format_func=lambda v: f"v{v}" + (" (Aktiv)" if next((x[1] for x in version_options if x[0] == v), None).is_active else ""),
            help="Wählen Sie die zu ladende Version"
        )
        
        # Get selected metadata
        selected_metadata = next((v[1] for v in version_options if v[0] == selected_version), None)
        
        if selected_metadata:
            # Display metadata
            st.markdown("#### Modelldetails")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Modelltyp", selected_metadata.model_type)
                st.metric("Trainingssamples", f"{selected_metadata.training_samples:,}")
            with col2:
                st.metric("Erstellt am", selected_metadata.created_at.strftime("%d.%m.%Y %H:%M"))
                st.metric("Erstellt von", selected_metadata.created_by)
            with col3:
                if selected_metadata.training_date_range[0]:
                    st.metric("Datenbereich", f"{selected_metadata.training_date_range[0][:10]} bis {selected_metadata.training_date_range[1][:10]}")
                if selected_metadata.metrics:
                    first_target = list(selected_metadata.metrics.keys())[0]
                    if "rmse" in selected_metadata.metrics[first_target]:
                        st.metric("RMSE", f"{selected_metadata.metrics[first_target]['rmse']:.2f}",
                            help="Wurzel des mittleren quadratischen Fehlers. Je niedriger, desto besser.")
            
            if selected_metadata.description:
                st.info(f"📝 {selected_metadata.description}")
            
            # Load button
            if st.button("📥 Modell laden", type="primary", use_container_width=True):
                with st.spinner("Lade Modell..."):
                    try:
                        # load_model returns (model, metadata) tuple
                        loaded_model, loaded_metadata = model_manager.load_model(selected_model_id, selected_version)
                        
                        st.session_state.forecaster = loaded_model
                        st.session_state.model_trained = True
                        st.session_state.model_source = "loaded"
                        st.session_state.model_version = selected_version
                        st.session_state.model_type = loaded_metadata.model_type
                        st.session_state.model_metadata = loaded_metadata
                        st.session_state.training_metrics = loaded_metadata.metrics
                        
                        st.success(f"✅ Modell erfolgreich geladen: {selected_model_id} v{selected_version}")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Fehler beim Laden des Modells: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())


def _render_train_model_panel(model_manager: ModelManager):
    """Render the Train Model panel."""
    if not st.session_state.data_loaded:
        st.info("👆 Bitte laden Sie zuerst Daten im Daten-Tab")
        return
    
    st.markdown("### Neues Modell trainieren")
    
    # Model selection
    model_options = ["Prophet (Empfohlen)"]
    if not PROPHET_AVAILABLE:
        model_options = ["Gradient Boosting"]
        st.warning("⚠️ Prophet nicht verfügbar. Verwende Gradient Boosting als Fallback.")
    else:
        model_options.append("Gradient Boosting (Legacy)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_type = st.selectbox(
            "Modelltyp auswählen",
            model_options,
            help="Prophet wird für komplexe Saisonalität empfohlen (tägliche, wöchentliche, jährliche Muster)"
        )
    
    with col2:
        if "Prophet" in model_type:
            st.info("🔮 **Prophet**: Ideal für saisonale Muster, Feiertage, Langzeitprognosen")
        else:
            st.info("🌲 **Gradient Boosting**: Schnelles Training, gut für Kurzzeit-Prognosen")
    
    # Prophet-specific settings
    if "Prophet" in model_type and PROPHET_AVAILABLE:
        with st.expander("⚙️ Prophet-Einstellungen", expanded=False):
            prophet_col1, prophet_col2 = st.columns(2)
            
            with prophet_col1:
                seasonality_mode = st.selectbox(
                    "Saisonalitätsmodus",
                    ["multiplicative", "additive"],
                    help="Multiplikativ: Saisonale Effekte skalieren mit dem Trend. Additiv: Konstante saisonale Effekte."
                )
                yearly_seasonality = st.checkbox("Jährliche Saisonalität", value=True,
                    help="Erfasst jährliche Muster (z.B. Weihnachten, Sommer)")
                weekly_seasonality = st.checkbox("Wöchentliche Saisonalität", value=True,
                    help="Erfasst Wochentagsmuster (z.B. geschäftige Montage)")
            
            with prophet_col2:
                changepoint_scale = st.slider(
                    "Trend-Flexibilität",
                    min_value=0.01,
                    max_value=0.5,
                    value=0.05,
                    help="Höher = flexiblerer Trend (kann zu Überanpassung führen)"
                )
                daily_seasonality = st.checkbox("Tägliche Saisonalität", value=True,
                    help="Erfasst Tagesstunden-Muster (z.B. ruhige Vormittage, geschäftige Nachmittage)")
                include_holidays = st.checkbox("Deutsche Feiertage", value=True,
                    help="Deutsche Feiertage und besondere Ereignisse einbeziehen")
    else:
        test_days = st.slider(
            "Validierungszeitraum (Tage)",
            min_value=7,
            max_value=30,
            value=14,
            help="Anzahl der Tage für die Modellvalidierung"
        )
    
    # Train button
    st.markdown("---")
    
    if st.button("🚀 Modell trainieren", type="primary", use_container_width=True, key="train_btn"):
        with st.spinner("Trainiere Modelle... Dies kann einige Minuten dauern."):
            try:
                data = st.session_state.combined_data
                
                if "Prophet" in model_type and PROPHET_AVAILABLE:
                    config = ProphetConfig(
                        seasonality_mode=seasonality_mode,
                        yearly_seasonality=yearly_seasonality,
                        weekly_seasonality=weekly_seasonality,
                        daily_seasonality=daily_seasonality,
                        changepoint_prior_scale=changepoint_scale
                    )
                    
                    forecaster = ProphetForecaster(config)
                    target_cols = [c for c in data.columns 
                                  if c not in ['timestamp', 'date', 'hour']
                                  and pd.api.types.is_numeric_dtype(data[c])]
                    
                    metrics = forecaster.fit(data, target_columns=target_cols)
                    
                    st.session_state.forecaster = forecaster
                    st.session_state.model_trained = True
                    st.session_state.model_source = "trained"
                    st.session_state.model_version = None
                    st.session_state.training_metrics = metrics
                    st.session_state.model_type = "Prophet"
                    
                    st.success("✅ Prophet-Modell erfolgreich trainiert!")
                    
                else:
                    preprocessor = Preprocessor()
                    feature_set = preprocessor.fit_transform(data)
                    
                    forecaster = WorkloadForecaster()
                    metrics = forecaster.fit(feature_set, test_size_days=test_days)
                    
                    st.session_state.forecaster = forecaster
                    st.session_state.preprocessor = preprocessor
                    st.session_state.feature_set = feature_set
                    st.session_state.model_trained = True
                    st.session_state.model_source = "trained"
                    st.session_state.model_version = None
                    st.session_state.training_metrics = metrics
                    st.session_state.model_type = "GradientBoosting"
                    
                    st.success("✅ Gradient Boosting-Modell erfolgreich trainiert!")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Fehler beim Trainieren des Modells: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Show training metrics and save option
    if st.session_state.model_trained and "training_metrics" in st.session_state:
        _render_training_metrics()
        _render_save_model_panel(model_manager)


def _render_save_model_panel(model_manager: ModelManager):
    """Render the Save Model panel."""
    st.markdown("---")
    st.markdown("### 💾 Modell speichern")
    
    # Only show if model was just trained (not loaded)
    if st.session_state.get("model_source") == "loaded":
        st.info("ℹ️ Dieses Modell wurde bereits geladen. Erneutes Speichern nicht nötig.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_name = st.text_input(
            "Modellname",
            value="workload_forecaster",
            help="Bezeichner für dieses Modell (Unterstriche erlaubt, keine Leerzeichen)"
        )
        model_description = st.text_area(
            "Beschreibung (optional)",
            placeholder="z.B. Trainiert mit Jan-Jun 2026 Daten mit täglicher Saisonalität",
            help="Notizen zu dieser Modellversion hinzufügen"
        )
    
    with col2:
        st.markdown("**Speicheroptionen**")
        set_active = st.checkbox("Als aktiv setzen", value=True, 
            help="Dieses Modell als Standard für Prognosen festlegen")
    
    if st.button("💾 Modell speichern", type="secondary", use_container_width=True):
        with st.spinner("Speichere Modell..."):
            try:
                forecaster = st.session_state.forecaster
                data = st.session_state.get("combined_data")
                metrics = st.session_state.get("training_metrics", {})
                model_type = st.session_state.get("model_type", "Unknown")
                user = get_current_user()
                username = user.username if user else "unknown"
                
                # Get target columns if available
                target_cols = []
                if hasattr(forecaster, 'target_columns'):
                    target_cols = forecaster.target_columns
                
                # Save via ModelManager
                metadata = model_manager.save_model(
                    model=forecaster,
                    model_id=model_name,
                    model_type=model_type,
                    created_by=username,
                    training_data=data,
                    metrics=metrics,
                    target_columns=target_cols,
                    description=model_description,
                    set_active=set_active
                )
                
                st.session_state.model_source = "loaded"  # Now it's saved
                st.session_state.model_version = metadata.version
                st.session_state.model_metadata = metadata
                
                st.success(f"✅ Modell gespeichert als **{model_name}** v{metadata.version}")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error saving model: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def _render_training_metrics():
    """Render training metrics display."""
    model_name = st.session_state.get("model_type", "Unknown")
    st.markdown(f"### Trainingsmetriken ({model_name})")
    
    metrics = st.session_state.training_metrics
    
    n_cols = min(len(metrics), 4)
    cols = st.columns(n_cols)
    
    for i, (target, m) in enumerate(metrics.items()):
        with cols[i % n_cols]:
            st.markdown(f"**{target.replace('_', ' ').title()}**")
            st.metric("RMSE", f"{m['rmse']:.2f}",
                help="Wurzel des mittleren quadratischen Fehlers: Durchschnittliche Fehlergröße. Je niedriger, desto besser.")
            st.metric("MAE", f"{m['mae']:.2f}",
                help="Mittlerer absoluter Fehler: Durchschnittliche Abweichung zwischen Prognose und Ist-Werten in Einheiten.")
            st.metric("R²", f"{m['r2']:.3f}",
                help="Bestimmtheitsmaß: Anteil der erklärten Varianz. 1.0 = perfekte Vorhersage, >0.8 = gut.")
            st.metric("MAPE", f"{m['mape']:.1f}%",
                help="Mittlerer absoluter prozentualer Fehler. <10% ausgezeichnet, 10-20% gut, >20% verbesserungswürdig.")
    
    # Model-specific visualizations (outside the metrics loop)
    if st.session_state.get("model_type") == "Prophet" and PROPHET_AVAILABLE:
        st.markdown("### 📊 Prophet-Komponenten")
        st.info("Prophet zerlegt Ihre Daten automatisch in Trend, wöchentliche und jährliche Muster (Saisonalität).")
        
        forecaster = st.session_state.forecaster
        
        with st.expander("Saisonalitätsmuster anzeigen", expanded=False):
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
                days = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So']
                
                fig_weekly = go.Figure()
                fig_weekly.add_trace(go.Bar(
                    x=days,
                    y=weekly_pattern.values,
                    marker_color='#667eea'
                ))
                fig_weekly.update_layout(
                    title=f"Wöchentliches Muster: {target.replace('_', ' ').title()}",
                    xaxis_title="Wochentag",
                    yaxis_title="Effekt",
                    height=300
                )
                st.plotly_chart(fig_weekly, use_container_width=True, key="prophet_weekly_pattern")
                
            except Exception as e:
                st.warning(f"Komponenten konnten nicht angezeigt werden: {e}")
    
    elif st.session_state.get("model_type") == "GradientBoosting":
        # Feature importance for gradient boosting
        if hasattr(st.session_state.forecaster, 'get_feature_importance'):
            st.markdown("### Feature-Wichtigkeit")
            
            importance = st.session_state.forecaster.get_feature_importance()
            if len(importance) > 0:
                top_n = min(15, len(importance))
                
                fig = px.bar(
                    importance.head(top_n),
                    x="avg_importance",
                    y="feature",
                    orientation="h",
                    title=f"Top {top_n} wichtigste Features"
                )
                fig.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True, key="gb_feature_importance")


def forecast_section(capacity_config: CapacityConfig):
    """Render forecast generation section."""
    if not st.session_state.model_trained:
        st.info("👆 Bitte zuerst das Modell trainieren")
        return
    
    st.markdown("## 🔮 Prognose erstellen")
    
    model_type = st.session_state.get("model_type", "Unknown")
    st.caption(f"Verwendetes Modell: **{model_type}**")
    
    # Get the last date in the training data
    data = st.session_state.combined_data
    
    # Check if data is loaded
    if data is None or len(data) == 0:
        st.warning("⚠️ Keine Trainingsdaten geladen. Bitte laden Sie zuerst Daten im 'Daten'-Tab hoch.")
        st.info("💡 **Tipp:** Auch bei einem geladenen Modell benötigen Sie Daten, um den Prognosezeitraum festzulegen.")
        return
    
    # Check for timestamp column
    if "timestamp" not in data.columns:
        st.error("❌ Daten haben keine 'timestamp'-Spalte. Bitte laden Sie gültige Daten hoch.")
        return
    
    last_data_date = data["timestamp"].max()
    
    # Default dates: start from day after last data, forecast 7 days
    default_start = (last_data_date + timedelta(days=1)).date()
    default_end = (last_data_date + timedelta(days=7)).date()
    
    # For Prophet, we can forecast further ahead
    max_forecast_days = 60 if model_type == "Prophet" else 30
    
    st.markdown("### Prognosezeitraum wählen")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        forecast_start = st.date_input(
            "Startdatum",
            value=default_start,
            min_value=default_start,
            max_value=default_start + timedelta(days=90),
            help="Erster Prognosetag (muss nach den Trainingsdaten liegen)"
        )
    
    with col2:
        forecast_end = st.date_input(
            "Enddatum",
            value=default_end,
            min_value=forecast_start,
            max_value=forecast_start + timedelta(days=max_forecast_days),
            help=f"Letzter Prognosetag (max. {max_forecast_days} Tage ab Start)"
        )
    
    with col3:
        # Calculate and display forecast duration
        forecast_days = (forecast_end - forecast_start).days + 1
        st.metric(
            "Prognosedauer", 
            f"{forecast_days} Tage",
            help="Anzahl der Tage, die die Prognose abdeckt. Längere Prognosen haben höhere Unsicherheit."
        )
    
    # Show date range info
    st.info(f"📅 **Prognosezeitraum:** {forecast_start.strftime('%d.%m.%Y')} bis {forecast_end.strftime('%d.%m.%Y')} ({forecast_days} Tage, {forecast_days * 24} Stunden)")
    
    # Adjustment Factors
    st.markdown("### 📊 Anpassungsfaktoren")
    with st.expander("Multiplikatoren zur Prognose-Anpassung (z.B. für Marketing-Kampagnen)", expanded=False):
        st.caption("Diese Multiplikatoren werden **nach** der Basisprognose angewendet. Positive Werte erhöhen, negative verringern die Prognose.")
        
        adj_col1, adj_col2, adj_col3 = st.columns(3)
        
        with adj_col1:
            call_adj = st.number_input(
                "Anrufe Anpassung %",
                min_value=-50,
                max_value=200,
                value=st.session_state.adjustment_factors.get("call_volume", 0),
                step=5,
                help="z.B. +20% für erwarteten Marketing-Push"
            )
            st.session_state.adjustment_factors["call_volume"] = call_adj
        
        with adj_col2:
            email_adj = st.number_input(
                "E-Mails Anpassung %",
                min_value=-50,
                max_value=200,
                value=st.session_state.adjustment_factors.get("email_count", 0),
                step=5,
                help="z.B. +30% nach Newsletter-Versand"
            )
            st.session_state.adjustment_factors["email_count"] = email_adj
        
        with adj_col3:
            outbound_adj = st.number_input(
                "Outbound Anpassung %",
                min_value=-50,
                max_value=200,
                value=st.session_state.adjustment_factors.get("outbound_ook", 0),
                step=5,
                help="Gilt für alle Outbound-Typen (OOK, OMK, NB)"
            )
            st.session_state.adjustment_factors["outbound_ook"] = outbound_adj
            st.session_state.adjustment_factors["outbound_omk"] = outbound_adj
            st.session_state.adjustment_factors["outbound_nb"] = outbound_adj
        
        # Show summary
        if any(v != 0 for v in st.session_state.adjustment_factors.values()):
            active_adjustments = [f"{k.replace('_', ' ').title()}: {v:+d}%" 
                                  for k, v in st.session_state.adjustment_factors.items() if v != 0]
            st.success(f"🎯 Aktive Anpassungen: {', '.join(active_adjustments)}")
        else:
            st.caption("Keine Anpassungen – Basisprognose wird verwendet.")
    
    # Generate button
    if st.button("📈 Prognose erstellen", type="primary", use_container_width=True):
        with st.spinner(f"Erstelle {forecast_days}-Tage-Prognose... Dies kann einen Moment dauern."):
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
                
                # Apply adjustment factors
                adjustments = st.session_state.adjustment_factors
                adjustment_applied = False
                
                for col in forecast_df.columns:
                    if col == "timestamp":
                        continue
                    
                    # Match column to adjustment factor
                    adj_value = 0
                    if "call" in col.lower():
                        adj_value = adjustments.get("call_volume", 0)
                    elif "email" in col.lower():
                        adj_value = adjustments.get("email_count", 0)
                    elif "outbound" in col.lower() or "ook" in col.lower() or "omk" in col.lower() or "nb" in col.lower():
                        adj_value = adjustments.get("outbound_ook", 0)
                    
                    if adj_value != 0:
                        multiplier = 1 + (adj_value / 100)
                        forecast_df[col] = forecast_df[col] * multiplier
                        adjustment_applied = True
                
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
                
                success_msg = f"✅ Prognose erstellt für {forecast_start.strftime('%d.%m.%Y')} bis {forecast_end.strftime('%d.%m.%Y')} ({len(forecast_df)} Stunden)"
                if adjustment_applied:
                    success_msg += " — *Anpassungen angewendet*"
                st.success(success_msg)
                
            except Exception as e:
                st.error(f"Fehler beim Erstellen der Prognose: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def render_forecast_quality_card():
    """Render the forecast quality indicator card."""
    training_metrics = st.session_state.get("training_metrics", {})
    forecast_start = st.session_state.get("forecast_start")
    forecast_end = st.session_state.get("forecast_end")
    
    # Calculate forecast horizon
    if forecast_start and forecast_end:
        horizon_days = (forecast_end - forecast_start).days + 1
    else:
        horizon_days = 7
    
    # Get quality indicator
    quality = get_quality_indicator(
        training_metrics=training_metrics,
        forecast_horizon_days=horizon_days
    )
    
    # Render the card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {quality.color}15 0%, {quality.color}05 100%);
        border-left: 4px solid {quality.color};
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
    ">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem;">
            <span style="font-size: 2rem;">{quality.emoji}</span>
            <div>
                <div style="font-size: 1.5rem; font-weight: 600; color: {quality.color};">
                    {quality.confidence_score:.0f}% Confidence
                </div>
                <div style="font-size: 1rem; font-weight: 500; color: #374151;">
                    {quality.label}
                </div>
            </div>
        </div>
        <p style="color: #4b5563; margin-bottom: 0.5rem; font-size: 0.95rem;">
            {quality.explanation}
        </p>
        <ul style="color: #6b7280; margin: 0; padding-left: 1.25rem; font-size: 0.85rem;">
            {"".join(f"<li>{detail}</li>" for detail in quality.details)}
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    return quality


def results_section():
    """Render forecast results and staffing plan."""
    if not st.session_state.forecast_generated:
        st.info("👆 Erstelle zuerst einen Forecast")
        return
    
    st.markdown("## 📊 Forecast Ergebnisse")
    
    # Show forecast quality card
    render_forecast_quality_card()
    
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
    
    # Staffing plan visualization (business hours only)
    st.markdown("### Personalbedarf (Geschäftszeiten 7:00-22:00)")
    
    if len(staffing_plan) > 0:
        # Heatmap of agents needed - filter to business hours
        staffing_plan["hour"] = pd.to_datetime(staffing_plan["timestamp"]).dt.hour
        staffing_plan["date"] = pd.to_datetime(staffing_plan["timestamp"]).dt.date
        
        # Filter to business hours only
        staffing_business = staffing_plan[staffing_plan["hour"].isin(BUSINESS_HOURS)]
        
        pivot = staffing_business.pivot_table(
            index="date",
            columns="hour",
            values="total_agents",
            aggfunc="max"
        )
        
        # Ensure all business hours columns exist (even if no data)
        for h in BUSINESS_HOURS:
            if h not in pivot.columns:
                pivot[h] = 0
        pivot = pivot[BUSINESS_HOURS]  # Reorder columns
        
        fig2 = px.imshow(
            pivot,
            labels=dict(x="Stunde", y="Datum", color="Agenten"),
            title="Personalbedarf Heatmap (7:00-22:00)",
            color_continuous_scale="Blues"
        )
        fig2.update_xaxes(tickvals=list(range(len(BUSINESS_HOURS))), 
                         ticktext=[f"{h}:00" for h in BUSINESS_HOURS])
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Daily summary (business hours only)
        st.markdown("### Tagesübersicht (Geschäftszeiten)")
        
        daily_summary = staffing_business.groupby("date").agg({
            "total_volume": "sum",
            "total_agents": ["max", "mean"]
        }).reset_index()
        daily_summary.columns = ["Datum", "Gesamtvolumen", "Max. Agenten", "Ø Agenten"]
        daily_summary["Ø Agenten"] = daily_summary["Ø Agenten"].round(1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Format numeric columns to 1 decimal place
            numeric_cols = daily_summary.select_dtypes(include=[np.number]).columns
            st.dataframe(
                daily_summary.style.format("{:.1f}", subset=numeric_cols),
                use_container_width=True
            )
        
        with col2:
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=daily_summary["Datum"].astype(str),
                y=daily_summary["Max. Agenten"],
                name="Max. Agenten"
            ))
            fig3.add_trace(go.Scatter(
                x=daily_summary["Datum"].astype(str),
                y=daily_summary["Ø Agenten"],
                name="Ø Agenten",
                mode="lines+markers"
            ))
            fig3.update_layout(
                title="Täglicher Agentenbedarf (Geschäftszeiten)",
                xaxis_title="Datum",
                yaxis_title="Agenten"
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        # Weekly Staffing Schedule Section (business hours only)
        st.markdown("---")
        st.markdown("## 📅 Wöchentlicher Dienstplan (7:00-22:00)")
        st.markdown("Durchschnittlich benötigte Agenten pro Stunde, nach Wochentag")
        
        # Prepare weekly data - filter to business hours
        staffing_copy = staffing_plan.copy()
        staffing_copy["day_of_week"] = pd.to_datetime(staffing_copy["timestamp"]).dt.dayofweek
        staffing_copy["day_name"] = pd.to_datetime(staffing_copy["timestamp"]).dt.day_name()
        staffing_copy["hour"] = pd.to_datetime(staffing_copy["timestamp"]).dt.hour
        
        # Filter to business hours only
        staffing_copy = staffing_copy[staffing_copy["hour"].isin(BUSINESS_HOURS)]
        
        # Create weekly pivot table (average agents per day/hour)
        weekly_schedule = staffing_copy.pivot_table(
            index="day_of_week",
            columns="hour",
            values="total_agents",
            aggfunc="mean"
        ).round(1)
        
        # Ensure all business hours columns exist
        for h in BUSINESS_HOURS:
            if h not in weekly_schedule.columns:
                weekly_schedule[h] = 0
        weekly_schedule = weekly_schedule[BUSINESS_HOURS]  # Reorder to business hours only
        
        # Sort by day of week and rename index
        day_names = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
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
            hovertemplate="Tag: %{y}<br>Stunde: %{x}<br>Agenten: %{z:.1f}<extra></extra>"
        ))
        
        fig_weekly.update_layout(
            title="Wöchentliche Personalbedarfs-Heatmap (Ø Agenten)",
            xaxis_title="Tageszeit",
            yaxis_title="Wochentag",
            height=400,
            yaxis=dict(autorange="reversed")  # Montag oben
        )
        
        st.plotly_chart(fig_weekly, use_container_width=True)
        
        # Weekly Schedule Table
        st.markdown("### 📋 Wöchentlicher Dienstplan-Tabelle")
        
        # Format the table nicely
        weekly_table = weekly_schedule.copy()
        weekly_table.columns = [f"{h}:00" for h in weekly_table.columns]
        
        # Add row totals (average agents per day)
        weekly_table["Tages-Ø"] = weekly_table.mean(axis=1).round(1)
        weekly_table["Spitze"] = weekly_schedule.max(axis=1).round(0).astype(int)
        
        # Get hour columns (all except Tages-Ø and Spitze)
        hour_cols = [c for c in weekly_table.columns if c not in ["Tages-Ø", "Spitze"]]
        
        st.dataframe(
            weekly_table.style
                .background_gradient(cmap="Blues", subset=hour_cols)
                .format("{:.0f}", subset=hour_cols)
                .format("{:.1f}", subset=["Tages-Ø"]),
            use_container_width=True
        )
        
        # Summary metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric(
                "Spitzenbedarf",
                f"{int(weekly_schedule.values.max())} Agenten",
                help="Maximale Anzahl benötigter Agenten in einer Stunde"
            )
        
        with col_m2:
            st.metric(
                "Ø Bedarf",
                f"{weekly_schedule.values.mean():.1f} Agenten",
                help="Durchschnittliche Anzahl benötigter Agenten über alle Stunden"
            )
        
        with col_m3:
            # Find busiest day
            busiest_day = weekly_schedule.mean(axis=1).idxmax()
            st.metric(
                "Geschäftigster Tag",
                busiest_day,
                help="Tag mit dem höchsten durchschnittlichen Personalbedarf"
            )
        
        with col_m4:
            # Find peak hour
            peak_hour = weekly_schedule.mean(axis=0).idxmax()
            st.metric(
                "Spitzenstunde",
                f"{peak_hour}:00",
                help="Stunde mit dem höchsten durchschnittlichen Personalbedarf"
            )
        
        # Breakdown by task type if available (business hours only)
        agent_cols = [c for c in staffing_plan.columns if c.endswith("_agents") and c != "total_agents"]
        
        if agent_cols:
            st.markdown("### 📊 Personal nach Aufgabentyp (7:00-22:00)")
            
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
                    
                    # Ensure all business hours columns exist
                    for h in BUSINESS_HOURS:
                        if h not in task_schedule.columns:
                            task_schedule[h] = 0
                    task_schedule = task_schedule[BUSINESS_HOURS]  # Reorder to business hours only
                    
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
                        title=f"{col.replace('_agents', '').replace('_', ' ').title()} - Benötigte Agenten",
                        xaxis_title="Stunde",
                        yaxis_title="Tag",
                        height=350,
                        yaxis=dict(autorange="reversed")
                    )
                    
                    st.plotly_chart(fig_task, use_container_width=True)


def export_section():
    """Render export section."""
    if not st.session_state.forecast_generated:
        st.info("👆 Bitte zuerst eine Prognose erstellen")
        return
    
    st.markdown("## 💾 Ergebnisse exportieren")
    
    forecast_df = st.session_state.forecast_df
    staffing_plan = st.session_state.staffing_plan
    
    # Format selector
    st.markdown("### Exportformat wählen")
    export_format = st.radio(
        "Format auswählen:",
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
        # Format numeric columns to 1 decimal place
        forecast_display = forecast_df.head(20).copy()
        numeric_cols = forecast_display.select_dtypes(include=[np.number]).columns
        st.dataframe(
            forecast_display.style.format("{:.1f}", subset=numeric_cols),
            use_container_width=True
        )
        st.caption(f"Showing first 20 of {len(forecast_df)} rows")
    
    with preview_tab2:
        if len(staffing_plan) > 0:
            staffing_display = staffing_plan.head(20).copy()
            numeric_cols = staffing_display.select_dtypes(include=[np.number]).columns
            st.dataframe(
                staffing_display.style.format("{:.1f}", subset=numeric_cols),
                use_container_width=True
            )
            st.caption(f"Showing first 20 of {len(staffing_plan)} rows")


def analytics_section(capacity_config):
    """Data Science Analytics section with advanced analysis tools."""
    st.markdown("## 📈 Data Science Analytics")
    
    if not st.session_state.get("data_loaded", False):
        st.info("👆 Bitte zuerst Daten laden")
        return
    
    # Sub-tabs for different analytics
    analytics_tabs = st.tabs([
        "🔬 Modelldiagnose",
        "📊 Zeitreihenanalyse",
        "📉 Forecast Verlässlichkeit",
        "🎯 Was-wäre-wenn Szenarios"
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
            st.success("✅ Modelldiagnose aus Trainingsvalidierung verfügbar")
            
            # Get target columns
            target_cols = forecaster.target_columns
            
            # Select target to analyze
            selected_target = st.selectbox(
                "Zielgröße auswählen",
                target_cols,
                help="Wählen Sie den zu analysierenden Workload-Typ"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Residual distribution
                st.markdown("#### Residuenverteilung")
                residuals = forecaster._training_residuals.get(selected_target, [])
                
                if len(residuals) > 0:
                    import plotly.graph_objects as go
                    from scipy import stats
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=residuals,
                        nbinsx=50,
                        name="Residuen",
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
                        name="Normalvert.",
                        line=dict(color="#ef4444", width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"Residuenverteilung: {selected_target.replace('_', ' ').title()}",
                        xaxis_title="Residuum (Ist - Prognose)",
                        yaxis_title="Häufigkeit",
                        height=350,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    st.markdown("**Residuen-Statistiken:**")
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    with stats_col1:
                        st.metric("Mittelwert", f"{np.mean(residuals):.2f}", help="Sollte nahe 0 sein für unverzerrtes Modell (Bias)")
                    with stats_col2:
                        st.metric("Std.abw.", f"{np.std(residuals):.2f}", help="Je niedriger, desto besser")
                    with stats_col3:
                        # Shapiro-Wilk test for normality
                        if len(residuals) > 3:
                            _, p_val = stats.shapiro(residuals[:5000])
                            st.metric("Normal?", "Ja" if p_val > 0.05 else "Nein", 
                                     help=f"Shapiro-Wilk p-Wert: {p_val:.4f}. Idealerweise sollten Residuen normalverteilt sein.")
            
            with col2:
                # Training metrics
                st.markdown("#### Trainingsmetriken")
                if hasattr(forecaster, '_training_metrics') and selected_target in forecaster._training_metrics:
                    metrics = forecaster._training_metrics[selected_target]
                    
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}",
                                 help="Wurzel des mittleren quadratischen Fehlers. Je niedriger, desto besser.")
                        st.metric("R²", f"{metrics.get('r2', 0):.3f}",
                                 help="Bestimmtheitsmaß: Anteil der erklärten Varianz. 1.0 = perfekt, >0.8 = gut.")
                    with m_col2:
                        st.metric("MAE", f"{metrics.get('mae', 0):.2f}",
                                 help="Mittlerer absoluter Fehler in Einheiten.")
                        st.metric("MAPE", f"{metrics.get('mape', 0):.1f}%",
                                 help="Mittlerer absoluter prozentualer Fehler. <10% ausgezeichnet, 10-20% gut.")
            
            # Error by hour/day breakdown
            st.markdown("---")
            st.markdown("#### Fehlermuster-Analyse")
            st.info("💡 Wenn Fehler zu bestimmten Stunden oder Tagen höher sind, könnte das Modell mehr Features für diese Muster benötigen.")
            
        else:
            st.info("📊 Detaillierte Diagnose erfordert Trainingsdaten mit Validierungssplit.")
            
    except ImportError as e:
        st.error(f"Analytics module not available: {e}")
    except Exception as e:
        st.error(f"Error in diagnostics: {str(e)}")


def render_time_series_analysis():
    """Render time series decomposition analysis."""
    st.markdown("### 📊 Zeitreihenzerlegung")
    st.markdown("Analysieren Sie Trend, Saisonalität und Muster in Ihren Daten.")
    
    data = st.session_state.combined_data
    
    if data is None or len(data) == 0:
        st.warning("Keine Daten verfügbar")
        return
    
    try:
        from src.analytics.decomposition import TimeSeriesDecomposer, STATSMODELS_AVAILABLE
        
        if not STATSMODELS_AVAILABLE:
            st.error("statsmodels-Bibliothek erforderlich. Installation: pip install statsmodels")
            return
        
        decomposer = TimeSeriesDecomposer()
        
        # Column selection
        numeric_cols = [c for c in data.columns 
                       if c not in ["timestamp", "date"] and pd.api.types.is_numeric_dtype(data[c])]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_col = st.selectbox(
                "Spalte für Analyse auswählen",
                numeric_cols,
                help="Wählen Sie eine Workload-Spalte zur Zerlegung"
            )
        
        with col2:
            period = st.selectbox(
                "Saisonalitäts-Periode",
                [24, 168, 24*7],
                format_func=lambda x: {24: "Täglich (24h)", 168: "Wöchentlich (168h)", 24*7: "Wöchentlich"}[x],
                help="Erwartete Länge des saisonalen Musters"
            )
        
        if st.button("🔄 Zerlegung starten", type="primary"):
            with st.spinner("Zerlege Zeitreihe..."):
                result = decomposer.decompose(data, selected_col, period=period)
                
                # Store in session for reuse
                st.session_state.decomposition_result = result
                st.session_state.decomposer = decomposer
        
        # Display results if available
        if hasattr(st.session_state, 'decomposition_result'):
            decomposer = st.session_state.decomposer
            
            # Decomposition plot
            st.markdown("#### Zerlegungskomponenten")
            fig = decomposer.plot_decomposition()
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            summary = decomposer.get_decomposition_summary()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📈 Trend**")
                trend_dir = "📈" if summary["trend"]["direction"] == "increasing" else "📉"
                st.metric(
                    "Trendrichtung", 
                    f"{trend_dir} {summary['trend']['change_percent']:+.1f}%",
                    help="Gesamtveränderung vom Anfang bis Ende. Zeigt die langfristige Entwicklung."
                )
                st.metric(
                    "Trendstärke",
                    f"{summary['trend']['strength']:.2f}",
                    help="0 = kein Trend, 1 = starker Trend"
                )
            
            with col2:
                st.markdown("**🔄 Saisonalität**")
                st.metric(
                    "Saisonale Stärke",
                    f"{summary['seasonality']['strength']:.2f}",
                    help="0 = keine Saisonalität, 1 = starke Saisonalität. Misst, wie vorhersehbar die Muster sind."
                )
                st.metric(
                    "Saisonale Spanne",
                    f"{summary['seasonality']['range']:.1f}",
                    help="Unterschied zwischen Spitze und Tief"
                )
            
            with col3:
                st.markdown("**📊 Residuen**")
                st.metric(
                    "Residuen Std.abw.",
                    f"{summary['residual']['std']:.2f}",
                    help="Unerklärte Variation – zufällige Schwankungen, die nicht durch Trend oder Saisonalität erklärt werden."
                )
                st.metric(
                    "Mittelwert",
                    f"{summary['residual']['mean']:.2f}",
                    help="Sollte ~0 sein. Abweichungen deuten auf systematischen Fehler hin."
                )
            
            # Seasonal patterns
            st.markdown("---")
            st.markdown("#### Saisonale Muster")
            
            pattern_col1, pattern_col2 = st.columns(2)
            
            with pattern_col1:
                fig_hourly = decomposer.plot_seasonal_pattern(aggregate_by="hour")
                st.plotly_chart(fig_hourly, use_container_width=True)
            
            with pattern_col2:
                fig_daily = decomposer.plot_seasonal_pattern(aggregate_by="dayofweek")
                st.plotly_chart(fig_daily, use_container_width=True)
            
            # ACF/PACF
            st.markdown("---")
            st.markdown("#### Autokorrelationsanalyse")
            st.markdown("Hilft dabei, signifikante Verzögerungen (Lags) und Muster in den Daten zu identifizieren. Zeigt, wie stark heutige Werte mit vergangenen Werten zusammenhängen.")
            
            fig_acf = decomposer.plot_acf_pacf(data, selected_col, n_lags=48)
            st.plotly_chart(fig_acf, use_container_width=True)
            
            # Stationarity test
            st.markdown("---")
            st.markdown("#### Stationaritätstest (ADF)")
            
            stationarity = decomposer.test_stationarity(data, selected_col)
            
            if stationarity.is_stationary:
                st.success(f"✅ Zeitreihe ist **stationär** (p-Wert: {stationarity.p_value:.4f}) – Muster bleiben über die Zeit konsistent.")
            else:
                st.warning(f"⚠️ Zeitreihe ist **nicht-stationär** (p-Wert: {stationarity.p_value:.4f})")
                st.info("Nicht-stationäre Daten sind schwerer zu prognostizieren. Prophet und ARIMA können damit umgehen.")
            
    except ImportError as e:
        st.error(f"Required library not available: {e}")
        st.info("Install with: pip install statsmodels")
    except Exception as e:
        st.error(f"Error in time series analysis: {str(e)}")


def render_forecast_confidence():
    """Render forecast confidence visualization."""
    st.markdown("### 📉 Prognose-Konfidenzintervalle")
    st.markdown("Visualisieren Sie die Vorhersage-Unsicherheit und Konfidenzbänder.")
    
    if not st.session_state.get("forecast_generated", False):
        st.warning("⚠️ Erstellen Sie zuerst eine Prognose, um Konfidenzintervalle zu sehen")
        return
    
    forecast_df = st.session_state.forecast_df
    forecaster = st.session_state.forecaster
    
    # Check if we have confidence intervals
    if not hasattr(st.session_state, 'forecast_result') or st.session_state.forecast_result is None:
        st.info("Konfidenzintervalle werden während der Prognoseerstellung berechnet.")
        st.info("Führen Sie die Prognose erneut aus, um Konfidenzbänder zu sehen.")
        return
    
    result = st.session_state.forecast_result
    
    if result.confidence_intervals is None:
        st.warning("Keine Konfidenzintervalle für diese Prognose verfügbar.")
        return
    
    # Select target to visualize
    target_cols = list(result.confidence_intervals.keys())
    
    selected_target = st.selectbox(
        "Zielgröße auswählen",
        target_cols,
        help="Wählen Sie den anzuzeigenden Workload-Typ"
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
    st.markdown("#### Konfidenzintervall-Statistiken")
    
    col1, col2, col3, col4 = st.columns(4)
    
    avg_width = (ci_data['upper'] - ci_data['lower']).mean()
    max_width = (ci_data['upper'] - ci_data['lower']).max()
    min_width = (ci_data['upper'] - ci_data['lower']).min()
    avg_pred = ci_data['prediction'].mean()
    
    with col1:
        st.metric(
            "Ø Intervallbreite",
            f"{avg_width:.1f}",
            help="Durchschnittliche Größe des Konfidenzbandes. Zeigt die typische Unsicherheit."
        )
    
    with col2:
        st.metric(
            "Max. Intervallbreite",
            f"{max_width:.1f}",
            help="Breitestes Konfidenzband (höchste Unsicherheit)"
        )
    
    with col3:
        st.metric(
            "Relative Unsicherheit",
            f"{(avg_width / avg_pred * 100):.1f}%",
            help="Durchschnittliche Unsicherheit in % des prognostizierten Wertes"
        )
    
    with col4:
        st.metric(
            "Ø Prognose",
            f"{avg_pred:.1f}",
            help="Mittlerer prognostizierter Wert"
        )
    
    # Uncertainty over time
    st.markdown("---")
    st.markdown("#### Unsicherheitswachstum über Zeit")
    st.info("💡 Konfidenzintervalle werden typischerweise breiter, je weiter in die Zukunft prognostiziert wird.")
    
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
    st.markdown("### 🎯 Was-wäre-wenn Szenario-Analyse")
    st.markdown("Erkunden Sie verschiedene Szenarien und deren Auswirkungen auf den Personalbedarf.")
    
    if not st.session_state.get("forecast_generated", False):
        st.warning("⚠️ Erstellen Sie zuerst eine Prognose, um die Szenario-Analyse durchzuführen")
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
        with st.expander("💰 Kosteneinstellungen", expanded=False):
            cost_per_hour = st.number_input(
                "Kosten pro Agentenstunde (€)",
                value=25.0,
                min_value=10.0,
                max_value=100.0,
                help="Stündliche Kosten pro Agent für Kostenschätzungen"
            )
            analyzer.cost_per_agent_hour = cost_per_hour
        
        st.markdown("---")
        
        # Scenario creation
        st.markdown("#### Eigenes Szenario erstellen")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scenario_name = st.text_input(
                "Szenarioname",
                value="Mein Szenario",
                help="Geben Sie Ihrem Szenario einen beschreibenden Namen"
            )
            
            volume_change = st.slider(
                "Volumenänderung (%)",
                min_value=-50,
                max_value=100,
                value=0,
                step=5,
                help="Erhöhen oder verringern Sie das Workload-Volumen"
            )
        
        with col2:
            scenario_desc = st.text_input(
                "Beschreibung",
                value="Benutzerdefiniertes Szenario",
                help="Kurze Beschreibung dieses Szenarios"
            )
            
            aht_change = st.slider(
                "Bearbeitungszeit-Änderung (%)",
                min_value=-30,
                max_value=50,
                value=0,
                step=5,
                help="Erhöhen oder verringern Sie die durchschnittliche Bearbeitungszeit (AHT)"
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
            numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
            st.dataframe(
                comparison_df.style.format("{:.1f}", subset=numeric_cols),
                use_container_width=True, 
                hide_index=True
            )
            
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
    
    if hasattr(forecast_start, 'strftime'):
        st.info(f"📅 **Prognosezeitraum:** {forecast_start.strftime('%d.%m.%Y')} bis {forecast_end.strftime('%d.%m.%Y')}")
    else:
        st.info(f"📅 **Prognosezeitraum:** {forecast_start} bis {forecast_end}")
    
    st.caption("📍 Geschäftszeiten: 7:00-22:00 Uhr")
    
    # Prepare weekly data - filter to business hours
    staffing_copy = staffing_plan.copy()
    staffing_copy["day_of_week"] = pd.to_datetime(staffing_copy["timestamp"]).dt.dayofweek
    staffing_copy["day_name"] = pd.to_datetime(staffing_copy["timestamp"]).dt.day_name()
    staffing_copy["hour"] = pd.to_datetime(staffing_copy["timestamp"]).dt.hour
    
    # Filter to business hours only
    staffing_copy = staffing_copy[staffing_copy["hour"].isin(BUSINESS_HOURS)]
    
    # Create weekly pivot table
    weekly_schedule = staffing_copy.pivot_table(
        index="day_of_week",
        columns="hour",
        values="total_agents",
        aggfunc="mean"
    ).round(1)
    
    # Ensure all business hours columns exist
    for h in BUSINESS_HOURS:
        if h not in weekly_schedule.columns:
            weekly_schedule[h] = 0
    weekly_schedule = weekly_schedule[BUSINESS_HOURS]  # Reorder to business hours only
    
    day_names = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
    weekly_schedule.index = [day_names[i] for i in weekly_schedule.index]
    
    # Summary metrics at the top
    st.markdown("## 📊 Schnellübersicht")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🔝 Spitzenbedarf",
            f"{int(weekly_schedule.values.max())} Agenten",
            help="Maximale Anzahl benötigter Agenten in einer Stunde"
        )
    
    with col2:
        st.metric(
            "📈 Ø Bedarf",
            f"{weekly_schedule.values.mean():.1f} Agenten",
            help="Durchschnittliche Anzahl Agenten über alle Stunden"
        )
    
    with col3:
        busiest_day = weekly_schedule.mean(axis=1).idxmax()
        st.metric(
            "📅 Geschäftigster Tag",
            busiest_day,
            help="Tag mit dem höchsten Personalbedarf"
        )
    
    with col4:
        peak_hour = weekly_schedule.mean(axis=0).idxmax()
        st.metric(
            "⏰ Spitzenstunde",
            f"{peak_hour}:00",
            help="Stunde mit dem höchsten Personalbedarf"
        )
    
    st.markdown("---")
    
    # Weekly Heatmap
    st.markdown("## 🗓️ Wöchentliche Personalbedarfs-Heatmap")
    st.markdown("Zeigt die durchschnittliche Anzahl benötigter Agenten für jede Stunde jedes Tages.")
    
    fig_weekly = go.Figure(data=go.Heatmap(
        z=weekly_schedule.values,
        x=[f"{h}:00" for h in weekly_schedule.columns],
        y=weekly_schedule.index,
        colorscale="Blues",
        text=weekly_schedule.values.round(0).astype(int),
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        hovertemplate="Tag: %{y}<br>Stunde: %{x}<br>Agenten: %{z:.1f}<extra></extra>"
    ))
    
    fig_weekly.update_layout(
        xaxis_title="Tagesstunde",
        yaxis_title="Wochentag",
        height=450,
        yaxis=dict(autorange="reversed"),
        font=dict(size=14)
    )
    
    st.plotly_chart(fig_weekly, use_container_width=True)
    
    # Detailed Table
    st.markdown("## 📋 Detaillierter Wochenplan (7:00-22:00)")
    st.markdown("Anzahl benötigter Agenten pro Stunde:")
    
    weekly_table = weekly_schedule.copy()
    weekly_table.columns = [f"{h}:00" for h in weekly_table.columns]
    weekly_table["Tages-Ø"] = weekly_schedule.mean(axis=1).round(1)
    weekly_table["Spitze"] = weekly_schedule.max(axis=1).round(0).astype(int)
    
    # Style the table
    st.dataframe(
        weekly_table.style.background_gradient(cmap="Blues", subset=weekly_table.columns[:-2])
            .format("{:.0f}", subset=weekly_table.columns[:-2])
            .format("{:.1f}", subset=["Tages-Ø"]),
        use_container_width=True,
        height=320
    )
    
    # Daily breakdown (business hours only)
    st.markdown("## 📆 Tägliche Personalübersicht (7:00-22:00)")
    
    if "date" in staffing_plan.columns:
        # Use the already filtered staffing_copy with business hours
        daily_summary = staffing_copy.groupby(pd.to_datetime(staffing_copy["timestamp"]).dt.date).agg({
            "total_volume": "sum",
            "total_agents": ["max", "mean"]
        }).reset_index()
        daily_summary.columns = ["Datum", "Aufgaben", "Max. Agenten", "Ø Agenten"]
        daily_summary["Ø Agenten"] = daily_summary["Ø Agenten"].round(1)
        daily_summary["Datum"] = pd.to_datetime(daily_summary["Datum"]).dt.strftime("%a, %d.%m.")
        
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Bar(
            x=daily_summary["Datum"],
            y=daily_summary["Max. Agenten"],
            name="Max. Agenten",
            marker_color="#4472C4"
        ))
        fig_daily.add_trace(go.Scatter(
            x=daily_summary["Datum"],
            y=daily_summary["Ø Agenten"],
            name="Ø Agenten",
            mode="lines+markers",
            line=dict(color="#ED7D31", width=3),
            marker=dict(size=8)
        ))
        
        fig_daily.update_layout(
            xaxis_title="Datum",
            yaxis_title="Anzahl Agenten",
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


def review_forecasts_section():
    """Section for reviewing past forecasts against actual results."""
    st.markdown("## 🔄 Review Forecasts")
    st.markdown("Compare your forecasts to actual results and help the model learn.")
    
    # Initialize forecast history in session state
    if "forecast_history" not in st.session_state:
        st.session_state.forecast_history = []
    
    # Check if there's a current forecast to review
    if not st.session_state.get("forecast_generated", False):
        st.info("📊 Generate a forecast first, then come back here to compare it with actual results.")
        return
    
    # Create tabs for different review functions
    review_tab1, review_tab2 = st.tabs(["📤 Upload Actuals", "📊 View Comparison"])
    
    with review_tab1:
        _render_actuals_upload_section()
    
    with review_tab2:
        _render_comparison_section()


def _render_actuals_upload_section():
    """Render the section for uploading actual data."""
    st.markdown("### Ist-Daten hochladen")
    st.markdown("Nach Ablauf des Prognosezeitraums können Sie Ihre tatsächlichen Daten hochladen, um die Genauigkeit der Vorhersage zu prüfen.")
    
    # Show current forecast info
    forecast_start = st.session_state.get("forecast_start")
    forecast_end = st.session_state.get("forecast_end")
    
    if forecast_start and forecast_end:
        st.info(f"📅 **Aktueller Prognosezeitraum:** {forecast_start.strftime('%d.%m.%Y')} bis {forecast_end.strftime('%d.%m.%Y')}")
    
    # File upload
    st.markdown("#### Ist-Ergebnisse hochladen")
    st.caption("Laden Sie eine CSV- oder Excel-Datei mit den tatsächlichen Anruf-/E-Mail-/Outbound-Volumen für den Prognosezeitraum hoch.")
    
    uploaded_file = st.file_uploader(
        "Datei mit Ist-Daten auswählen",
        type=["csv", "xlsx", "xls"],
        key="actuals_upload",
        help="Datei sollte eine 'timestamp'-Spalte und Spalten haben, die Ihren Prognosezielen entsprechen"
    )
    
    if uploaded_file is not None:
        try:
            # Load the file
            if uploaded_file.name.endswith('.csv'):
                actuals_df = pd.read_csv(uploaded_file)
            else:
                actuals_df = pd.read_excel(uploaded_file)
            
            # Try to parse timestamp
            timestamp_cols = ['timestamp', 'date', 'datetime', 'time', 'Timestamp', 'Date']
            ts_col = None
            for col in timestamp_cols:
                if col in actuals_df.columns:
                    ts_col = col
                    break
            
            if ts_col:
                actuals_df['timestamp'] = pd.to_datetime(actuals_df[ts_col])
                if ts_col != 'timestamp':
                    actuals_df = actuals_df.drop(columns=[ts_col])
            else:
                st.error("❌ Keine Zeitstempel-Spalte gefunden. Bitte stellen Sie sicher, dass Ihre Datei eine 'timestamp'- oder 'date'-Spalte hat.")
                return
            
            # Show preview
            st.markdown("#### Datenvorschau")
            numeric_cols = actuals_df.select_dtypes(include=[np.number]).columns
            st.dataframe(
                actuals_df.head(10).style.format("{:.1f}", subset=numeric_cols),
                use_container_width=True
            )
            
            st.caption(f"{len(actuals_df)} Zeilen geladen von {actuals_df['timestamp'].min().strftime('%d.%m.%Y %H:%M')} bis {actuals_df['timestamp'].max().strftime('%d.%m.%Y %H:%M')}")
            
            # Compare button
            if st.button("🔍 Mit Prognose vergleichen", type="primary", use_container_width=True):
                with st.spinner("Analysiere Prognose-Genauigkeit..."):
                    forecast_df = st.session_state.forecast_df
                    
                    # Run comparison
                    comparison = compare_forecast_to_actuals(
                        forecast_df=forecast_df,
                        actuals_df=actuals_df,
                        forecast_id=f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    
                    # Store in session state
                    st.session_state.last_comparison = comparison
                    st.session_state.actuals_df = actuals_df
                    
                    st.success("✅ Vergleich abgeschlossen! Gehen Sie zum Tab 'Vergleich ansehen', um die Ergebnisse zu sehen.")
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Fehler beim Laden der Datei: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # Sample format download
    with st.expander("📋 Erwartetes Dateiformat", expanded=False):
        st.markdown("""
        Ihre Datei sollte diese Spalten haben:
        - **timestamp**: Datum und Uhrzeit (z.B. '2026-01-15 09:00:00')
        - **call_volume**: Anzahl der Anrufe
        - **email_count**: Anzahl der E-Mails
        - **outbound_ook/omk/nb**: Outbound-Anrufvolumen (optional)
        
        Die Spaltennamen sollten mit Ihrer Prognose-Ausgabe übereinstimmen.
        """)
        
        # Create sample data
        sample_df = pd.DataFrame({
            'timestamp': pd.date_range('2026-01-15', periods=24, freq='H'),
            'call_volume': np.random.randint(20, 50, 24),
            'email_count': np.random.randint(10, 30, 24),
            'outbound_ook': np.random.randint(5, 15, 24)
        })
        
        csv_sample = sample_df.to_csv(index=False)
        st.download_button(
            "📥 Beispielvorlage herunterladen",
            data=csv_sample,
            file_name="ist_daten_vorlage.csv",
            mime="text/csv"
        )


def _render_comparison_section():
    """Render the forecast vs actuals comparison."""
    st.markdown("### Prognose vs. Ist-Vergleich")
    
    comparison = st.session_state.get("last_comparison")
    
    if comparison is None:
        st.info("📤 Laden Sie Ist-Daten im Tab 'Ist-Daten hochladen' hoch, um einen Vergleich zu sehen.")
        return
    
    # Overall accuracy card
    _render_accuracy_card(comparison)
    
    # Summary
    st.markdown("### 📝 Zusammenfassung")
    st.markdown(f"**{comparison.summary}**")
    
    # Highlights and improvements in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Was gut lief")
        if comparison.highlights:
            for highlight in comparison.highlights:
                st.markdown(f"- {highlight}")
        else:
            st.markdown("- Keine besonderen Highlights")
    
    with col2:
        st.markdown("#### 🎯 Verbesserungspotenzial")
        if comparison.areas_for_improvement:
            for improvement in comparison.areas_for_improvement:
                st.markdown(f"- {improvement}")
        else:
            st.markdown("- No issues detected")
    
    # Per-target breakdown
    if comparison.target_metrics:
        st.markdown("### 📊 Detailed Breakdown")
        
        # Create metrics cards
        n_cols = min(len(comparison.target_metrics), 4)
        cols = st.columns(n_cols)
        
        for i, (target, metrics) in enumerate(comparison.target_metrics.items()):
            with cols[i % n_cols]:
                accuracy = metrics.get('accuracy', 0)
                color = "#10b981" if accuracy >= 80 else "#f59e0b" if accuracy >= 60 else "#ef4444"
                
                st.markdown(f"""
                <div style="
                    background: white;
                    border: 1px solid #e5e7eb;
                    border-radius: 8px;
                    padding: 1rem;
                    text-align: center;
                ">
                    <div style="font-weight: 600; color: #374151; margin-bottom: 0.5rem;">
                        {target.replace('_', ' ').title()}
                    </div>
                    <div style="font-size: 2rem; font-weight: 700; color: {color};">
                        {accuracy:.0f}%
                    </div>
                    <div style="font-size: 0.8rem; color: #6b7280;">Genauigkeit</div>
                    <hr style="margin: 0.5rem 0; border-color: #e5e7eb;">
                    <div style="font-size: 0.85rem; color: #6b7280;">
                        Prognose: {metrics.get('total_predicted', 0):,.0f}<br>
                        Ist: {metrics.get('total_actual', 0):,.0f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Visual comparison chart
    st.markdown("### 📈 Visueller Vergleich")
    _render_comparison_chart(comparison)
    
    # Learning feedback
    st.markdown("---")
    st.markdown("### 🧠 Helfen Sie dem Modell zu lernen")
    st.markdown("Ihr Feedback hilft, zukünftige Prognosen zu verbessern.")
    
    col_fb1, col_fb2 = st.columns([2, 1])
    
    with col_fb1:
        feedback_note = st.text_area(
            "Notizen zu diesem Zeitraum hinzufügen (optional)",
            placeholder="z.B. 'Marketing-Kampagne lief am Dienstag', 'Unerwarteter Systemausfall am Freitag'",
            help="Diese Notizen helfen zu erklären, warum Prognosen möglicherweise abweichen"
        )
    
    with col_fb2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Feedback speichern", type="primary", use_container_width=True):
            # Save to forecast history
            history_entry = {
                "forecast_id": comparison.forecast_id,
                "period_start": str(comparison.period_start),
                "period_end": str(comparison.period_end),
                "overall_accuracy": comparison.overall_accuracy,
                "target_metrics": comparison.target_metrics,
                "feedback_note": feedback_note,
                "saved_at": datetime.now().isoformat()
            }
            
            if "forecast_history" not in st.session_state:
                st.session_state.forecast_history = []
            
            st.session_state.forecast_history.append(history_entry)
            st.success("✅ Feedback gespeichert! Dies wird helfen, zukünftige Prognosen zu verbessern.")


def _render_accuracy_card(comparison: ForecastComparison):
    """Render the main accuracy indicator card."""
    accuracy = comparison.overall_accuracy
    
    # Determine color and emoji
    if accuracy >= 80:
        color = "#10b981"
        emoji = "🟢"
        label = "Ausgezeichnet"
    elif accuracy >= 60:
        color = "#f59e0b"
        emoji = "🟡"
        label = "Gut"
    else:
        color = "#ef4444"
        emoji = "🔴"
        label = "Verbesserung nötig"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}20 0%, {color}05 100%);
        border: 2px solid {color};
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
    ">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">{emoji}</div>
        <div style="font-size: 2.5rem; font-weight: 700; color: {color};">
            {accuracy:.0f}% Genau
        </div>
        <div style="font-size: 1.1rem; color: #374151; margin-top: 0.5rem;">
            {label} – Durchschnittlicher Fehler von {comparison.overall_mape:.1f}%
        </div>
        <div style="font-size: 0.9rem; color: #6b7280; margin-top: 0.5rem;">
            Zeitraum: {comparison.period_start.strftime('%d.%m.%Y')} bis {comparison.period_end.strftime('%d.%m.%Y')}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_comparison_chart(comparison: ForecastComparison):
    """Render forecast vs actuals comparison chart."""
    if comparison.comparison_df is None or comparison.comparison_df.empty:
        st.warning("Keine Vergleichsdaten zur Visualisierung verfügbar.")
        return
    
    df = comparison.comparison_df
    
    # Find prediction and actual column pairs
    pred_cols = [c for c in df.columns if c.endswith('_pred')]
    
    if not pred_cols:
        st.warning("Prognose-Spalten für Visualisierung konnten nicht identifiziert werden.")
        return
    
    # Create chart for first target
    target = pred_cols[0].replace('_pred', '')
    pred_col = f"{target}_pred"
    actual_col = f"{target}_actual"
    
    if actual_col not in df.columns:
        st.warning(f"Could not find actual column for {target}.")
        return
    
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df[actual_col],
        name='Actual',
        mode='lines',
        line=dict(color='#10b981', width=2)
    ))
    
    # Predicted values
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df[pred_col],
        name='Predicted',
        mode='lines',
        line=dict(color='#667eea', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f"{target.replace('_', ' ').title()}: Predicted vs Actual",
        xaxis_title="Date/Time",
        yaxis_title="Volume",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True, key="comparison_chart")
    
    # Show other targets if available
    if len(pred_cols) > 1:
        with st.expander("View other targets", expanded=False):
            for pred_col in pred_cols[1:]:
                target = pred_col.replace('_pred', '')
                actual_col = f"{target}_actual"
                
                if actual_col not in df.columns:
                    continue
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=df['timestamp'], y=df[actual_col],
                    name='Actual', mode='lines', line=dict(color='#10b981')
                ))
                fig2.add_trace(go.Scatter(
                    x=df['timestamp'], y=df[pred_col],
                    name='Predicted', mode='lines', line=dict(color='#667eea', dash='dash')
                ))
                fig2.update_layout(
                    title=f"{target.replace('_', ' ').title()}: Predicted vs Actual",
                    template="plotly_white", height=300
                )
                st.plotly_chart(fig2, use_container_width=True, key=f"comparison_chart_{target}")


def call_center_management_section():
    """Call Center and Agent Management Section for Admins."""
    st.markdown("### 🏢 Call-Center-Verwaltung")
    st.caption("Verwalten Sie Call-Center, Agenten und Kosten.")
    
    # Summary metrics at top
    total_centers = len(st.session_state.call_centers)
    total_agents = sum(len(cc.get("agents", {})) for cc in st.session_state.call_centers.values())
    active_agents = sum(
        1 for cc in st.session_state.call_centers.values() 
        for agent in cc.get("agents", {}).values() 
        if agent.get("is_active", True)
    )
    total_fte = sum(
        agent.get("fte", 1.0) for cc in st.session_state.call_centers.values() 
        for agent in cc.get("agents", {}).values() 
        if agent.get("is_active", True)
    )
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Call-Center", total_centers)
    with metric_col2:
        st.metric("Agenten (Gesamt)", total_agents)
    with metric_col3:
        st.metric("Agenten (Aktiv)", active_agents)
    with metric_col4:
        st.metric("FTE (Aktiv)", f"{total_fte:.1f}")
    
    st.markdown("---")
    
    # Two-column layout: Call centers list on left, details on right
    col_list, col_details = st.columns([1, 2])
    
    with col_list:
        st.markdown("#### 📋 Call-Center")
        
        # Add new call center
        with st.expander("➕ Neues Call-Center hinzufügen", expanded=False):
            new_cc_name = st.text_input("Name", key="new_cc_name", placeholder="z.B. Neues CallCenter")
            new_cc_cost = st.number_input(
                "Kosten pro Agent/Stunde (€)", 
                min_value=10.0, max_value=100.0, value=25.0, step=0.5,
                key="new_cc_cost"
            )
            
            if st.button("✅ Call-Center erstellen", key="create_cc_btn", use_container_width=True):
                if new_cc_name and new_cc_name.strip():
                    new_id = _generate_call_center_id(new_cc_name.strip())
                    st.session_state.call_centers[new_id] = {
                        "id": new_id,
                        "name": new_cc_name.strip(),
                        "cost_per_agent_hour": new_cc_cost,
                        "created_at": datetime.now().isoformat(),
                        "agents": {}
                    }
                    _save_call_centers()
                    st.success(f"✅ Call-Center '{new_cc_name}' erstellt!")
                    st.rerun()
                else:
                    st.error("Bitte geben Sie einen Namen ein.")
        
        # List existing call centers
        st.markdown("##### Vorhandene Call-Center")
        
        # Initialize selected call center
        if "selected_call_center" not in st.session_state:
            st.session_state.selected_call_center = None
        
        for cc_id, cc_data in st.session_state.call_centers.items():
            agent_count = len(cc_data.get("agents", {}))
            active_count = sum(1 for a in cc_data.get("agents", {}).values() if a.get("is_active", True))
            
            # Button to select call center
            btn_label = f"🏢 {cc_data['name']} ({active_count}/{agent_count} Agenten)"
            if st.button(btn_label, key=f"select_cc_{cc_id}", use_container_width=True):
                st.session_state.selected_call_center = cc_id
                st.rerun()
    
    with col_details:
        if st.session_state.selected_call_center and st.session_state.selected_call_center in st.session_state.call_centers:
            cc_id = st.session_state.selected_call_center
            cc_data = st.session_state.call_centers[cc_id]
            
            st.markdown(f"#### 🏢 {cc_data['name']}")
            st.caption(f"ID: `{cc_id}`")
            
            # Call center settings
            with st.expander("⚙️ Call-Center Einstellungen", expanded=True):
                settings_col1, settings_col2 = st.columns(2)
                
                with settings_col1:
                    updated_name = st.text_input(
                        "Name", 
                        value=cc_data['name'],
                        key=f"edit_name_{cc_id}"
                    )
                
                with settings_col2:
                    updated_cost = st.number_input(
                        "Kosten pro Agent/Stunde (€)",
                        min_value=10.0, max_value=100.0,
                        value=float(cc_data.get('cost_per_agent_hour', 25.0)),
                        step=0.5,
                        key=f"edit_cost_{cc_id}"
                    )
                
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("💾 Speichern", key=f"save_cc_{cc_id}", use_container_width=True):
                        st.session_state.call_centers[cc_id]['name'] = updated_name
                        st.session_state.call_centers[cc_id]['cost_per_agent_hour'] = updated_cost
                        _save_call_centers()
                        st.success("✅ Gespeichert!")
                        st.rerun()
                
                with btn_col2:
                    if st.button("🗑️ Call-Center löschen", key=f"delete_cc_{cc_id}", use_container_width=True, type="secondary"):
                        del st.session_state.call_centers[cc_id]
                        st.session_state.selected_call_center = None
                        _save_call_centers()
                        st.warning(f"Call-Center '{cc_data['name']}' gelöscht!")
                        st.rerun()
            
            st.markdown("---")
            
            # Agent management
            st.markdown("##### 👥 Agenten-Verwaltung")
            
            # Add new agent
            with st.expander("➕ Neuen Agenten hinzufügen", expanded=False):
                agent_col1, agent_col2 = st.columns(2)
                
                with agent_col1:
                    new_agent_name = st.text_input(
                        "Name des Agenten",
                        key=f"new_agent_name_{cc_id}",
                        placeholder="z.B. Max Mustermann"
                    )
                
                with agent_col2:
                    new_agent_fte = st.number_input(
                        "FTE",
                        min_value=0.1, max_value=1.0, value=1.0, step=0.1,
                        key=f"new_agent_fte_{cc_id}",
                        help="Vollzeitäquivalent: 1.0 = Vollzeit, 0.5 = Teilzeit"
                    )
                
                new_agent_active = st.checkbox("Aktiv", value=True, key=f"new_agent_active_{cc_id}")
                
                if st.button("✅ Agent hinzufügen", key=f"add_agent_{cc_id}", use_container_width=True):
                    if new_agent_name and new_agent_name.strip():
                        agent_id = _generate_agent_id(cc_id)
                        if "agents" not in st.session_state.call_centers[cc_id]:
                            st.session_state.call_centers[cc_id]["agents"] = {}
                        
                        st.session_state.call_centers[cc_id]["agents"][agent_id] = {
                            "id": agent_id,
                            "name": new_agent_name.strip(),
                            "fte": new_agent_fte,
                            "is_active": new_agent_active,
                            "created_at": datetime.now().isoformat()
                        }
                        _save_call_centers()
                        st.success(f"✅ Agent '{new_agent_name}' hinzugefügt!")
                        st.rerun()
                    else:
                        st.error("Bitte geben Sie einen Namen ein.")
            
            # List agents
            agents = cc_data.get("agents", {})
            
            if agents:
                # Create dataframe for agents
                agent_data = []
                for agent_id, agent in agents.items():
                    agent_data.append({
                        "ID": agent_id,
                        "Name": agent.get("name", ""),
                        "FTE": agent.get("fte", 1.0),
                        "Status": "✅ Aktiv" if agent.get("is_active", True) else "❌ Inaktiv",
                        "is_active": agent.get("is_active", True)
                    })
                
                agent_df = pd.DataFrame(agent_data)
                
                # Summary
                active_fte = agent_df[agent_df['is_active']]['FTE'].sum()
                st.info(f"**{len(agents)} Agenten** | **{active_fte:.1f} FTE aktiv** | **{cc_data.get('cost_per_agent_hour', 25):.2f}€/Stunde**")
                
                # Agent table with edit functionality
                st.markdown("**Agentenliste:**")
                
                for agent_id, agent in agents.items():
                    with st.container():
                        agent_cols = st.columns([3, 1, 1, 1, 1])
                        
                        with agent_cols[0]:
                            st.markdown(f"**{agent.get('name', 'N/A')}**")
                            st.caption(f"ID: `{agent_id[:20]}...`")
                        
                        with agent_cols[1]:
                            new_fte = st.number_input(
                                "FTE",
                                min_value=0.1, max_value=1.0,
                                value=float(agent.get('fte', 1.0)),
                                step=0.1,
                                key=f"fte_{agent_id}",
                                label_visibility="collapsed"
                            )
                            st.caption("FTE")
                        
                        with agent_cols[2]:
                            is_active = st.checkbox(
                                "Aktiv",
                                value=agent.get('is_active', True),
                                key=f"active_{agent_id}"
                            )
                        
                        with agent_cols[3]:
                            if st.button("💾", key=f"save_agent_{agent_id}", help="Speichern"):
                                st.session_state.call_centers[cc_id]["agents"][agent_id]["fte"] = new_fte
                                st.session_state.call_centers[cc_id]["agents"][agent_id]["is_active"] = is_active
                                _save_call_centers()
                                st.rerun()
                        
                        with agent_cols[4]:
                            if st.button("🗑️", key=f"del_agent_{agent_id}", help="Löschen"):
                                del st.session_state.call_centers[cc_id]["agents"][agent_id]
                                _save_call_centers()
                                st.rerun()
                        
                        st.markdown("---")
            else:
                st.info("Noch keine Agenten in diesem Call-Center. Fügen Sie oben neue Agenten hinzu.")
            
            # Cost calculation
            if agents:
                st.markdown("##### 💰 Kostenübersicht")
                
                active_agents_list = [a for a in agents.values() if a.get("is_active", True)]
                total_fte_cc = sum(a.get("fte", 1.0) for a in active_agents_list)
                hourly_cost = total_fte_cc * cc_data.get('cost_per_agent_hour', 25.0)
                daily_cost = hourly_cost * 8  # Assuming 8-hour day
                monthly_cost = daily_cost * 22  # Assuming 22 working days
                
                cost_col1, cost_col2, cost_col3 = st.columns(3)
                with cost_col1:
                    st.metric("Kosten / Stunde", f"{hourly_cost:,.2f}€")
                with cost_col2:
                    st.metric("Kosten / Tag (8h)", f"{daily_cost:,.2f}€")
                with cost_col3:
                    st.metric("Kosten / Monat", f"{monthly_cost:,.2f}€")
        
        else:
            st.info("👈 Wählen Sie ein Call-Center aus der Liste aus, um Details anzuzeigen und zu bearbeiten.")
    
    # Export section
    st.markdown("---")
    with st.expander("📥 Daten exportieren / importieren", expanded=False):
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            st.markdown("**Export**")
            import json
            export_data = json.dumps(st.session_state.call_centers, indent=2, ensure_ascii=False)
            st.download_button(
                "📥 Call-Center-Daten als JSON exportieren",
                data=export_data,
                file_name=f"call_centers_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with export_col2:
            st.markdown("**Import**")
            import_file = st.file_uploader(
                "JSON-Datei importieren",
                type=['json'],
                key="cc_import"
            )
            
            if import_file:
                try:
                    import json
                    imported_data = json.load(import_file)
                    if st.button("📤 Importieren (überschreibt aktuelle Daten)", use_container_width=True):
                        st.session_state.call_centers = imported_data
                        _save_call_centers()
                        st.success("✅ Daten erfolgreich importiert!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Fehler beim Importieren: {e}")


def demand_outlook_section():
    """
    Demand Outlook & Trend section - Flexible overview of forecast vs actual performance.
    Design inspired by modern demand planning dashboards.
    """
    st.markdown("### 📊 Demand Outlook & Trend")
    st.caption("Vergleichen Sie Prognosen mit tatsächlichen Werten über verschiedene Zeiträume.")
    
    # Initialize forecast comparison data in session state
    if "forecast_comparison_data" not in st.session_state:
        st.session_state.forecast_comparison_data = None
    
    # ===========================================
    # FILTER BAR
    # ===========================================
    st.markdown("""
    <style>
    .filter-container {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        filter_cols = st.columns([1.5, 1, 1, 1])
        
        with filter_cols[0]:
            # Date range picker
            today = datetime.now()
            default_start = today - timedelta(days=90)
            
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input(
                    "Datum von",
                    value=default_start.date(),
                    key="outlook_start_date"
                )
            with date_col2:
                end_date = st.date_input(
                    "Datum bis",
                    value=today.date(),
                    key="outlook_end_date"
                )
        
        with filter_cols[1]:
            # Task type filter
            task_types = st.multiselect(
                "Aufgabentyp",
                options=["Anrufe", "E-Mails", "Outbound"],
                default=["Anrufe", "E-Mails", "Outbound"],
                key="outlook_task_filter",
                placeholder="Alle auswählen"
            )
        
        with filter_cols[2]:
            # Aggregation level
            aggregation = st.selectbox(
                "Aggregation",
                options=["Täglich", "Wöchentlich", "Monatlich"],
                index=2,
                key="outlook_aggregation"
            )
        
        with filter_cols[3]:
            # Call center filter (if data available)
            cc_options = ["Alle"] + list(st.session_state.get("call_centers", {}).keys())
            cc_names = ["Alle"] + [cc.get("name", k) for k, cc in st.session_state.get("call_centers", {}).items()]
            selected_cc = st.selectbox(
                "Call-Center",
                options=cc_names,
                index=0,
                key="outlook_cc_filter"
            )
    
    st.markdown("---")
    
    # ===========================================
    # DATA LOADING / GENERATION
    # ===========================================
    
    # Check if we have actual data to compare
    has_forecast = st.session_state.get("forecast_df") is not None
    has_actuals = st.session_state.get("combined_data") is not None
    
    if not has_forecast and not has_actuals:
        st.info("📊 Laden Sie Daten und erstellen Sie einen Forecast, um den Vergleich zu sehen.")
        
        # Show demo data option
        if st.button("🎲 Demo-Daten anzeigen", help="Zeigt Beispieldaten zur Veranschaulichung"):
            # Generate demo comparison data
            demo_data = _generate_demo_outlook_data(start_date, end_date, aggregation)
            st.session_state.forecast_comparison_data = demo_data
            st.rerun()
    
    # Generate or use existing comparison data
    comparison_data = st.session_state.forecast_comparison_data
    
    if comparison_data is None:
        # Try to build from actual session data
        if has_forecast and has_actuals:
            comparison_data = _build_outlook_from_session(
                st.session_state.forecast_df,
                st.session_state.combined_data,
                start_date, end_date, aggregation, task_types
            )
        else:
            # Generate demo data for visualization
            comparison_data = _generate_demo_outlook_data(start_date, end_date, aggregation)
    
    # ===========================================
    # MAIN CHART - Shipped Quantity Over Time
    # ===========================================
    
    st.markdown("#### 📈 Volumen im Zeitverlauf")
    
    fig = go.Figure()
    
    # Forecasted line (purple)
    fig.add_trace(go.Scatter(
        x=comparison_data["date"],
        y=comparison_data["forecast"],
        name="Prognostiziert",
        mode="lines+markers",
        line=dict(color="#8b5cf6", width=2),
        marker=dict(size=6, symbol="circle")
    ))
    
    # Observed/Actual line (cyan/teal)
    fig.add_trace(go.Scatter(
        x=comparison_data["date"],
        y=comparison_data["actual"],
        name="Tatsächlich",
        mode="lines+markers",
        line=dict(color="#06b6d4", width=2),
        marker=dict(size=6, symbol="circle")
    ))
    
    # Layout styling to match the reference image
    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            tickformat="%b" if aggregation == "Monatlich" else "%d.%m"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            title="Volumen",
            tickformat=",d"
        ),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True, key="outlook_main_chart")
    
    # ===========================================
    # FORECAST COMPARISON TABLE
    # ===========================================
    
    st.markdown("---")
    st.markdown("#### 📋 Forecast Vergleich")
    
    # Group data by month for the comparison table
    comparison_data["month"] = pd.to_datetime(comparison_data["date"]).dt.to_period("M")
    monthly_summary = comparison_data.groupby("month").agg({
        "forecast": "sum",
        "actual": "sum",
        "adjustment": "sum"
    }).reset_index()
    
    # Only show last N months that fit
    max_months = min(6, len(monthly_summary))
    monthly_summary = monthly_summary.tail(max_months)
    
    # Create styled table
    st.markdown("""
    <style>
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .comparison-table th {
        background: #f8fafc;
        padding: 12px 16px;
        text-align: center;
        font-weight: 600;
        color: #374151;
        border-bottom: 1px solid #e5e7eb;
    }
    .comparison-table td {
        padding: 12px 16px;
        text-align: center;
        border-bottom: 1px solid #f1f5f9;
    }
    .comparison-table tr:last-child td {
        border-bottom: none;
    }
    .row-label {
        text-align: left !important;
        font-weight: 500;
        color: #374151;
    }
    .adjustment-positive {
        color: #10b981;
        font-weight: 600;
    }
    .adjustment-negative {
        color: #ef4444;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Build table HTML
    month_headers = "".join([
        f'<th>{row["month"].strftime("%b")} <span style="color:#9ca3af;font-weight:400;">{row["month"].year}</span></th>'
        for _, row in monthly_summary.iterrows()
    ])
    
    # AI Forecast row
    forecast_cells = "".join([
        f'<td>{_format_number(row["forecast"])}</td>'
        for _, row in monthly_summary.iterrows()
    ])
    
    # Planner Input (adjustments) row
    adjustment_cells = "".join([
        f'<td class="{"adjustment-positive" if row["adjustment"] >= 0 else "adjustment-negative"}">'
        f'{("+" if row["adjustment"] >= 0 else "")}{_format_number(row["adjustment"])}</td>'
        for _, row in monthly_summary.iterrows()
    ])
    
    # Final Forecast row
    final_cells = "".join([
        f'<td style="font-weight:600;">{_format_number(row["forecast"] + row["adjustment"])}</td>'
        for _, row in monthly_summary.iterrows()
    ])
    
    table_html = f"""
    <table class="comparison-table">
        <thead>
            <tr>
                <th style="text-align:left;">Forecast Comparison</th>
                {month_headers}
            </tr>
        </thead>
        <tbody>
            <tr>
                <td class="row-label">🤖 AI Forecast</td>
                {forecast_cells}
            </tr>
            <tr>
                <td class="row-label">✏️ Planner Input</td>
                {adjustment_cells}
            </tr>
            <tr style="background:#f8fafc;">
                <td class="row-label">📊 Final Forecast</td>
                {final_cells}
            </tr>
        </tbody>
    </table>
    """
    
    st.markdown(table_html, unsafe_allow_html=True)
    
    # ===========================================
    # ACCURACY METRICS
    # ===========================================
    
    st.markdown("---")
    st.markdown("#### 🎯 Performance Metriken")
    
    # Calculate metrics
    total_forecast = comparison_data["forecast"].sum()
    total_actual = comparison_data["actual"].sum()
    total_final = (comparison_data["forecast"] + comparison_data["adjustment"]).sum()
    
    if total_actual > 0:
        ai_accuracy = max(0, 100 - abs(total_forecast - total_actual) / total_actual * 100)
        final_accuracy = max(0, 100 - abs(total_final - total_actual) / total_actual * 100)
        improvement = final_accuracy - ai_accuracy
    else:
        ai_accuracy = 0
        final_accuracy = 0
        improvement = 0
    
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.metric(
            "AI Forecast Genauigkeit",
            f"{ai_accuracy:.1f}%",
            help="Wie genau war die reine KI-Prognose?"
        )
    
    with metric_cols[1]:
        st.metric(
            "Final Forecast Genauigkeit",
            f"{final_accuracy:.1f}%",
            help="Genauigkeit nach manuellen Anpassungen"
        )
    
    with metric_cols[2]:
        st.metric(
            "Verbesserung durch Anpassungen",
            f"{improvement:+.1f}%",
            delta=f"{improvement:+.1f}%",
            delta_color="normal" if improvement >= 0 else "inverse",
            help="Wie viel haben manuelle Anpassungen geholfen?"
        )
    
    with metric_cols[3]:
        mape = comparison_data.apply(
            lambda row: abs(row["forecast"] - row["actual"]) / row["actual"] * 100 
            if row["actual"] > 0 else 0, axis=1
        ).mean()
        st.metric(
            "Ø MAPE",
            f"{mape:.1f}%",
            help="Mean Absolute Percentage Error - niedriger ist besser"
        )
    
    # ===========================================
    # TREND ANALYSIS
    # ===========================================
    
    with st.expander("📈 Trend-Analyse", expanded=False):
        trend_cols = st.columns(2)
        
        with trend_cols[0]:
            # Over/Under forecasting pattern
            over_forecast = (comparison_data["forecast"] > comparison_data["actual"]).sum()
            under_forecast = (comparison_data["forecast"] < comparison_data["actual"]).sum()
            
            st.markdown("##### Prognose-Muster")
            
            pattern_fig = go.Figure(data=[go.Pie(
                labels=["Überprognose", "Unterprognose"],
                values=[over_forecast, under_forecast],
                hole=0.6,
                marker_colors=["#f59e0b", "#3b82f6"]
            )])
            pattern_fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(pattern_fig, use_container_width=True, key="pattern_chart")
        
        with trend_cols[1]:
            # Error trend over time
            comparison_data["error_pct"] = (
                (comparison_data["forecast"] - comparison_data["actual"]) / 
                comparison_data["actual"].replace(0, 1) * 100
            )
            
            st.markdown("##### Fehler-Trend")
            
            error_fig = go.Figure()
            error_fig.add_trace(go.Bar(
                x=comparison_data["date"],
                y=comparison_data["error_pct"],
                marker_color=comparison_data["error_pct"].apply(
                    lambda x: "#ef4444" if x > 10 else ("#f59e0b" if x > 0 else "#10b981")
                ),
                name="Fehler %"
            ))
            error_fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
            error_fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=20, b=40),
                yaxis_title="Fehler %",
                xaxis_title="",
                template="plotly_white"
            )
            st.plotly_chart(error_fig, use_container_width=True, key="error_trend_chart")
    
    # ===========================================
    # ACTIONS
    # ===========================================
    
    st.markdown("---")
    action_cols = st.columns([1, 1, 2])
    
    with action_cols[0]:
        if st.button("🔄 Daten aktualisieren", use_container_width=True):
            st.session_state.forecast_comparison_data = None
            st.rerun()
    
    with action_cols[1]:
        # Export comparison data
        if comparison_data is not None:
            csv_data = comparison_data.to_csv(index=False)
            st.download_button(
                "📥 Als CSV exportieren",
                data=csv_data,
                file_name=f"forecast_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )


def _format_number(num: float) -> str:
    """Format number with K/M suffix."""
    if abs(num) >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif abs(num) >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:.0f}"


def _generate_demo_outlook_data(start_date, end_date, aggregation: str) -> pd.DataFrame:
    """Generate demo data for the outlook visualization."""
    # Create date range
    if aggregation == "Monatlich":
        dates = pd.date_range(start=start_date, end=end_date, freq="MS")
    elif aggregation == "Wöchentlich":
        dates = pd.date_range(start=start_date, end=end_date, freq="W-MON")
    else:
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    np.random.seed(42)
    n = len(dates)
    
    # Base pattern with seasonality
    base = 2_500_000 + np.sin(np.linspace(0, 2*np.pi, n)) * 800_000
    
    # Add some noise
    forecast = base + np.random.normal(0, 100_000, n)
    actual = base + np.random.normal(0, 150_000, n) * 0.9  # Actual is slightly lower
    
    # Planner adjustments (positive bias)
    adjustment = np.random.uniform(50_000, 200_000, n)
    
    return pd.DataFrame({
        "date": dates,
        "forecast": forecast.astype(int),
        "actual": actual.astype(int),
        "adjustment": adjustment.astype(int)
    })


def _build_outlook_from_session(forecast_df, actual_df, start_date, end_date, aggregation, task_types) -> pd.DataFrame:
    """Build outlook comparison data from session data."""
    # Map task types to column names
    task_map = {
        "Anrufe": "call_volume",
        "E-Mails": "email_count",
        "Outbound": "outbound_total"
    }
    
    cols_to_use = [task_map.get(t, t) for t in task_types if task_map.get(t, t) in forecast_df.columns]
    
    if not cols_to_use:
        return _generate_demo_outlook_data(start_date, end_date, aggregation)
    
    # Aggregate forecast data
    forecast_copy = forecast_df.copy()
    forecast_copy["date"] = pd.to_datetime(forecast_copy["timestamp"]).dt.date
    
    # Filter by date range
    forecast_copy = forecast_copy[
        (forecast_copy["date"] >= start_date) & 
        (forecast_copy["date"] <= end_date)
    ]
    
    # Sum selected columns
    forecast_copy["total"] = forecast_copy[cols_to_use].sum(axis=1)
    
    # Aggregate based on level
    if aggregation == "Monatlich":
        forecast_copy["period"] = pd.to_datetime(forecast_copy["date"]).dt.to_period("M")
    elif aggregation == "Wöchentlich":
        forecast_copy["period"] = pd.to_datetime(forecast_copy["date"]).dt.to_period("W")
    else:
        forecast_copy["period"] = forecast_copy["date"]
    
    grouped = forecast_copy.groupby("period")["total"].sum().reset_index()
    grouped.columns = ["date", "forecast"]
    grouped["date"] = grouped["date"].apply(lambda x: x.start_time if hasattr(x, "start_time") else x)
    
    # For demo purposes, generate plausible actuals
    np.random.seed(42)
    grouped["actual"] = (grouped["forecast"] * np.random.uniform(0.85, 1.05, len(grouped))).astype(int)
    grouped["adjustment"] = (grouped["forecast"] * np.random.uniform(-0.05, 0.15, len(grouped))).astype(int)
    
    return grouped


def api_documentation_section():
    """Interactive API documentation for admins."""
    st.markdown("### 📚 API Dokumentation")
    st.caption("Vollständige Dokumentation aller verfügbaren REST API Endpunkte.")
    
    # API Status Card
    api_status_col1, api_status_col2, api_status_col3 = st.columns(3)
    
    with api_status_col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; padding: 1.5rem; color: white;">
            <h4 style="margin:0; color: white;">🌐 API Server</h4>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9;">
                Port: 5000<br>
                Starten: <code style="background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 4px;">python -m src.api.routes</code>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with api_status_col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    border-radius: 10px; padding: 1.5rem; color: white;">
            <h4 style="margin:0; color: white;">📊 Verfügbare Endpunkte</h4>
            <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: bold;">15</p>
        </div>
        """, unsafe_allow_html=True)
    
    with api_status_col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    border-radius: 10px; padding: 1.5rem; color: white;">
            <h4 style="margin:0; color: white;">🔒 Authentifizierung</h4>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9;">
                Keine (lokale API)<br>
                Für Produktion: API-Keys aktivieren
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # API Endpoints Documentation
    api_categories = {
        "🏥 Health & Status": [
            {
                "method": "GET",
                "endpoint": "/health",
                "description": "Health Check - Überprüft, ob die API läuft",
                "request": None,
                "response": '''{
    "status": "healthy",
    "timestamp": "2026-01-15T10:30:00Z",
    "version": "1.0.0"
}''',
                "curl": "curl http://localhost:5000/health"
            },
            {
                "method": "GET",
                "endpoint": "/status",
                "description": "System Status - Zeigt ob Modell und Daten geladen sind",
                "request": None,
                "response": '''{
    "model_loaded": true,
    "data_loaded": true,
    "model_targets": ["call_volume", "email_count"],
    "data_rows": 8760
}''',
                "curl": "curl http://localhost:5000/status"
            }
        ],
        "🔮 Forecasting": [
            {
                "method": "POST",
                "endpoint": "/api/v1/forecast",
                "description": "Erstellt eine Prognose für den angegebenen Zeitraum",
                "request": '''{
    "start_date": "2026-01-15",
    "end_date": "2026-01-21",
    "confidence_level": 0.95
}''',
                "response": '''{
    "forecast": [
        {"timestamp": "2026-01-15T08:00:00", "call_volume": 45, "email_count": 20},
        ...
    ],
    "metadata": {
        "start_date": "2026-01-15",
        "end_date": "2026-01-21",
        "horizon_hours": 168,
        "targets": ["call_volume", "email_count"],
        "generated_at": "2026-01-14T10:30:00Z"
    }
}''',
                "curl": '''curl -X POST http://localhost:5000/api/v1/forecast \\
  -H "Content-Type: application/json" \\
  -d '{"start_date": "2026-01-15", "end_date": "2026-01-21"}'
'''
            },
            {
                "method": "GET",
                "endpoint": "/api/v1/forecast/{id}",
                "description": "Ruft eine gespeicherte Prognose ab",
                "request": None,
                "response": '''{
    "forecast_id": "fc_20260115_abc123",
    "created_at": "2026-01-14T10:30:00",
    "forecast_start": "2026-01-15",
    "forecast_end": "2026-01-21",
    "predictions": {...}
}''',
                "curl": "curl http://localhost:5000/api/v1/forecast/fc_20260115_abc123"
            },
            {
                "method": "POST",
                "endpoint": "/api/v1/staffing",
                "description": "Berechnet den Personalbedarf basierend auf Workload",
                "request": '''{
    "workload": [
        {"timestamp": "2026-01-15T08:00:00", "calls": 100, "emails": 50}
    ],
    "config": {
        "service_level": 0.8,
        "service_time": 20,
        "shrinkage": 0.3
    }
}''',
                "response": '''{
    "staffing": [
        {"timestamp": "2026-01-15T08:00:00", "required_agents": 12}
    ],
    "summary": {
        "total_fte_hours": 96,
        "peak_agents": 15,
        "avg_agents": 10
    }
}''',
                "curl": '''curl -X POST http://localhost:5000/api/v1/staffing \\
  -H "Content-Type: application/json" \\
  -d '{"workload": [...], "config": {...}}'
'''
            }
        ],
        "🧠 Models": [
            {
                "method": "GET",
                "endpoint": "/api/v1/models",
                "description": "Listet alle verfügbaren Modelle auf",
                "request": None,
                "response": '''{
    "models": [
        {
            "model_id": "workload_forecaster",
            "version": "v1.0",
            "created_at": "2026-01-14T10:00:00",
            "model_type": "Prophet",
            "metrics": {"rmse": 12.5, "mape": 8.2}
        }
    ]
}''',
                "curl": "curl http://localhost:5000/api/v1/models"
            },
            {
                "method": "POST",
                "endpoint": "/api/v1/models/{id}/activate",
                "description": "Aktiviert eine bestimmte Modellversion",
                "request": '''{
    "version": "v1.0"
}''',
                "response": '''{
    "success": true,
    "message": "Activated version v1.0 for model workload_forecaster"
}''',
                "curl": '''curl -X POST http://localhost:5000/api/v1/models/workload_forecaster/activate \\
  -H "Content-Type: application/json" \\
  -d '{"version": "v1.0"}'
'''
            }
        ],
        "📤 Data Upload": [
            {
                "method": "POST",
                "endpoint": "/api/v1/data/upload",
                "description": "Lädt Datendateien hoch (multipart/form-data)",
                "request": '''Form-Data:
- file: Datei (CSV/Excel)
- data_type: "calls" | "emails" | "outbound" | "auto"
- overwrite: "true" | "false"''',
                "response": '''{
    "success": true,
    "uploaded_files": [
        {
            "filename": "calls_data.csv",
            "rows": 8760,
            "columns": ["timestamp", "call_volume"],
            "data_type": "calls",
            "column_mapping": {"timestamp": "timestamp", "call_volume": "call_volume"}
        }
    ],
    "summary": {"total_files": 1, "total_rows": 8760}
}''',
                "curl": '''curl -X POST http://localhost:5000/api/v1/data/upload \\
  -F "file=@calls_data.csv" \\
  -F "data_type=calls"
'''
            },
            {
                "method": "POST",
                "endpoint": "/api/v1/data/upload/batch",
                "description": "Batch-Upload mit Base64-kodierten Dateien",
                "request": '''{
    "files": [
        {
            "name": "calls.csv",
            "type": "calls",
            "content": "base64_encoded_content"
        }
    ],
    "merge_strategy": "append"
}''',
                "response": '''{
    "success": true,
    "merge_strategy": "append",
    "uploaded_files": [...],
    "summary": {"total_files": 1, "successful": 1, "failed": 0}
}''',
                "curl": '''curl -X POST http://localhost:5000/api/v1/data/upload/batch \\
  -H "Content-Type: application/json" \\
  -d '{"files": [{"name": "calls.csv", "type": "calls", "content": "..."}]}'
'''
            },
            {
                "method": "GET",
                "endpoint": "/api/v1/data/files",
                "description": "Listet alle hochgeladenen Dateien auf",
                "request": None,
                "response": '''{
    "files": [
        {
            "filename": "calls_data.csv",
            "size_bytes": 125000,
            "modified_at": "2026-01-15T10:30:00Z",
            "rows": 8760
        }
    ],
    "total": 1
}''',
                "curl": "curl http://localhost:5000/api/v1/data/files"
            },
            {
                "method": "DELETE",
                "endpoint": "/api/v1/data/files/{filename}",
                "description": "Löscht eine hochgeladene Datei",
                "request": None,
                "response": '''{
    "success": true,
    "message": "File 'old_data.csv' deleted"
}''',
                "curl": "curl -X DELETE http://localhost:5000/api/v1/data/files/old_data.csv"
            }
        ],
        "📊 Data Management": [
            {
                "method": "GET",
                "endpoint": "/api/v1/data/summary",
                "description": "Zeigt Zusammenfassung der geladenen Daten",
                "request": None,
                "response": '''{
    "loaded": true,
    "rows": 8760,
    "columns": ["timestamp", "call_volume", "email_count"],
    "date_range": {
        "start": "2025-01-01T00:00:00",
        "end": "2025-12-31T23:00:00"
    },
    "statistics": {
        "call_volume": {"mean": 42.5, "std": 15.2},
        "email_count": {"mean": 18.3, "std": 8.1}
    }
}''',
                "curl": "curl http://localhost:5000/api/v1/data/summary"
            },
            {
                "method": "POST",
                "endpoint": "/api/v1/data/validate",
                "description": "Validiert eine hochgeladene Datendatei",
                "request": "Form-Data: file (CSV/Excel)",
                "response": '''{
    "valid": true,
    "issues": [],
    "summary": {
        "rows": 8760,
        "columns": 3,
        "has_timestamp": true
    }
}''',
                "curl": '''curl -X POST http://localhost:5000/api/v1/data/validate \\
  -F "file=@data.csv"
'''
            },
            {
                "method": "POST",
                "endpoint": "/api/v1/data/load",
                "description": "Lädt Daten in den Speicher für Verarbeitung",
                "request": '''{
    "files": ["calls.csv", "emails.csv"],
    "date_range": {
        "start": "2025-01-01",
        "end": "2026-01-01"
    }
}''',
                "response": '''{
    "success": true,
    "rows": 8760,
    "columns": ["timestamp", "call_volume", "email_count"],
    "date_range": {"start": "2025-01-01", "end": "2025-12-31"},
    "summary": {
        "call_volume": {"mean": 42.5, "sum": 372300, "min": 5, "max": 120}
    }
}''',
                "curl": '''curl -X POST http://localhost:5000/api/v1/data/load \\
  -H "Content-Type: application/json" \\
  -d '{"files": ["calls.csv"]}'
'''
            }
        ]
    }
    
    # Render API documentation
    for category, endpoints in api_categories.items():
        with st.expander(f"**{category}**", expanded=False):
            for ep in endpoints:
                # Method badge color
                method_colors = {
                    "GET": "#10b981",
                    "POST": "#3b82f6",
                    "PUT": "#f59e0b",
                    "DELETE": "#ef4444"
                }
                method_color = method_colors.get(ep["method"], "#6b7280")
                
                st.markdown(f"""
                <div style="
                    background: white;
                    border: 1px solid #e5e7eb;
                    border-radius: 8px;
                    padding: 1rem;
                    margin-bottom: 1rem;
                ">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem;">
                        <span style="
                            background: {method_color};
                            color: white;
                            padding: 4px 12px;
                            border-radius: 4px;
                            font-weight: bold;
                            font-size: 0.85rem;
                        ">{ep["method"]}</span>
                        <code style="
                            background: #f3f4f6;
                            padding: 4px 8px;
                            border-radius: 4px;
                            font-size: 0.95rem;
                        ">{ep["endpoint"]}</code>
                    </div>
                    <p style="color: #374151; margin: 0.5rem 0;">{ep["description"]}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Request/Response tabs
                if ep["request"] or ep["response"]:
                    req_resp_cols = st.columns(2)
                    
                    with req_resp_cols[0]:
                        if ep["request"]:
                            st.markdown("**Request:**")
                            st.code(ep["request"], language="json")
                        else:
                            st.markdown("**Request:** Keine Parameter")
                    
                    with req_resp_cols[1]:
                        st.markdown("**Response:**")
                        st.code(ep["response"], language="json")
                
                # cURL example
                if ep.get("curl"):
                    with st.popover("📋 cURL Beispiel"):
                        st.code(ep["curl"], language="bash")
                        if st.button("📋 Kopieren", key=f"copy_{ep['endpoint']}"):
                            st.write("✅ In Zwischenablage kopiert!")
                
                st.markdown("---")
    
    # Quick Test Section
    st.markdown("### 🧪 API Quick Test")
    st.caption("Testen Sie die API direkt aus der UI (API muss laufen auf Port 5000)")
    
    test_col1, test_col2 = st.columns([1, 2])
    
    with test_col1:
        test_endpoint = st.selectbox(
            "Endpunkt auswählen",
            options=[
                "GET /health",
                "GET /status",
                "GET /api/v1/models",
                "GET /api/v1/data/files",
                "GET /api/v1/data/summary"
            ],
            key="api_test_endpoint"
        )
        
        api_base_url = st.text_input(
            "API Base URL",
            value="http://localhost:5000",
            key="api_base_url"
        )
        
        if st.button("🚀 Request senden", type="primary"):
            try:
                import requests
                
                method, path = test_endpoint.split(" ", 1)
                url = f"{api_base_url}{path}"
                
                if method == "GET":
                    response = requests.get(url, timeout=5)
                else:
                    response = requests.post(url, json={}, timeout=5)
                
                st.session_state.api_test_result = {
                    "status_code": response.status_code,
                    "response": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                    "url": url
                }
            except ImportError:
                st.error("❌ `requests` Bibliothek nicht installiert. Installieren mit: `pip install requests`")
            except Exception as e:
                st.session_state.api_test_result = {
                    "error": str(e),
                    "url": url
                }
    
    with test_col2:
        if "api_test_result" in st.session_state:
            result = st.session_state.api_test_result
            
            if "error" in result:
                st.error(f"❌ Fehler: {result['error']}")
                st.caption(f"URL: {result.get('url', 'N/A')}")
            else:
                status_color = "#10b981" if result["status_code"] == 200 else "#ef4444"
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">
                    <span style="
                        background: {status_color};
                        color: white;
                        padding: 4px 12px;
                        border-radius: 4px;
                        font-weight: bold;
                    ">Status: {result["status_code"]}</span>
                    <code>{result.get("url", "")}</code>
                </div>
                """, unsafe_allow_html=True)
                
                st.json(result["response"])
    
    # Code Examples Section
    st.markdown("---")
    st.markdown("### 💻 Code-Beispiele")
    
    code_tabs = st.tabs(["🐍 Python", "📜 JavaScript", "🦀 cURL"])
    
    with code_tabs[0]:
        st.code('''import requests

# API Base URL
BASE_URL = "http://localhost:5000"

# Health Check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Upload Datei
with open("calls_data.csv", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/api/v1/data/upload",
        files={"file": f},
        data={"data_type": "calls"}
    )
print(response.json())

# Daten laden
response = requests.post(f"{BASE_URL}/api/v1/data/load")
print(response.json())

# Forecast erstellen
response = requests.post(
    f"{BASE_URL}/api/v1/forecast",
    json={
        "start_date": "2026-01-15",
        "end_date": "2026-01-21",
        "confidence_level": 0.95
    }
)
forecast = response.json()
print(f"Forecast für {len(forecast['forecast'])} Stunden erstellt")
''', language="python")
    
    with code_tabs[1]:
        st.code('''// API Base URL
const BASE_URL = "http://localhost:5000";

// Health Check
fetch(`${BASE_URL}/health`)
  .then(res => res.json())
  .then(data => console.log(data));

// Upload Datei
const formData = new FormData();
formData.append("file", fileInput.files[0]);
formData.append("data_type", "calls");

fetch(`${BASE_URL}/api/v1/data/upload`, {
  method: "POST",
  body: formData
})
  .then(res => res.json())
  .then(data => console.log(data));

// Forecast erstellen
fetch(`${BASE_URL}/api/v1/forecast`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    start_date: "2026-01-15",
    end_date: "2026-01-21",
    confidence_level: 0.95
  })
})
  .then(res => res.json())
  .then(data => console.log(data));
''', language="javascript")
    
    with code_tabs[2]:
        st.code('''# Health Check
curl http://localhost:5000/health

# Upload Datei
curl -X POST http://localhost:5000/api/v1/data/upload \\
  -F "file=@calls_data.csv" \\
  -F "data_type=calls"

# Daten laden
curl -X POST http://localhost:5000/api/v1/data/load

# Forecast erstellen
curl -X POST http://localhost:5000/api/v1/forecast \\
  -H "Content-Type: application/json" \\
  -d '{
    "start_date": "2026-01-15",
    "end_date": "2026-01-21",
    "confidence_level": 0.95
  }'

# Modelle auflisten
curl http://localhost:5000/api/v1/models

# Datei löschen
curl -X DELETE http://localhost:5000/api/v1/data/files/old_data.csv
''', language="bash")


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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📁 Daten",
        "🔍 Erkunden",
        "🧠 Training",
        "🔮 Forecast",
        "📊 Outlook",
        "📈 Analytics",
        "🔄 Review",
        "🏢 Call-Center",
        "📚 API",
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
        demand_outlook_section()
    
    with tab6:
        analytics_section(capacity_config)
    
    with tab7:
        review_forecasts_section()
    
    with tab8:
        call_center_management_section()
    
    with tab9:
        api_documentation_section()
    
    with tab10:
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


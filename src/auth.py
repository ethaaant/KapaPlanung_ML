"""
Authentication system for the Workforce Planning app.
Provides login/logout, role-based access control, and session management.

Security features:
- bcrypt password hashing (with fallback to sha256 for compatibility)
- Session timeout
- Login attempt tracking with account lockout
- Audit logging
"""
import streamlit as st
import hashlib
import secrets
from typing import Optional, Dict
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from functools import wraps

# Try to import bcrypt, fallback to sha256 if not available
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False


class UserRole(Enum):
    """User roles with different access levels."""
    ADMIN = "admin"
    DIENSTLEISTER = "dienstleister"


@dataclass
class User:
    """User account."""
    username: str
    password_hash: str
    role: UserRole
    display_name: str
    email: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True


@dataclass
class LoginAttempt:
    """Track login attempts for rate limiting."""
    username: str
    timestamp: datetime
    success: bool
    ip_address: Optional[str] = None


class AuthConfig:
    """Authentication configuration."""
    SESSION_TIMEOUT_MINUTES: int = 60
    MAX_LOGIN_ATTEMPTS: int = 5
    LOCKOUT_DURATION_MINUTES: int = 15
    MIN_PASSWORD_LENGTH: int = 8


def hash_password(password: str) -> str:
    """
    Hash a password securely.
    Uses bcrypt if available, otherwise falls back to SHA-256.
    """
    if BCRYPT_AVAILABLE:
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode(), salt).decode()
    else:
        # Fallback to SHA-256 with salt
        salt = secrets.token_hex(16)
        hash_val = hashlib.sha256((salt + password).encode()).hexdigest()
        return f"sha256${salt}${hash_val}"


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a password against its hash.
    Handles both bcrypt and SHA-256 hashes.
    """
    if password_hash.startswith("sha256$"):
        # SHA-256 with salt format
        _, salt, expected_hash = password_hash.split("$")
        actual_hash = hashlib.sha256((salt + password).encode()).hexdigest()
        return secrets.compare_digest(actual_hash, expected_hash)
    elif password_hash.startswith("$2"):
        # bcrypt hash (starts with $2a$, $2b$, or $2y$)
        if BCRYPT_AVAILABLE:
            try:
                return bcrypt.checkpw(password.encode(), password_hash.encode())
            except (ValueError, AttributeError):
                return False
        return False
    else:
        # Legacy SHA-256 without salt (for backwards compatibility)
        legacy_hash = hashlib.sha256(password.encode()).hexdigest()
        return secrets.compare_digest(password_hash, legacy_hash)


# Default users - In production, this should be in a database
# Note: These use the legacy SHA-256 hash for backwards compatibility
DEFAULT_USERS: Dict[str, User] = {
    "admin": User(
        username="admin",
        password_hash=hashlib.sha256("admin123".encode()).hexdigest(),
        role=UserRole.ADMIN,
        display_name="Administrator",
        email="admin@example.com"
    ),
    "dienstleister": User(
        username="dienstleister",
        password_hash=hashlib.sha256("service123".encode()).hexdigest(),
        role=UserRole.DIENSTLEISTER,
        display_name="Dienstleister",
        email="service@example.com"
    ),
    "manager": User(
        username="manager",
        password_hash=hashlib.sha256("manager123".encode()).hexdigest(),
        role=UserRole.ADMIN,
        display_name="Manager",
        email="manager@example.com"
    ),
    "agent": User(
        username="agent",
        password_hash=hashlib.sha256("agent123".encode()).hexdigest(),
        role=UserRole.DIENSTLEISTER,
        display_name="Service Agent",
        email="agent@example.com"
    ),
}


def init_auth_state():
    """Initialize authentication session state."""
    defaults = {
        "authenticated": False,
        "user": None,
        "user_role": None,
        "login_time": None,
        "last_activity": None,
        "login_attempts": {},  # {username: [timestamps]}
        "locked_accounts": {},  # {username: unlock_time}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_settings():
    """Get settings safely."""
    try:
        from src.utils.settings import settings
        return settings
    except ImportError:
        return AuthConfig()


def _log_login_attempt(username: str, success: bool):
    """Log a login attempt for audit purposes."""
    try:
        from src.utils.logging_config import audit_logger
        audit_logger.log_login(username, success)
    except ImportError:
        pass  # Logging not available


def is_account_locked(username: str) -> tuple[bool, Optional[datetime]]:
    """Check if an account is locked due to too many failed attempts."""
    username = username.lower()
    locked_accounts = st.session_state.get("locked_accounts", {})
    
    if username in locked_accounts:
        unlock_time = locked_accounts[username]
        if datetime.now() < unlock_time:
            return True, unlock_time
        else:
            # Lockout expired, remove it
            del locked_accounts[username]
            st.session_state.locked_accounts = locked_accounts
    
    return False, None


def record_login_attempt(username: str, success: bool):
    """Record a login attempt and handle lockout logic."""
    username = username.lower()
    settings = _get_settings()
    max_attempts = getattr(settings, "max_login_attempts", AuthConfig.MAX_LOGIN_ATTEMPTS)
    lockout_duration = getattr(settings, "lockout_duration_minutes", AuthConfig.LOCKOUT_DURATION_MINUTES)
    
    # Initialize if needed
    if "login_attempts" not in st.session_state:
        st.session_state.login_attempts = {}
    
    attempts = st.session_state.login_attempts.get(username, [])
    
    if success:
        # Clear attempts on successful login
        if username in st.session_state.login_attempts:
            del st.session_state.login_attempts[username]
    else:
        # Record failed attempt
        now = datetime.now()
        # Remove attempts older than lockout duration
        cutoff = now - timedelta(minutes=lockout_duration)
        attempts = [t for t in attempts if t > cutoff]
        attempts.append(now)
        st.session_state.login_attempts[username] = attempts
        
        # Check if we should lock the account
        if len(attempts) >= max_attempts:
            if "locked_accounts" not in st.session_state:
                st.session_state.locked_accounts = {}
            st.session_state.locked_accounts[username] = now + timedelta(minutes=lockout_duration)


def check_session_timeout() -> bool:
    """
    Check if the current session has timed out.
    
    Returns:
        True if session is valid, False if timed out.
    """
    if not st.session_state.get("authenticated"):
        return False
    
    settings = _get_settings()
    timeout_minutes = getattr(settings, "session_timeout_minutes", AuthConfig.SESSION_TIMEOUT_MINUTES)
    
    last_activity = st.session_state.get("last_activity")
    if last_activity:
        if datetime.now() - last_activity > timedelta(minutes=timeout_minutes):
            # Session timed out
            logout()
            return False
    
    # Update last activity
    st.session_state.last_activity = datetime.now()
    return True


def authenticate(username: str, password: str) -> Optional[User]:
    """
    Authenticate a user.
    
    Args:
        username: The username.
        password: The plain text password.
        
    Returns:
        User object if authenticated, None otherwise.
    """
    username_lower = username.lower()
    user = DEFAULT_USERS.get(username_lower)
    
    if not user or not user.is_active:
        return None
    
    # Check password
    if verify_password(password, user.password_hash):
        return user
    
    return None


def login(username: str, password: str) -> dict:
    """
    Attempt to log in a user.
    
    Args:
        username: The username.
        password: The password.
        
    Returns:
        Dict with 'success', 'message', and optionally 'user' keys.
    """
    # Check if account is locked
    is_locked, unlock_time = is_account_locked(username)
    if is_locked:
        minutes_remaining = int((unlock_time - datetime.now()).total_seconds() / 60) + 1
        return {
            "success": False,
            "message": f"Account is locked. Please try again in {minutes_remaining} minute(s).",
            "locked": True
        }
    
    # Attempt authentication
    user = authenticate(username, password)
    
    if user:
        # Successful login
        st.session_state.authenticated = True
        st.session_state.user = user
        st.session_state.user_role = user.role
        st.session_state.login_time = datetime.now()
        st.session_state.last_activity = datetime.now()
        
        record_login_attempt(username, success=True)
        _log_login_attempt(username, success=True)
        
        return {
            "success": True,
            "message": f"Welcome, {user.display_name}!",
            "user": user
        }
    else:
        # Failed login
        record_login_attempt(username, success=False)
        _log_login_attempt(username, success=False)
        
        # Get remaining attempts
        attempts = st.session_state.login_attempts.get(username.lower(), [])
        settings = _get_settings()
        max_attempts = getattr(settings, "max_login_attempts", AuthConfig.MAX_LOGIN_ATTEMPTS)
        remaining = max_attempts - len(attempts)
        
        if remaining > 0:
            return {
                "success": False,
                "message": f"Invalid username or password. {remaining} attempt(s) remaining."
            }
        else:
            return {
                "success": False,
                "message": "Account locked due to too many failed attempts.",
                "locked": True
            }


def logout():
    """Log out the current user."""
    user = get_current_user()
    if user:
        try:
            from src.utils.logging_config import audit_logger
            audit_logger.log_logout(user.username)
        except ImportError:
            pass
    
    # Clear authentication state only - preserve forecast/model data
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.user_role = None
    st.session_state.login_time = None
    st.session_state.last_activity = None
    
    # Keys to preserve across user sessions (forecast data, trained models, etc.)
    # This allows dienstleisters to see forecasts created by admins
    PRESERVE_KEYS = {
        "login_attempts",
        "locked_accounts",
        # Forecast and model data
        "forecast_df",
        "staffing_plan",
        "forecast_generated",
        "forecast_start",
        "forecast_end",
        "forecast_result",
        "forecaster",
        "preprocessor",
        "feature_set",
        "model_trained",
        "training_metrics",
        "model_type",
        # Data
        "combined_data",
        "data_loaded",
        "call_data",
        "email_data",
        "outbound_data",
        # Scenario analyzer
        "scenario_analyzer",
        # Business metrics
        "business_metrics",
        # Model management
        "model_source",
        "model_version",
        "model_metadata",
        "adjustment_factors",
    }
    
    # Preserve important data
    preserved_data = {key: st.session_state[key] for key in PRESERVE_KEYS if key in st.session_state}
    
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Restore preserved data
    for key, value in preserved_data.items():
        st.session_state[key] = value


def is_authenticated() -> bool:
    """Check if user is authenticated and session is valid."""
    if not st.session_state.get("authenticated", False):
        return False
    
    # Check session timeout
    return check_session_timeout()


def get_current_user() -> Optional[User]:
    """Get the current logged-in user."""
    return st.session_state.get("user")


def get_user_role() -> Optional[UserRole]:
    """Get the current user's role."""
    return st.session_state.get("user_role")


def is_admin() -> bool:
    """Check if current user is an admin."""
    return get_user_role() == UserRole.ADMIN


def is_dienstleister() -> bool:
    """Check if current user is a Dienstleister."""
    return get_user_role() == UserRole.DIENSTLEISTER


def get_session_info() -> dict:
    """Get information about the current session."""
    if not is_authenticated():
        return {"authenticated": False}
    
    login_time = st.session_state.get("login_time")
    last_activity = st.session_state.get("last_activity")
    settings = _get_settings()
    timeout_minutes = getattr(settings, "session_timeout_minutes", AuthConfig.SESSION_TIMEOUT_MINUTES)
    
    session_duration = None
    time_remaining = None
    
    if login_time:
        session_duration = datetime.now() - login_time
    
    if last_activity:
        timeout_at = last_activity + timedelta(minutes=timeout_minutes)
        time_remaining = timeout_at - datetime.now()
        if time_remaining.total_seconds() < 0:
            time_remaining = timedelta(0)
    
    return {
        "authenticated": True,
        "login_time": login_time,
        "last_activity": last_activity,
        "session_duration": session_duration,
        "time_remaining": time_remaining,
        "timeout_minutes": timeout_minutes
    }


def require_auth(func):
    """Decorator to require authentication for a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not is_authenticated():
            st.warning("Please log in to access this feature.")
            return None
        return func(*args, **kwargs)
    return wrapper


def require_admin(func):
    """Decorator to require admin role for a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not is_authenticated():
            st.warning("Please log in to access this feature.")
            return None
        if not is_admin():
            st.error("⛔ Access denied. Admin privileges required.")
            return None
        return func(*args, **kwargs)
    return wrapper


def require_role(required_roles: list):
    """Decorator factory to require specific roles for a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_authenticated():
                st.warning("Please log in to access this feature.")
                return None
            
            user_role = get_user_role()
            if user_role not in required_roles:
                st.error("⛔ Access denied. You don't have permission for this action.")
                return None
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def render_login_page():
    """Render the login page with clean, modern design."""
    
    # Add spacing at the top
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        # Logo/Header section
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">📊</div>
            <h1 style="color: #1a1a2e; font-size: 1.75rem; font-weight: 600; margin-bottom: 0.25rem;">
                Workforce Planning
            </h1>
            <p style="color: #6b7280; font-size: 0.95rem;">
                Sign in to access the dashboard
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check for locked accounts notification
        init_auth_state()
        
        # Login form in a card
        with st.container():
            with st.form("login_form"):
                st.markdown("##### Username")
                username = st.text_input(
                    "Username", 
                    placeholder="Enter your username", 
                    label_visibility="collapsed",
                    key="login_username"
                )
                
                st.markdown("##### Password")
                password = st.text_input(
                    "Password", 
                    type="password", 
                    placeholder="Enter your password", 
                    label_visibility="collapsed",
                    key="login_password"
                )
                
                st.markdown("<br>", unsafe_allow_html=True)
                submit = st.form_submit_button("Sign In", use_container_width=True, type="primary")
                
                if submit:
                    if username and password:
                        result = login(username, password)
                        if result["success"]:
                            st.success(result["message"])
                            st.rerun()
                        else:
                            if result.get("locked"):
                                st.error(f"🔒 {result['message']}")
                            else:
                                st.error(result["message"])
                    else:
                        st.warning("Please enter both username and password")
        
        # Demo accounts section
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.expander("📋 Demo Accounts", expanded=False):
            st.markdown("""
            **Admin Access** (Full Features)
            - Username: `admin`
            - Password: `admin123`
            
            **Dienstleister Access** (View Only)
            - Username: `dienstleister`
            - Password: `service123`
            """)
        
        # Security info
        settings = _get_settings()
        max_attempts = getattr(settings, "max_login_attempts", AuthConfig.MAX_LOGIN_ATTEMPTS)
        timeout = getattr(settings, "session_timeout_minutes", AuthConfig.SESSION_TIMEOUT_MINUTES)
        
        st.markdown(f"""
        <div style="text-align: center; margin-top: 1.5rem; color: #9ca3af; font-size: 0.75rem;">
            🔒 Session timeout: {timeout} minutes | Max attempts: {max_attempts}
        </div>
        """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
        <div style="text-align: center; margin-top: 2rem; color: #9ca3af; font-size: 0.85rem;">
            CHECK24 Mobilfunk | Workforce Planning System
        </div>
        """, unsafe_allow_html=True)


def render_user_info_sidebar():
    """Render user info in the sidebar with clean design."""
    if is_authenticated():
        user = get_current_user()
        session_info = get_session_info()
        
        st.sidebar.markdown("---")
        
        # User info card
        role_color = "#4f46e5" if user.role == UserRole.ADMIN else "#10b981"
        role_bg = "#eef2ff" if user.role == UserRole.ADMIN else "#ecfdf5"
        
        # Format session time remaining
        time_remaining = session_info.get("time_remaining")
        time_str = ""
        if time_remaining:
            minutes = int(time_remaining.total_seconds() / 60)
            time_str = f"Session: {minutes}m remaining"
        
        st.sidebar.markdown(f"""
        <div style="
            background: white;
            border-radius: 10px;
            padding: 1rem;
            border: 1px solid #e5e7eb;
            margin-bottom: 1rem;
        ">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    background: {role_bg};
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.25rem;
                ">👤</div>
                <div>
                    <div style="font-weight: 600; color: #1a1a2e; font-size: 0.95rem;">
                        {user.display_name}
                    </div>
                    <div style="
                        font-size: 0.75rem;
                        color: {role_color};
                        background: {role_bg};
                        padding: 0.15rem 0.5rem;
                        border-radius: 4px;
                        display: inline-block;
                        margin-top: 0.25rem;
                    ">
                        {user.role.value.title()}
                    </div>
                </div>
            </div>
            {f'<div style="font-size: 0.7rem; color: #9ca3af; margin-top: 0.5rem; text-align: center;">{time_str}</div>' if time_str else ''}
        </div>
        """, unsafe_allow_html=True)
        
        if st.sidebar.button("Sign Out", use_container_width=True):
            logout()
            st.rerun()

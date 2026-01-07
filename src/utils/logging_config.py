"""
Structured logging configuration.
Provides consistent logging across the application.
"""
import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from functools import lru_cache


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "user"):
            log_data["user"] = record.user
        if hasattr(record, "action"):
            log_data["action"] = record.action
        if hasattr(record, "details"):
            log_data["details"] = record.details
        
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_format: str = "text",
    log_file: Optional[str] = None,
    app_name: str = "workforce_planning"
) -> logging.Logger:
    """
    Set up application logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ("json" or "text")
        log_file: Path to log file (optional)
        app_name: Application name for the logger
        
    Returns:
        Configured logger instance
    """
    # Get numeric level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(app_name)
    logger.setLevel(numeric_level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if log_format.lower() == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ColoredFormatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
    
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
    
    return logger


@lru_cache()
def get_logger(name: str = "workforce_planning") -> logging.Logger:
    """Get a logger instance."""
    from src.utils.settings import settings
    
    return setup_logging(
        level=settings.log_level,
        log_format=settings.log_format,
        log_file=str(settings.logs_path / "app.log"),
        app_name=name
    )


class AuditLogger:
    """
    Audit logger for tracking user actions.
    """
    
    def __init__(self):
        self.logger = get_logger("audit")
    
    def log_action(
        self,
        user: str,
        action: str,
        details: dict = None,
        success: bool = True
    ):
        """Log a user action."""
        extra = {
            "user": user,
            "action": action,
            "details": details or {},
            "success": success
        }
        
        if success:
            self.logger.info(
                f"User '{user}' performed action: {action}",
                extra=extra
            )
        else:
            self.logger.warning(
                f"User '{user}' failed action: {action}",
                extra=extra
            )
    
    def log_login(self, username: str, success: bool, ip: str = None):
        """Log a login attempt."""
        self.log_action(
            user=username,
            action="login",
            details={"ip": ip} if ip else {},
            success=success
        )
    
    def log_logout(self, username: str):
        """Log a logout."""
        self.log_action(user=username, action="logout")
    
    def log_data_access(self, username: str, data_type: str, operation: str):
        """Log data access."""
        self.log_action(
            user=username,
            action="data_access",
            details={"data_type": data_type, "operation": operation}
        )
    
    def log_model_action(self, username: str, action: str, model_id: str = None):
        """Log model-related action."""
        self.log_action(
            user=username,
            action=f"model_{action}",
            details={"model_id": model_id} if model_id else {}
        )
    
    def log_export(self, username: str, export_type: str, format: str):
        """Log data export."""
        self.log_action(
            user=username,
            action="export",
            details={"type": export_type, "format": format}
        )


# Global audit logger instance
audit_logger = AuditLogger()


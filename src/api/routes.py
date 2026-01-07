"""
REST API routes for Workforce Planning.
Uses Flask for lightweight API endpoints.

Note: This is a basic API implementation. For production,
consider using FastAPI for better async support and auto-documentation.
"""
import json
import pandas as pd
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Optional

# Try to import Flask (optional dependency)
try:
    from flask import Flask, request, jsonify, Response
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None


def require_flask(func):
    """Decorator to check if Flask is available."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not FLASK_AVAILABLE:
            raise ImportError(
                "Flask is required for API endpoints. "
                "Install with: pip install flask"
            )
        return func(*args, **kwargs)
    return wrapper


@require_flask
def create_app(config: dict = None) -> "Flask":
    """
    Create and configure the Flask application.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured Flask app
    """
    app = Flask(__name__)
    
    # Configuration
    app.config.update(config or {})
    app.config.setdefault("JSON_SORT_KEYS", False)
    
    # Store for loaded models and data
    app.forecaster = None
    app.model_manager = None
    app.data = None
    
    # ===========================================
    # MIDDLEWARE
    # ===========================================
    
    @app.before_request
    def log_request():
        """Log incoming requests."""
        try:
            from src.utils.logging_config import get_logger
            logger = get_logger("api")
            logger.info(f"API Request: {request.method} {request.path}")
        except ImportError:
            pass
    
    @app.after_request
    def add_headers(response):
        """Add common headers."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response
    
    # ===========================================
    # ERROR HANDLERS
    # ===========================================
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            "error": "Bad Request",
            "message": str(error.description)
        }), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        return jsonify({
            "error": "Unauthorized",
            "message": "Authentication required"
        }), 401
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "error": "Not Found",
            "message": "Resource not found"
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            "error": "Internal Server Error",
            "message": "An unexpected error occurred"
        }), 500
    
    # ===========================================
    # HEALTH & STATUS ENDPOINTS
    # ===========================================
    
    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "version": "1.0.0"
        })
    
    @app.route("/status", methods=["GET"])
    def system_status():
        """Get system status."""
        model_loaded = app.forecaster is not None and app.forecaster.is_fitted
        data_loaded = app.data is not None and len(app.data) > 0
        
        return jsonify({
            "model_loaded": model_loaded,
            "data_loaded": data_loaded,
            "model_targets": app.forecaster.target_columns if model_loaded else [],
            "data_rows": len(app.data) if data_loaded else 0
        })
    
    # ===========================================
    # FORECAST ENDPOINTS
    # ===========================================
    
    @app.route("/api/v1/forecast", methods=["POST"])
    def generate_forecast():
        """
        Generate a forecast.
        
        Request body:
        {
            "start_date": "2026-01-15",
            "end_date": "2026-01-21",
            "confidence_level": 0.95
        }
        
        Returns:
        {
            "forecast": [...],
            "metadata": {...}
        }
        """
        if app.forecaster is None or not app.forecaster.is_fitted:
            return jsonify({
                "error": "Model not loaded",
                "message": "No trained model available. Please train a model first."
            }), 400
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "Invalid request",
                "message": "Request body must be JSON"
            }), 400
        
        try:
            start_date = pd.Timestamp(data.get("start_date", datetime.now()))
            end_date = pd.Timestamp(data.get("end_date", datetime.now() + timedelta(days=7)))
            confidence_level = float(data.get("confidence_level", 0.95))
            
            # Calculate horizon
            horizon_hours = int((end_date - start_date).total_seconds() / 3600) + 24
            
            # Generate forecast
            from src.data.preprocessor import Preprocessor
            preprocessor = Preprocessor()
            
            result = app.forecaster.forecast_horizon(
                last_known_data=app.data,
                horizon_hours=horizon_hours,
                preprocessor=preprocessor,
                start_date=start_date,
                confidence_level=confidence_level
            )
            
            # Format response
            forecast_data = result.predictions.copy()
            forecast_data["timestamp"] = result.timestamps.values
            
            return jsonify({
                "forecast": forecast_data.to_dict(orient="records"),
                "metadata": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "horizon_hours": horizon_hours,
                    "targets": result.predictions.columns.tolist(),
                    "generated_at": datetime.utcnow().isoformat() + "Z"
                }
            })
            
        except Exception as e:
            return jsonify({
                "error": "Forecast error",
                "message": str(e)
            }), 500
    
    @app.route("/api/v1/forecast/<forecast_id>", methods=["GET"])
    def get_forecast(forecast_id: str):
        """Get a saved forecast by ID."""
        try:
            from src.models.forecast_tracker import get_forecast_tracker
            tracker = get_forecast_tracker()
            
            forecast = tracker.get_forecast(forecast_id)
            if not forecast:
                return jsonify({
                    "error": "Not found",
                    "message": f"Forecast '{forecast_id}' not found"
                }), 404
            
            return jsonify({
                "forecast_id": forecast.forecast_id,
                "created_at": forecast.created_at.isoformat(),
                "model_version": forecast.model_version,
                "forecast_start": forecast.forecast_start.isoformat(),
                "forecast_end": forecast.forecast_end.isoformat(),
                "predictions": forecast.predictions
            })
            
        except Exception as e:
            return jsonify({
                "error": "Error retrieving forecast",
                "message": str(e)
            }), 500
    
    # ===========================================
    # STAFFING ENDPOINTS
    # ===========================================
    
    @app.route("/api/v1/staffing", methods=["POST"])
    def calculate_staffing():
        """
        Calculate required staffing.
        
        Request body:
        {
            "workload": [{"timestamp": "...", "calls": 100, "emails": 50, ...}],
            "config": {
                "service_level": 0.8,
                "service_time": 20,
                "avg_handle_time": {...},
                "shrinkage": 0.3
            }
        }
        
        Returns staffing requirements.
        """
        data = request.get_json()
        
        if not data or "workload" not in data:
            return jsonify({
                "error": "Invalid request",
                "message": "Request must include 'workload' data"
            }), 400
        
        try:
            from src.models.capacity import CapacityPlanner
            from src.utils.config import CapacityConfig
            
            # Parse config
            config_data = data.get("config", {})
            config = CapacityConfig(
                service_level_target=config_data.get("service_level", 0.8),
                service_time_seconds=config_data.get("service_time", 20),
                shrinkage_factor=config_data.get("shrinkage", 0.3)
            )
            
            # Parse workload
            workload_df = pd.DataFrame(data["workload"])
            workload_df["timestamp"] = pd.to_datetime(workload_df["timestamp"])
            
            # Calculate staffing
            planner = CapacityPlanner(config)
            staffing = planner.calculate_staffing(workload_df)
            
            return jsonify({
                "staffing": staffing.to_dict(orient="records"),
                "summary": {
                    "total_fte_hours": float(staffing["required_agents"].sum()),
                    "peak_agents": int(staffing["required_agents"].max()),
                    "avg_agents": float(staffing["required_agents"].mean())
                }
            })
            
        except Exception as e:
            return jsonify({
                "error": "Staffing calculation error",
                "message": str(e)
            }), 500
    
    # ===========================================
    # MODEL ENDPOINTS
    # ===========================================
    
    @app.route("/api/v1/models", methods=["GET"])
    def list_models():
        """List available models."""
        try:
            from src.models.model_manager import get_model_manager
            manager = get_model_manager()
            
            models = manager.list_models()
            
            return jsonify({
                "models": [
                    {
                        "model_id": mid,
                        "version": meta.version,
                        "created_at": meta.created_at.isoformat(),
                        "model_type": meta.model_type,
                        "metrics": meta.metrics
                    }
                    for mid, meta in models.items()
                ]
            })
            
        except Exception as e:
            return jsonify({
                "error": "Error listing models",
                "message": str(e)
            }), 500
    
    @app.route("/api/v1/models/<model_id>/activate", methods=["POST"])
    def activate_model(model_id: str):
        """Activate a model version."""
        try:
            from src.models.model_manager import get_model_manager
            manager = get_model_manager()
            
            data = request.get_json() or {}
            version = data.get("version")
            
            if not version:
                return jsonify({
                    "error": "Version required",
                    "message": "Please specify a version to activate"
                }), 400
            
            manager.set_active_version(model_id, version)
            
            return jsonify({
                "success": True,
                "message": f"Activated version {version} for model {model_id}"
            })
            
        except FileNotFoundError as e:
            return jsonify({
                "error": "Not found",
                "message": str(e)
            }), 404
        except Exception as e:
            return jsonify({
                "error": "Error activating model",
                "message": str(e)
            }), 500
    
    # ===========================================
    # DATA ENDPOINTS
    # ===========================================
    
    @app.route("/api/v1/data/summary", methods=["GET"])
    def data_summary():
        """Get summary of loaded data."""
        if app.data is None or len(app.data) == 0:
            return jsonify({
                "loaded": False,
                "message": "No data loaded"
            })
        
        df = app.data
        
        return jsonify({
            "loaded": True,
            "rows": len(df),
            "columns": list(df.columns),
            "date_range": {
                "start": df["timestamp"].min().isoformat() if "timestamp" in df.columns else None,
                "end": df["timestamp"].max().isoformat() if "timestamp" in df.columns else None
            },
            "statistics": {
                col: {
                    "mean": float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                    "std": float(df[col].std()) if pd.api.types.is_numeric_dtype(df[col]) else None
                }
                for col in df.columns if col != "timestamp"
            }
        })
    
    @app.route("/api/v1/data/validate", methods=["POST"])
    def validate_data():
        """Validate uploaded data."""
        if "file" not in request.files:
            return jsonify({
                "error": "No file provided",
                "message": "Please upload a file"
            }), 400
        
        try:
            from src.data.validator import validate_data as run_validation
            
            file = request.files["file"]
            
            # Read file
            if file.filename.endswith(".csv"):
                df = pd.read_csv(file)
            elif file.filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file)
            else:
                return jsonify({
                    "error": "Invalid format",
                    "message": "Please upload CSV or Excel file"
                }), 400
            
            # Validate
            result = run_validation(df)
            
            return jsonify({
                "valid": result.is_valid,
                "issues": [
                    {
                        "severity": issue.severity.value,
                        "code": issue.code,
                        "message": issue.message
                    }
                    for issue in result.issues
                ],
                "summary": result.summary
            })
            
        except Exception as e:
            return jsonify({
                "error": "Validation error",
                "message": str(e)
            }), 500
    
    return app


def run_api(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """
    Run the API server.
    
    Args:
        host: Host to bind to
        port: Port number
        debug: Enable debug mode
    """
    app = create_app()
    
    print(f"Starting API server on http://{host}:{port}")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  GET  /status - System status")
    print("  POST /api/v1/forecast - Generate forecast")
    print("  POST /api/v1/staffing - Calculate staffing")
    print("  GET  /api/v1/models - List models")
    print("  GET  /api/v1/data/summary - Data summary")
    
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_api(debug=True)


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
    
    # ===========================================
    # FILE UPLOAD ENDPOINTS
    # ===========================================
    
    @app.route("/api/v1/data/upload", methods=["POST"])
    def upload_data_file():
        """
        Upload data files to the system.
        
        Supports single or multiple file uploads.
        
        Request:
            - Content-Type: multipart/form-data
            - file: Single file (CSV or Excel)
            - files: Multiple files (alternative)
            - data_type: Optional type hint ("calls", "emails", "outbound", "combined")
            - overwrite: Boolean - overwrite existing files (default: false)
        
        Returns:
        {
            "success": true,
            "uploaded_files": [
                {
                    "filename": "calls_data.csv",
                    "rows": 8760,
                    "columns": ["timestamp", "call_volume"],
                    "data_type": "calls",
                    "column_mapping": {...}
                }
            ],
            "summary": {
                "total_files": 1,
                "total_rows": 8760
            }
        }
        """
        # Check for files in request
        if "file" not in request.files and "files" not in request.files:
            return jsonify({
                "error": "No file provided",
                "message": "Please upload a file using 'file' or 'files' field"
            }), 400
        
        # Get files list
        files = []
        if "file" in request.files:
            files.append(request.files["file"])
        if "files" in request.files:
            files.extend(request.files.getlist("files"))
        
        # Filter empty files
        files = [f for f in files if f.filename]
        
        if not files:
            return jsonify({
                "error": "No valid files",
                "message": "Uploaded files are empty or invalid"
            }), 400
        
        # Get options from form data
        data_type = request.form.get("data_type", "auto")
        overwrite = request.form.get("overwrite", "false").lower() == "true"
        
        # Process each file
        uploaded_files = []
        total_rows = 0
        
        for file in files:
            try:
                result = _process_upload_file(file, data_type, overwrite)
                uploaded_files.append(result)
                total_rows += result.get("rows", 0)
            except Exception as e:
                return jsonify({
                    "error": "File processing error",
                    "message": f"Error processing '{file.filename}': {str(e)}",
                    "failed_file": file.filename
                }), 500
        
        return jsonify({
            "success": True,
            "uploaded_files": uploaded_files,
            "summary": {
                "total_files": len(uploaded_files),
                "total_rows": total_rows
            }
        })
    
    @app.route("/api/v1/data/upload/batch", methods=["POST"])
    def upload_data_batch():
        """
        Upload multiple data files as a batch with specific type assignments.
        
        Request (JSON):
        {
            "files": [
                {
                    "name": "calls_jan.csv",
                    "type": "calls",
                    "content": "base64_encoded_content"
                },
                {
                    "name": "emails_jan.csv",
                    "type": "emails",
                    "content": "base64_encoded_content"
                }
            ],
            "merge_strategy": "append" | "replace" | "merge_by_timestamp"
        }
        
        Returns:
            Processed files summary
        """
        import base64
        import io
        
        data = request.get_json()
        
        if not data or "files" not in data:
            return jsonify({
                "error": "Invalid request",
                "message": "Request must include 'files' array with base64 encoded file content"
            }), 400
        
        merge_strategy = data.get("merge_strategy", "append")
        uploaded_files = []
        
        for file_info in data["files"]:
            try:
                filename = file_info.get("name", "unknown.csv")
                file_type = file_info.get("type", "auto")
                content_b64 = file_info.get("content", "")
                
                # Decode base64 content
                content = base64.b64decode(content_b64)
                
                # Create file-like object
                file_obj = io.BytesIO(content)
                file_obj.name = filename
                
                # Read into dataframe
                if filename.endswith(".csv"):
                    try:
                        df = pd.read_csv(file_obj, sep=';')
                        if len(df.columns) < 2:
                            file_obj.seek(0)
                            df = pd.read_csv(file_obj, sep=',')
                    except:
                        file_obj.seek(0)
                        df = pd.read_csv(file_obj)
                elif filename.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(file_obj)
                else:
                    continue
                
                # Detect column mapping
                column_mapping = _detect_column_mapping(df)
                
                # Save file
                from pathlib import Path
                save_dir = Path("data/raw")
                save_dir.mkdir(parents=True, exist_ok=True)
                
                save_path = save_dir / filename
                df.to_csv(save_path, index=False)
                
                uploaded_files.append({
                    "filename": filename,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "data_type": file_type,
                    "column_mapping": column_mapping,
                    "saved_to": str(save_path)
                })
                
            except Exception as e:
                uploaded_files.append({
                    "filename": file_info.get("name", "unknown"),
                    "error": str(e),
                    "success": False
                })
        
        return jsonify({
            "success": True,
            "merge_strategy": merge_strategy,
            "uploaded_files": uploaded_files,
            "summary": {
                "total_files": len(uploaded_files),
                "successful": sum(1 for f in uploaded_files if "error" not in f),
                "failed": sum(1 for f in uploaded_files if "error" in f)
            }
        })
    
    @app.route("/api/v1/data/files", methods=["GET"])
    def list_data_files():
        """
        List all uploaded data files.
        
        Returns:
        {
            "files": [
                {
                    "filename": "calls_data.csv",
                    "size_bytes": 12345,
                    "modified_at": "2026-01-15T10:30:00Z",
                    "rows": 8760
                }
            ]
        }
        """
        from pathlib import Path
        
        data_dir = Path("data/raw")
        
        if not data_dir.exists():
            return jsonify({
                "files": [],
                "message": "No data directory found"
            })
        
        files = []
        for file_path in data_dir.glob("*.csv"):
            try:
                stat = file_path.stat()
                
                # Try to get row count
                try:
                    with open(file_path, 'r') as f:
                        row_count = sum(1 for _ in f) - 1  # Subtract header
                except:
                    row_count = None
                
                files.append({
                    "filename": file_path.name,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat() + "Z",
                    "rows": row_count
                })
            except Exception:
                continue
        
        # Also check for Excel files
        for ext in ["*.xlsx", "*.xls"]:
            for file_path in data_dir.glob(ext):
                try:
                    stat = file_path.stat()
                    files.append({
                        "filename": file_path.name,
                        "size_bytes": stat.st_size,
                        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat() + "Z",
                        "rows": None  # Can't quickly count Excel rows
                    })
                except:
                    continue
        
        return jsonify({
            "files": files,
            "total": len(files)
        })
    
    @app.route("/api/v1/data/files/<filename>", methods=["DELETE"])
    def delete_data_file(filename: str):
        """Delete a data file."""
        from pathlib import Path
        
        file_path = Path("data/raw") / filename
        
        if not file_path.exists():
            return jsonify({
                "error": "File not found",
                "message": f"File '{filename}' does not exist"
            }), 404
        
        try:
            file_path.unlink()
            return jsonify({
                "success": True,
                "message": f"File '{filename}' deleted"
            })
        except Exception as e:
            return jsonify({
                "error": "Delete error",
                "message": str(e)
            }), 500
    
    @app.route("/api/v1/data/load", methods=["POST"])
    def load_data_into_memory():
        """
        Load uploaded files into memory for processing.
        
        This combines all uploaded data files and prepares them for forecasting.
        
        Request (optional JSON):
        {
            "files": ["calls.csv", "emails.csv"],  # Optional: specific files to load
            "date_range": {
                "start": "2025-01-01",
                "end": "2026-01-01"
            }
        }
        
        Returns:
            Data summary after loading
        """
        try:
            from src.data.loader import DataLoader
            
            data = request.get_json() or {}
            specific_files = data.get("files", None)
            date_range = data.get("date_range", None)
            
            loader = DataLoader()
            loaded_data = loader.load_all()
            
            if not loaded_data:
                return jsonify({
                    "success": False,
                    "error": "No data loaded",
                    "message": "No valid data files found in data/raw directory"
                }), 400
            
            combined = loader.combine_data()
            
            # Apply date filter if specified
            if date_range and "timestamp" in combined.columns:
                combined["timestamp"] = pd.to_datetime(combined["timestamp"])
                if date_range.get("start"):
                    combined = combined[combined["timestamp"] >= date_range["start"]]
                if date_range.get("end"):
                    combined = combined[combined["timestamp"] <= date_range["end"]]
            
            # Store in app context
            app.data = combined
            
            return jsonify({
                "success": True,
                "rows": len(combined),
                "columns": list(combined.columns),
                "date_range": {
                    "start": combined["timestamp"].min().isoformat() if "timestamp" in combined.columns else None,
                    "end": combined["timestamp"].max().isoformat() if "timestamp" in combined.columns else None
                },
                "summary": {
                    col: {
                        "mean": float(combined[col].mean()),
                        "sum": float(combined[col].sum()),
                        "min": float(combined[col].min()),
                        "max": float(combined[col].max())
                    }
                    for col in combined.columns 
                    if pd.api.types.is_numeric_dtype(combined[col])
                }
            })
            
        except Exception as e:
            return jsonify({
                "error": "Load error",
                "message": str(e)
            }), 500
    
    return app


def _process_upload_file(file, data_type: str, overwrite: bool) -> dict:
    """Process a single uploaded file."""
    from pathlib import Path
    
    filename = file.filename
    
    # Read file into dataframe
    if filename.endswith(".csv"):
        try:
            df = pd.read_csv(file, sep=';')
            if len(df.columns) < 2:
                file.seek(0)
                df = pd.read_csv(file, sep=',')
        except:
            file.seek(0)
            df = pd.read_csv(file)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    
    # Detect data type if auto
    detected_type = data_type
    if data_type == "auto":
        detected_type = _detect_data_type(df, filename)
    
    # Detect column mapping
    column_mapping = _detect_column_mapping(df)
    
    # Save file
    save_dir = Path("data/raw")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = save_dir / filename
    
    if save_path.exists() and not overwrite:
        # Generate unique filename
        base = save_path.stem
        ext = save_path.suffix
        counter = 1
        while save_path.exists():
            save_path = save_dir / f"{base}_{counter}{ext}"
            counter += 1
    
    # Save as CSV for consistency
    df.to_csv(save_path, index=False)
    
    return {
        "filename": save_path.name,
        "original_filename": filename,
        "rows": len(df),
        "columns": list(df.columns),
        "data_type": detected_type,
        "column_mapping": column_mapping,
        "saved_to": str(save_path)
    }


def _detect_data_type(df: pd.DataFrame, filename: str) -> str:
    """Detect the type of data based on columns and filename."""
    filename_lower = filename.lower()
    columns_lower = [c.lower() for c in df.columns]
    
    # Check filename hints
    if any(x in filename_lower for x in ["call", "anruf", "inbound"]):
        return "calls"
    if any(x in filename_lower for x in ["email", "mail"]):
        return "emails"
    if any(x in filename_lower for x in ["outbound", "ook", "omk"]):
        return "outbound"
    
    # Check column hints
    if any("call" in c for c in columns_lower):
        return "calls"
    if any("email" in c or "mail" in c for c in columns_lower):
        return "emails"
    if any("outbound" in c or "ook" in c or "omk" in c for c in columns_lower):
        return "outbound"
    
    # Check if it has multiple data types
    has_calls = any("call" in c for c in columns_lower)
    has_emails = any("email" in c or "mail" in c for c in columns_lower)
    has_outbound = any("outbound" in c for c in columns_lower)
    
    if sum([has_calls, has_emails, has_outbound]) > 1:
        return "combined"
    
    return "unknown"


def _detect_column_mapping(df: pd.DataFrame) -> dict:
    """Detect which standard columns are present in the dataframe."""
    mapping = {}
    columns_lower = {c.lower(): c for c in df.columns}
    
    # Timestamp detection
    timestamp_variants = ['timestamp', 'date', 'datetime', 'zeit', 'datum', 'time']
    for variant in timestamp_variants:
        if variant in columns_lower:
            mapping['timestamp'] = columns_lower[variant]
            break
    
    # If no timestamp found, try to detect datetime column
    if 'timestamp' not in mapping:
        for col in df.columns:
            try:
                pd.to_datetime(df[col].head(10))
                mapping['timestamp'] = col
                break
            except:
                pass
    
    # Call volume detection
    call_variants = ['call_volume', 'calls', 'call', 'anrufe', 'inbound_calls', 'call_count']
    for variant in call_variants:
        if variant in columns_lower:
            mapping['call_volume'] = columns_lower[variant]
            break
    
    # Email count detection
    email_variants = ['email_count', 'emails', 'email', 'e-mail', 'mail_count', 'mails']
    for variant in email_variants:
        if variant in columns_lower:
            mapping['email_count'] = columns_lower[variant]
            break
    
    # Outbound detection
    outbound_variants = {
        'outbound_ook': ['outbound_ook', 'ook', 'order_confirmation'],
        'outbound_omk': ['outbound_omk', 'omk', 'customer_contact'],
        'outbound_nb': ['outbound_nb', 'nb', 'follow_up', 'nachbearbeitung'],
        'outbound_total': ['outbound_total', 'outbound', 'total_outbound']
    }
    
    for key, variants in outbound_variants.items():
        for variant in variants:
            if variant in columns_lower:
                mapping[key] = columns_lower[variant]
                break
    
    return mapping


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
    print("\nAvailable endpoints:")
    print("\n  Health & Status:")
    print("    GET  /health - Health check")
    print("    GET  /status - System status")
    print("\n  Forecasting:")
    print("    POST /api/v1/forecast - Generate forecast")
    print("    GET  /api/v1/forecast/{id} - Get saved forecast")
    print("    POST /api/v1/staffing - Calculate staffing")
    print("\n  Models:")
    print("    GET  /api/v1/models - List models")
    print("    POST /api/v1/models/{id}/activate - Activate model version")
    print("\n  Data:")
    print("    GET  /api/v1/data/summary - Data summary")
    print("    POST /api/v1/data/validate - Validate data file")
    print("    POST /api/v1/data/upload - Upload data files (multipart/form-data)")
    print("    POST /api/v1/data/upload/batch - Batch upload (base64 JSON)")
    print("    GET  /api/v1/data/files - List uploaded files")
    print("    DELETE /api/v1/data/files/{filename} - Delete file")
    print("    POST /api/v1/data/load - Load data into memory")
    print()
    
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    import os
    
    # Railway uses PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "development") == "development"
    
    run_api(host="0.0.0.0", port=port, debug=debug)


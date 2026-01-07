# Workforce Planning ML System

A production-ready machine learning system for forecasting customer service workload and planning agent capacity.

## Features

### Core Functionality
- **Workload Forecasting**: Predict hourly volumes for calls, emails, and outbound tasks
- **Confidence Intervals**: Statistical uncertainty bounds on predictions
- **Capacity Planning**: Calculate required agents using Erlang-C formula
- **Interactive Dashboard**: Streamlit-based visualization and analysis
- **Multi-format Export**: Download forecasts and staffing plans (Excel & CSV)

### Production Features
- **Authentication**: Role-based access control (Admin / Dienstleister)
- **Session Management**: Secure sessions with timeout
- **Model Versioning**: Track and manage multiple model versions
- **Forecast Tracking**: Compare predictions against actual outcomes
- **Data Validation**: Automated data quality checks
- **Audit Logging**: Track user actions and system events
- **REST API**: External integration endpoints

## Project Structure

```
KapaPlanung_ML/
├── data/
│   ├── raw/                    # Place your CSV/Excel files here
│   └── processed/              # Cleaned and transformed data
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py           # REST API endpoints
│   ├── data/
│   │   ├── loader.py           # Data loading from CSV/Excel
│   │   ├── preprocessor.py     # Cleaning and feature engineering
│   │   └── validator.py        # Data validation
│   ├── models/
│   │   ├── forecaster.py       # Time series forecasting models
│   │   ├── capacity.py         # Agent capacity calculations
│   │   ├── model_manager.py    # Model versioning
│   │   └── forecast_tracker.py # Actual vs forecast tracking
│   ├── utils/
│   │   ├── config.py           # Configuration and constants
│   │   ├── settings.py         # Environment-based settings
│   │   ├── logging_config.py   # Structured logging
│   │   ├── exceptions.py       # Custom exceptions
│   │   ├── ui_components.py    # Reusable UI components
│   │   └── export.py           # Export utilities
│   ├── auth.py                 # Authentication system
│   └── app.py                  # Streamlit dashboard
├── notebooks/
│   └── exploration.ipynb       # Data exploration
├── outputs/                    # Generated forecasts and exports
├── models/                     # Saved model files
├── logs/                       # Application logs
├── config.example.env          # Example environment config
├── requirements.txt
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### Environment Variables

Copy `config.example.env` to `.env` and configure:

```bash
# Application
APP_ENV=production
SECRET_KEY=your-secure-secret-key

# Authentication
SESSION_TIMEOUT_MINUTES=60
MAX_LOGIN_ATTEMPTS=5

# Paths
DATA_RAW_PATH=data/raw
MODELS_PATH=models

# Logging
LOG_LEVEL=INFO
```

### Application Configuration

Edit `src/utils/config.py` to customize:

- **Average Handling Times**: Set AHT for each task type
- **Service Level Targets**: Define SLA requirements
- **Shrinkage Factor**: Account for breaks, meetings, etc.
- **Working Hours**: Define operational hours
- **Forecast Parameters**: Adjust model hyperparameters

## Data Format

Place your historical data files in `data/raw/`. Expected formats:

### Calls Data
| Column | Description |
|--------|-------------|
| timestamp | Date and time of the call |
| call_volume | Number of calls (if aggregated) |

### Emails Data
| Column | Description |
|--------|-------------|
| timestamp | Date and time |
| email_count | Number of emails |

### Outbound Data
| Column | Description |
|--------|-------------|
| timestamp | Date and time |
| type | Task type (OOK, OMK, NB) |
| count | Number of tasks |

## Usage

### Run the Dashboard

```bash
streamlit run src/app.py
```

Access at: http://localhost:8501

### Default Login Credentials

| Role | Username | Password |
|------|----------|----------|
| Admin | admin | admin123 |
| Dienstleister | dienstleister | service123 |

### Run the API (Optional)

```bash
python -m src.api.routes
```

Access at: http://localhost:5000

### Train Models (Programmatic)

```python
from src.data.loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.models.forecaster import WorkloadForecaster
from src.models.model_manager import get_model_manager

# Load and preprocess data
loader = DataLoader()
data = loader.load_all()

preprocessor = Preprocessor()
features = preprocessor.fit_transform(data)

# Train forecaster
forecaster = WorkloadForecaster()
metrics = forecaster.fit(features)

# Save with versioning
manager = get_model_manager()
manager.save_model(
    model=forecaster,
    model_id="workload_forecaster",
    model_type="HistGradientBoostingRegressor",
    created_by="system",
    metrics=metrics
)

# Generate forecast with confidence intervals
result = forecaster.forecast_horizon(
    last_known_data=data,
    horizon_hours=168,  # 7 days
    confidence_level=0.95
)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| GET | /status | System status |
| POST | /api/v1/forecast | Generate forecast |
| GET | /api/v1/forecast/{id} | Get saved forecast |
| POST | /api/v1/staffing | Calculate staffing |
| GET | /api/v1/models | List models |
| POST | /api/v1/models/{id}/activate | Activate model version |
| GET | /api/v1/data/summary | Data summary |
| POST | /api/v1/data/validate | Validate data |

## Task Types

- **Inbound Calls**: Customer incoming calls
- **E-Mails**: Customer email inquiries
- **Outbound OOK**: Outbound calls - Order Confirmation
- **Outbound OMK**: Outbound calls - Customer Contact
- **Outbound NB**: Outbound calls - Follow-up

## Security Considerations

For production deployment:

1. **Change default passwords** in `src/auth.py`
2. **Set a strong SECRET_KEY** in environment
3. **Enable HTTPS** via reverse proxy (nginx, Caddy)
4. **Configure firewall** to restrict access
5. **Set up proper database** instead of file-based storage
6. **Review and customize** session timeout settings

## License

Internal use only - CHECK24


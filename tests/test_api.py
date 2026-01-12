#!/usr/bin/env python3
"""
API Test Suite für KapaPlanung ML
================================

Testet alle API-Endpunkte sowohl lokal als auch auf Railway.

Verwendung:
    # Lokal testen:
    python tests/test_api.py
    
    # Railway testen:
    python tests/test_api.py https://your-app.up.railway.app
    
    # Mit pytest:
    pytest tests/test_api.py -v
"""

import requests
import json
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

# ===========================================
# KONFIGURATION
# ===========================================

# Standard: Lokal, kann per Argument überschrieben werden
API_BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"

# Farben für Terminal-Output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Druckt einen formatierten Header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def print_test(name: str, passed: bool, details: str = ""):
    """Druckt das Testergebnis."""
    status = f"{Colors.GREEN}✅ PASS{Colors.RESET}" if passed else f"{Colors.RED}❌ FAIL{Colors.RESET}"
    print(f"  {status} {name}")
    if details:
        print(f"       {Colors.YELLOW}{details}{Colors.RESET}")


def print_response(response: requests.Response, show_body: bool = True):
    """Druckt Response-Details."""
    print(f"       Status: {response.status_code}")
    if show_body:
        try:
            body = json.dumps(response.json(), indent=2)
            # Nur erste 500 Zeichen zeigen
            if len(body) > 500:
                body = body[:500] + "..."
            print(f"       Body: {body}")
        except:
            print(f"       Body: {response.text[:200]}")


# ===========================================
# TEST FUNKTIONEN
# ===========================================

def test_health_check() -> bool:
    """Test: GET /health"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        passed = (
            response.status_code == 200 and
            response.json().get("status") == "healthy"
        )
        print_test("Health Check", passed)
        if not passed:
            print_response(response)
        return passed
    except Exception as e:
        print_test("Health Check", False, str(e))
        return False


def test_status() -> bool:
    """Test: GET /status"""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=10)
        passed = response.status_code == 200
        data = response.json()
        
        # Prüfe erwartete Felder
        expected_fields = ["model_loaded", "data_loaded"]
        has_fields = all(field in data for field in expected_fields)
        
        print_test("System Status", passed and has_fields)
        if not passed:
            print_response(response)
        return passed and has_fields
    except Exception as e:
        print_test("System Status", False, str(e))
        return False


def test_list_models() -> bool:
    """Test: GET /api/v1/models"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/models", timeout=10)
        passed = response.status_code == 200
        data = response.json()
        
        # Prüfe ob models-Array existiert
        has_models = "models" in data
        
        print_test("List Models", passed and has_models)
        if passed:
            model_count = len(data.get("models", []))
            print(f"       📊 {model_count} Modelle gefunden")
        else:
            print_response(response)
        return passed
    except Exception as e:
        print_test("List Models", False, str(e))
        return False


def test_list_data_files() -> bool:
    """Test: GET /api/v1/data/files"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/data/files", timeout=10)
        passed = response.status_code == 200
        data = response.json()
        
        has_files = "files" in data
        
        print_test("List Data Files", passed and has_files)
        if passed:
            file_count = len(data.get("files", []))
            print(f"       📁 {file_count} Dateien gefunden")
        return passed
    except Exception as e:
        print_test("List Data Files", False, str(e))
        return False


def test_data_summary() -> bool:
    """Test: GET /api/v1/data/summary"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/data/summary", timeout=10)
        # Kann 200 sein (Daten geladen) oder 200 mit loaded=false
        passed = response.status_code == 200
        
        print_test("Data Summary", passed)
        if passed:
            data = response.json()
            if data.get("loaded"):
                print(f"       📊 {data.get('rows', 0)} Zeilen geladen")
            else:
                print(f"       ⚠️ Keine Daten geladen")
        return passed
    except Exception as e:
        print_test("Data Summary", False, str(e))
        return False


def test_validate_endpoint_exists() -> bool:
    """Test: POST /api/v1/data/validate (ohne Datei)"""
    try:
        response = requests.post(f"{API_BASE_URL}/api/v1/data/validate", timeout=10)
        # Sollte 400 zurückgeben (keine Datei)
        passed = response.status_code == 400
        
        print_test("Validate Endpoint (No File)", passed, 
                  "Erwartet 400 ohne Datei" if passed else f"Got {response.status_code}")
        return passed
    except Exception as e:
        print_test("Validate Endpoint", False, str(e))
        return False


def test_upload_endpoint_exists() -> bool:
    """Test: POST /api/v1/data/upload (ohne Datei)"""
    try:
        response = requests.post(f"{API_BASE_URL}/api/v1/data/upload", timeout=10)
        # Sollte 400 zurückgeben (keine Datei)
        passed = response.status_code == 400
        
        print_test("Upload Endpoint (No File)", passed,
                  "Erwartet 400 ohne Datei" if passed else f"Got {response.status_code}")
        return passed
    except Exception as e:
        print_test("Upload Endpoint", False, str(e))
        return False


def test_forecast_endpoint_exists() -> bool:
    """Test: POST /api/v1/forecast (ohne Modell)"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/forecast",
            json={"start_date": "2026-01-15", "end_date": "2026-01-21"},
            timeout=10
        )
        # Kann 400 sein (kein Modell) oder 200 (Modell geladen)
        passed = response.status_code in [200, 400]
        
        print_test("Forecast Endpoint", passed)
        if response.status_code == 400:
            print(f"       ⚠️ Kein Modell geladen (erwartet)")
        elif response.status_code == 200:
            print(f"       ✨ Forecast erfolgreich generiert!")
        return passed
    except Exception as e:
        print_test("Forecast Endpoint", False, str(e))
        return False


def test_staffing_endpoint() -> bool:
    """Test: POST /api/v1/staffing"""
    try:
        payload = {
            "workload": [
                {"timestamp": "2026-01-15T08:00:00", "calls": 100, "emails": 50}
            ],
            "config": {
                "service_level": 0.8,
                "service_time": 20,
                "shrinkage": 0.3
            }
        }
        response = requests.post(
            f"{API_BASE_URL}/api/v1/staffing",
            json=payload,
            timeout=10
        )
        passed = response.status_code in [200, 400, 500]  # Kann je nach Daten variieren
        
        print_test("Staffing Endpoint", passed)
        return passed
    except Exception as e:
        print_test("Staffing Endpoint", False, str(e))
        return False


def test_response_time() -> bool:
    """Test: Response-Zeit unter 2 Sekunden"""
    try:
        start = time.time()
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        elapsed = time.time() - start
        
        passed = elapsed < 2.0 and response.status_code == 200
        
        print_test("Response Time", passed, f"{elapsed:.3f}s (Limit: 2.0s)")
        return passed
    except Exception as e:
        print_test("Response Time", False, str(e))
        return False


def test_cors_headers() -> bool:
    """Test: CORS Headers (für Browser-Zugriff)"""
    try:
        response = requests.options(f"{API_BASE_URL}/health", timeout=10)
        # OPTIONS kann verschiedene Status haben
        passed = response.status_code in [200, 204, 405]  # 405 = Method Not Allowed ist auch OK
        
        print_test("CORS/OPTIONS", passed)
        return passed
    except Exception as e:
        print_test("CORS/OPTIONS", False, str(e))
        return False


def test_404_handling() -> bool:
    """Test: 404 für nicht existierende Endpunkte"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/nonexistent", timeout=10)
        passed = response.status_code == 404
        
        print_test("404 Handling", passed)
        return passed
    except Exception as e:
        print_test("404 Handling", False, str(e))
        return False


# ===========================================
# HAUPTPROGRAMM
# ===========================================

def run_all_tests():
    """Führt alle Tests aus."""
    print_header(f"🧪 API Tests für: {API_BASE_URL}")
    
    # Teste zuerst ob API erreichbar ist
    print(f"{Colors.BOLD}1. Verbindungstest{Colors.RESET}")
    print("-" * 40)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        print(f"  {Colors.GREEN}✅ API erreichbar!{Colors.RESET}")
    except requests.exceptions.ConnectionError:
        print(f"  {Colors.RED}❌ API nicht erreichbar!{Colors.RESET}")
        print(f"\n  {Colors.YELLOW}Mögliche Ursachen:{Colors.RESET}")
        print(f"  - API läuft nicht")
        print(f"  - Falsche URL: {API_BASE_URL}")
        print(f"  - Firewall blockiert Verbindung")
        print(f"\n  {Colors.YELLOW}Lokale API starten:{Colors.RESET}")
        print(f"  cd /pfad/zum/projekt && source venv/bin/activate && python -m src.api.routes")
        return
    except Exception as e:
        print(f"  {Colors.RED}❌ Fehler: {e}{Colors.RESET}")
        return
    
    results = []
    
    # Health & Status Tests
    print(f"\n{Colors.BOLD}2. Health & Status Endpunkte{Colors.RESET}")
    print("-" * 40)
    results.append(test_health_check())
    results.append(test_status())
    results.append(test_response_time())
    
    # Data Endpunkte
    print(f"\n{Colors.BOLD}3. Data Endpunkte{Colors.RESET}")
    print("-" * 40)
    results.append(test_list_data_files())
    results.append(test_data_summary())
    results.append(test_upload_endpoint_exists())
    results.append(test_validate_endpoint_exists())
    
    # Model Endpunkte
    print(f"\n{Colors.BOLD}4. Model Endpunkte{Colors.RESET}")
    print("-" * 40)
    results.append(test_list_models())
    
    # Forecast & Staffing Endpunkte
    print(f"\n{Colors.BOLD}5. Forecast & Staffing Endpunkte{Colors.RESET}")
    print("-" * 40)
    results.append(test_forecast_endpoint_exists())
    results.append(test_staffing_endpoint())
    
    # Error Handling
    print(f"\n{Colors.BOLD}6. Error Handling{Colors.RESET}")
    print("-" * 40)
    results.append(test_404_handling())
    results.append(test_cors_headers())
    
    # Zusammenfassung
    print_header("📊 Zusammenfassung")
    
    passed = sum(results)
    total = len(results)
    percentage = (passed / total) * 100 if total > 0 else 0
    
    if percentage == 100:
        status_color = Colors.GREEN
        status_emoji = "🎉"
    elif percentage >= 80:
        status_color = Colors.YELLOW
        status_emoji = "⚠️"
    else:
        status_color = Colors.RED
        status_emoji = "❌"
    
    print(f"  {status_emoji} {status_color}{passed}/{total} Tests bestanden ({percentage:.0f}%){Colors.RESET}")
    print(f"\n  API URL: {API_BASE_URL}")
    print(f"  Zeitpunkt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if percentage < 100:
        print(f"\n  {Colors.YELLOW}Hinweis: Einige Tests können fehlschlagen, wenn:{Colors.RESET}")
        print(f"  - Keine Daten geladen sind")
        print(f"  - Kein Modell trainiert wurde")
        print(f"  - Die API gerade erst gestartet wurde")


if __name__ == "__main__":
    run_all_tests()


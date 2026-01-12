#!/bin/bash
#
# Schnelle API-Tests via cURL
# ============================
#
# Verwendung:
#   ./tests/test_api.sh                          # Testet localhost:5000
#   ./tests/test_api.sh https://your.railway.app # Testet Railway
#

API_URL="${1:-http://localhost:5000}"

echo "========================================"
echo "🧪 API Quick Tests"
echo "========================================"
echo "URL: $API_URL"
echo ""

# Farben
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test Funktion
test_endpoint() {
    local name="$1"
    local method="$2"
    local endpoint="$3"
    local expected_status="$4"
    local data="$5"
    
    if [ "$method" == "GET" ]; then
        response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL$endpoint" 2>/dev/null)
    else
        response=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/json" -d "$data" "$API_URL$endpoint" 2>/dev/null)
    fi
    
    if [ "$response" == "$expected_status" ]; then
        echo -e "${GREEN}✅ $name${NC} (Status: $response)"
    else
        echo -e "${RED}❌ $name${NC} (Expected: $expected_status, Got: $response)"
    fi
}

echo "1. Health & Status"
echo "----------------------------------------"
test_endpoint "Health Check" "GET" "/health" "200"
test_endpoint "System Status" "GET" "/status" "200"

echo ""
echo "2. Data Endpoints"
echo "----------------------------------------"
test_endpoint "List Files" "GET" "/api/v1/data/files" "200"
test_endpoint "Data Summary" "GET" "/api/v1/data/summary" "200"

echo ""
echo "3. Model Endpoints"
echo "----------------------------------------"
test_endpoint "List Models" "GET" "/api/v1/models" "200"

echo ""
echo "4. Response Beispiele"
echo "----------------------------------------"
echo ""
echo "📋 Health Response:"
curl -s "$API_URL/health" | head -c 200
echo ""
echo ""
echo "📋 Status Response:"
curl -s "$API_URL/status" | head -c 300
echo ""
echo ""

echo "========================================"
echo "✅ Tests abgeschlossen"
echo "========================================"


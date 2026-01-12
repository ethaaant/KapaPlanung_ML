# 🚀 Deployment Guide

Dieses Projekt besteht aus zwei Diensten:
1. **Streamlit Dashboard** → Streamlit Cloud
2. **REST API** → Railway

---

## 📊 Dashboard auf Streamlit Cloud (bereits deployed)

URL: https://kapaplanung-ml.streamlit.app

---

## 🔌 API auf Railway deployen

### Voraussetzungen
- GitHub Account (Repository bereits vorhanden)
- Railway Account (kostenlos: https://railway.app)

### Schritt-für-Schritt Anleitung

#### 1. Railway Account erstellen
1. Gehe zu https://railway.app
2. Klicke auf "Login" → "Login with GitHub"
3. Autorisiere Railway für GitHub

#### 2. Neues Projekt erstellen
1. Klicke auf "New Project"
2. Wähle "Deploy from GitHub repo"
3. Wähle das Repository: `ethaaant/KapaPlanung_ML`
4. Railway erkennt automatisch die `railway.json` Konfiguration

#### 3. Umgebungsvariablen setzen (optional)
In Railway Dashboard → Variables:
```
FLASK_ENV=production
SECRET_KEY=your-secure-secret-key
```

#### 4. Deploy starten
- Railway startet automatisch den Build
- Nach ~2-3 Minuten ist die API online
- Du erhältst eine URL wie: `https://kapaplanung-ml-production.up.railway.app`

#### 5. Domain anpassen (optional)
1. Gehe zu Settings → Domains
2. Klicke "Generate Domain" für eine Railway-Domain
3. Oder füge eine Custom Domain hinzu

---

## 🧪 API testen

Nach dem Deployment:

```bash
# Ersetze YOUR-RAILWAY-URL mit deiner URL
export API_URL="https://kapaplanung-ml-production.up.railway.app"

# Health Check
curl $API_URL/health

# Status
curl $API_URL/status

# Modelle auflisten
curl $API_URL/api/v1/models

# Dateien auflisten
curl $API_URL/api/v1/data/files
```

---

## 📁 Projekt-Struktur für Deployment

```
KapaPlanung_ML/
├── Dockerfile           # Für Streamlit (Heroku/andere)
├── Dockerfile.api       # Für Flask API (Railway)
├── Procfile            # Für Streamlit (Heroku)
├── railway.json        # Railway Konfiguration
├── requirements.txt    # Python Abhängigkeiten
└── src/
    ├── app.py          # Streamlit Dashboard
    └── api/
        └── routes.py   # Flask API
```

---

## 🔄 Automatische Deploys

- **Streamlit Cloud**: Automatisch bei Push zu `main`
- **Railway**: Automatisch bei Push zu `main`

---

## 💰 Kosten

| Dienst | Free Tier |
|--------|-----------|
| Streamlit Cloud | Unbegrenzt für öffentliche Repos |
| Railway | $5/Monat Guthaben (~500h Laufzeit) |

---

## 🛠️ Troubleshooting

### API startet nicht
- Prüfe die Logs in Railway Dashboard
- Stelle sicher, dass `flask` in `requirements.txt` ist

### Cold Start langsam
- Railway hält Container aktiv (kein Sleep wie bei Render)
- Erster Request nach Deploy kann 10-20s dauern

### Port-Fehler
- Railway setzt automatisch die `PORT` Umgebungsvariable
- Die API liest diese automatisch

---

## 📞 Support

Bei Problemen:
- Railway Docs: https://docs.railway.app
- Streamlit Docs: https://docs.streamlit.io


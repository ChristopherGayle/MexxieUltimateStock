# Deploy Guide (Backend + PWA Frontend)

## 1) Backend deployment (Render)

This repo now includes:
- `Dockerfile`
- `requirements.txt`
- `render.yaml`

### Steps
1. Push this project to your GitHub repo.
2. In Render, create a new Blueprint service from your repo (it will read `render.yaml`).
3. Set env vars in Render:
   - `QUESTDB_HOST` (required)
   - `QUESTDB_PORT` (default `8812`)
   - Optional CAPE overrides:
     - `CAPE_EUROPE`
     - `CAPE_ASIA`
     - `CAPE_AFRICA`
     - `CAPE_WORLDWIDE`
4. Deploy.

Your backend URL will look like:
- `https://<your-service>.onrender.com`

## 2) Frontend deployment (GitHub Pages)

The frontend file is:
- `MexxUltimateScreenerProd-v3.html`

PWA assets added:
- `manifest.webmanifest`
- `sw.js`
- `icons/icon.svg`

### Serve as root page
Either rename:
- `MexxUltimateScreenerProd-v3.html` -> `index.html`

Or keep name and link to it from your `index.html`.

## 3) Point frontend to backend

Set this once in browser console:
```js
localStorage.setItem('mexx_api_base', 'https://<your-service>.onrender.com');
location.reload();
```

If not set:
- localhost uses `http://localhost:8000`
- hosted pages default to same origin (`window.location.origin`)

## 4) Verify

- CAPE API: `https://<your-service>.onrender.com/api/cape`
- Stocks API: `https://<your-service>.onrender.com/api/stocks?region=US&sector=All&limit=20`
- Backtest API: `https://<your-service>.onrender.com/api/backtest?region=US&sector=All&start_date=2025-12-01&end_date=2026-02-28&horizon_days=21&rebalance_days=7&quantiles=5&min_universe=30`

## 5) Install as PWA

- Open the hosted frontend over HTTPS.
- Use browser install prompt/menu (`Install app`).

